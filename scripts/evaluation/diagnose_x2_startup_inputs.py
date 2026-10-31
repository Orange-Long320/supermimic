"""CPU-only factorial probes of recorded X2 inputs; never supplies motor targets.

Substituted features are counterfactual diagnostics, not encoder calibration,
new sensor measurements, executable startup trajectories, or policy acceptance.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def reference_observation(goal):
    frame = np.asarray(goal[:72])
    return np.concatenate([np.tile(term, 10) for term in
                           (frame[:31], frame[31:62], frame[65:68], frame[62:65], np.zeros(31))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('profile', 'frames', 'urdf', 'policy', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--samples', type=int, default=25)
    args = parser.parse_args()
    if args.samples < 2:
        raise ValueError('At least two diagnostic samples required')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    if digest(args.policy) != '4424f231fc1896da098b2fca366e8522df6791596638b2b9257ba682520bbe4c':
        raise ValueError('Wrong policy; expected selected 11500 export')
    spec = json.loads(args.profile.read_text())
    names = spec['joint_names']
    if len(names) != 31:
        raise ValueError('Expected 31-joint X2 contract')
    default = np.asarray([spec['default_joint_pos'][n] for n in names])
    scale = np.asarray([spec['action_scale'][n] for n in names])
    xml = Path(spec['xml_path'])
    model = mujoco.MjModel.from_xml_path(str(xml if xml.is_absolute() else args.profile.parent/xml))
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    if min(ids) < 0:
        raise ValueError('Missing model joints')
    model_bounds = model.jnt_range[ids]
    urdf = ET.parse(args.urdf).getroot()
    bounds = {j.get('name'): [float(j.find('limit').get(k)) for k in ('lower', 'upper')]
              for j in urdf.findall('joint') if j.get('type') == 'revolute'}
    pc2_bounds = np.asarray([bounds[n] for n in names])
    joint_groups = {'legs': list(range(12)), 'waist': list(range(12, 15)), 'upper': list(range(15, 31))}
    factors = [*joint_groups, 'gravity', 'velocities', 'anchor']
    masks = list(itertools.product((False, True), repeat=len(factors)))
    rows = [json.loads(line) for line in args.frames.open()]
    if not rows:
        raise ValueError('Empty input')
    indices = np.unique(np.linspace(0, len(rows)-1, min(args.samples, len(rows)), dtype=int))
    policy = torch.jit.load(str(args.policy), map_location='cpu').eval()
    cases = [{'reference_substitutions': [f for f, on in zip(factors, mask) if on],
              'pc2_bad_frames': 0, 'model_bad_frames': 0, 'max_pc2_excess_rad': 0.,
              'max_action_abs': 0., 'max_target_change_from_measured_rad': 0.}
             for mask in masks]
    baseline_error = 0.
    identity_anchor = np.asarray([0., 0., 0., 1., 0., 0., 1., 0., 0.])
    output_log = (args.out/'probes.jsonl').open('x')
    started = time.perf_counter()
    try:
        for index in indices:
            row = rows[index]
            obs = np.asarray(row['zero']['observation'])
            goal = np.asarray(row['zero']['goal'])
            if obs.shape != (990,) or goal.shape != (729,) or not np.isfinite(np.r_[obs, goal]).all():
                raise ValueError('Invalid saved input dimensions or values')
            if np.any(obs[680:] != 0.):
                raise ValueError('Expected zero-history branch')
            reference = reference_observation(goal)
            observations, goals = [], []
            for mask in masks:
                altered = obs.copy()
                altered_goal = goal.copy()
                for factor, on in zip(factors, mask):
                    if not on:
                        continue
                    if factor in joint_groups:
                        joints = joint_groups[factor]
                        altered[:310].reshape(10, 31)[:, joints] = reference[:310].reshape(10, 31)[:, joints]
                    elif factor == 'gravity':
                        altered[650:680] = reference[650:680]
                    elif factor == 'velocities':
                        altered[310:650] = reference[310:650]
                    else:
                        altered_goal[-9:] = identity_anchor
                observations.append(altered)
                goals.append(altered_goal)
            with torch.inference_mode():
                actions = policy(torch.tensor(np.asarray(observations), dtype=torch.float32),
                                 torch.tensor(np.asarray(goals), dtype=torch.float32)).numpy()
            if actions.shape != (len(masks), 31) or not np.isfinite(actions).all():
                raise ValueError('Invalid probe outputs')
            baseline_error = max(baseline_error, float(np.abs(actions[0]-np.asarray(row['zero']['action'])).max()))
            targets = default+scale*actions
            for case, target, action in zip(cases, targets, actions):
                model_excess = np.maximum(model_bounds[:, 0]-target, target-model_bounds[:, 1])
                pc2_excess = np.maximum(pc2_bounds[:, 0]-target, target-pc2_bounds[:, 1])
                case['pc2_bad_frames'] += int(np.any(pc2_excess > 0.))
                case['model_bad_frames'] += int(np.any(model_excess > 0.))
                case['max_pc2_excess_rad'] = max(case['max_pc2_excess_rad'], float(max(0., pc2_excess.max())))
                case['max_action_abs'] = max(case['max_action_abs'], float(np.abs(action).max()))
                case['max_target_change_from_measured_rad'] = max(case['max_target_change_from_measured_rad'],
                                                                  float(np.abs(target-np.asarray(row['joint_position'])).max()))
            output_log.write(json.dumps({'source_row': int(index), 'tick_ns': row['tick_ns'],
                                         'targets_counterfactual_only': targets.tolist()}, allow_nan=False)+'\n')
    finally:
        output_log.close()
    if baseline_error > 2e-5:
        raise ValueError('Recorded baseline not reproduced: '+str(baseline_error))
    ref_q = np.asarray(rows[0]['zero']['goal'][:31])+default
    measured_q = np.asarray(rows[0]['joint_position'])
    difference = [{'joint': n, 'measured_rad': float(q), 'reference_rad': float(r),
                   'difference_rad': float(q-r), 'difference_deg': float(np.degrees(q-r))}
                  for n, q, r in zip(names, measured_q, ref_q)]
    difference.sort(key=lambda r: abs(r['difference_rad']), reverse=True)
    result = {'scope': __doc__, 'sha256': {k: digest(getattr(args, k)) for k in ('profile', 'frames', 'urdf', 'policy')},
              'selected_checkpoint': 11500, 'source_rows': len(rows), 'sampled_rows': len(indices),
              'sample_indices': indices.tolist(), 'factors': factors, 'cases': cases,
              'recorded_baseline_max_action_error': baseline_error,
              'elapsed_cpu_s': time.perf_counter()-started, 'reference_joint_position_rad': dict(zip(names, ref_q.tolist())),
              'reference_position_outside_pc2_urdf': [names[i] for i in np.flatnonzero((ref_q<pc2_bounds[:, 0])|(ref_q>pc2_bounds[:, 1]))],
              'measured_minus_reference': difference,
              'physical_encoder_calibration_validated': False, 'startup_trajectory_validated': False,
              'motor_deployment_allowed': False}
    (args.out/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'sampled_rows': len(indices), 'recorded_baseline_max_action_error': baseline_error,
                      'passing_counterfactual_cases': [c for c in cases if not c['pc2_bad_frames']],
                      'largest_posture_differences': difference[:8]}, indent=2))


if __name__ == '__main__':
    main()
