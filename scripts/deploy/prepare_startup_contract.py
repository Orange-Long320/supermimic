"""Prepare an unapproved X2 startup contract from selected files; no robot I/O.

The training model, policy, and references remain unchanged. File intersections
do not establish physical calibration or allowable hardware PD command ranges.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('profile', 'dataset', 'urdf', 'policy', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    if digest(args.policy) != '4424f231fc1896da098b2fca366e8522df6791596638b2b9257ba682520bbe4c':
        raise ValueError('Expected selected 11500 policy')
    manifest = args.dataset/'manifest.json'
    if digest(manifest) != '40a2a31c1b84160f2ea6aa26548eb966f2e3c9ec23e10fa46ec1e9b2532b5b7d':
        raise ValueError('Expected selected reference manifest')
    meta = json.loads(manifest.read_text())
    spec = json.loads(args.profile.read_text())
    names = spec['joint_names']
    xml = Path(spec['xml_path'])
    xml = xml if xml.is_absolute() else args.profile.parent/xml
    model = mujoco.MjModel.from_xml_path(str(xml))
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    if len(names) != 31 or min(ids) < 0:
        raise ValueError('Invalid X2 joint mapping')
    source_bounds = model.jnt_range[ids]
    urdf = ET.parse(args.urdf).getroot()
    device = {j.get('name'): [float(j.find('limit').get(k)) for k in ('lower', 'upper')]
              for j in urdf.findall('joint') if j.get('type') == 'revolute'}
    device_bounds = np.asarray([device[n] for n in names])
    lower = np.maximum(source_bounds[:, 0], device_bounds[:, 0])
    upper = np.minimum(source_bounds[:, 1], device_bounds[:, 1])
    if np.any(lower >= upper):
        raise ValueError('Incompatible model and device-file bounds')
    paths = sorted(args.dataset.glob('*.npz'))
    clips, reference = {}, None
    for path in paths:
        # Bind every input to the selected manifest, not merely its filename.
        entry = next((c for c in meta['clips'] if c['name'] == path.stem), None)
        if entry is None:
            raise ValueError('Unlisted reference: '+path.name)
        expected = entry.get('sha256')
        if expected is None or digest(path) != expected:
            raise ValueError('Reference hash mismatch: '+path.name)
        with np.load(path, allow_pickle=False) as data:
            if data['joint_names'].tolist() != names:
                raise ValueError('Reference joint order mismatch')
            positions = data['joint_pos']
            if positions.ndim != 2 or positions.shape[1] != 31 or not np.isfinite(positions).all():
                raise ValueError('Invalid reference positions')
            bad = (positions < lower) | (positions > upper)
            excess = np.maximum(lower-positions, positions-upper)
            clips[path.stem] = {'sha256': expected, 'frames': len(positions),
                               'frames_outside_file_intersection': int(bad.any(axis=1).sum()),
                               'maximum_excess_rad': float(max(0., excess.max())),
                               'outside_by_joint': {n: int(bad[:, j].sum()) for j, n in enumerate(names) if bad[:, j].any()}}
            if path.stem == 'walk_stop':
                pelvis = data['body_names'].tolist().index('pelvis')
                reference = {'joint_position': positions[-1].astype(float).tolist(),
                             'root_position': data['body_pos_w'][-1, pelvis].astype(float).tolist(),
                             'root_quat_wxyz': data['body_quat_w'][-1, pelvis].astype(float).tolist()}
    if len(paths) != len(meta['clips']) or reference is None:
        raise ValueError('Incomplete selected reference library')
    quat = np.asarray(reference['root_quat_wxyz'])
    reference['gravity'] = Rotation.from_quat(quat[[1, 2, 3, 0]]).inv().apply([0., 0., -1.]).tolist()
    reference['source'] = 'walk_stop final frame, matching baseline SkillPhase.stand'
    raw = np.asarray(reference.pop('joint_position'))
    prepared = np.clip(raw, lower, upper)
    reference['reference_joint_position'] = raw.tolist()
    reference['position'] = prepared.tolist()
    reference['preparation_target_adjustments_rad'] = {n: float(prepared[j]-raw[j])
                                                     for j, n in enumerate(names) if prepared[j] != raw[j]}
    reference['adjustment_scope'] = 'Independent preparation target constrained to file intersection; original policy and reference files are unchanged. Not a measured-state correction or an approved hardware trajectory.'
    reference['position_tolerance'] = None
    reference['max_speed'] = None
    reference['max_gravity_angle'] = None
    default = np.asarray([spec['default_joint_pos'][n] for n in names])
    result = {'scope': __doc__, 'selected_checkpoint': 11500, 'joint_names': names,
              'sha256': {'profile': digest(args.profile), 'training_xml': digest(xml), 'urdf': digest(args.urdf),
                         'policy': digest(args.policy), 'dataset_manifest': digest(manifest)},
              'model_lower': source_bounds[:, 0].tolist(), 'model_upper': source_bounds[:, 1].tolist(),
              'pc2_urdf_lower': device_bounds[:, 0].tolist(), 'pc2_urdf_upper': device_bounds[:, 1].tolist(),
              'measured_lower_candidate': lower.tolist(), 'measured_upper_candidate': upper.tolist(),
              'target_lower': None, 'target_upper': None, 'max_target_step': None,
              'action_default': default.tolist(), 'action_scale': [spec['action_scale'][n] for n in names],
              'startup': reference, 'clips': clips,
              'physical_encoder_calibration_validated': False, 'hardware_command_range_validated': False,
              'startup_motion_validated': False, 'user_deployment_approval': False, 'permit_motor_output': False}
    (args.out/'startup-candidate.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'reference_clips': len(clips),
                      'reference_frames_outside_file_intersection': sum(c['frames_outside_file_intersection'] for c in clips.values()),
                      'maximum_reference_excess_rad': max(c['maximum_excess_rad'] for c in clips.values()),
                      'preparation_target_adjustments_rad': reference['preparation_target_adjustments_rad'],
                      'reference_is_action_default': bool(np.allclose(raw, default)),
                      'permit_motor_output': False}, indent=2))


if __name__ == '__main__':
    main()
