"""Validate a native short preview including the frozen policy's feedback."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from supermimic.data.native_policy_preview import NativePolicyPreview
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.robots.dimensions import PROFILE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--ckpt', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--horizon', type=int, default=2, help='Number of feedback control frames to compare')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    digest = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    config = json.loads((args.trace/'configuration.json').read_text())
    a = config['arguments']
    unsupported = [k for k in ['tail_of', 'landing_ckpt', 'consistent_retiming', 'initial_command_margin',
        'bounded_target_joints', 'joint_stop_margin', 'physics_command_projection'] if a.get(k)]
    if unsupported:
        raise ValueError(f'Preview does not reproduce these controllers: {unsupported}')
    if args.ckpt.name == 'ckpt_latest.pt' or digest(args.ckpt) != config['checkpoint_sha256']:
        raise ValueError('Use the immutable controller that generated the trace')
    source = Path(a['source']); parameters = Path(a['warmstart'])
    if digest(source) != config['source_sha256'] or digest(parameters) != config['warmstart_sha256']:
        raise ValueError('Calibration source/parameters changed')
    policy, policy_config = load_policy_from_checkpoint(args.ckpt, 'cpu')
    policy.requires_grad_(False)
    q0 = torch.tensor([PROFILE['default_joint_pos'][n] for n in PROFILE['joint_names']])
    library = MotionLibrary([source], tuple(PROFILE['tracked_bodies']), q0, device='cpu')
    model = mujoco.MjModel.from_binary_path(str(args.trace/'native-model.mjb'))
    observations = np.load(args.trace/'baseline-observations.npz')
    physics = np.load(args.trace/'baseline-physics.npz')
    trajectory = np.load(args.trace/'baseline-trajectory.npz')
    if not 2<=args.horizon<=len(trajectory['actions']):
        p.error('Preview horizon must span at least two recorded control frames')
    preview = NativePolicyPreview(model, policy, library, config['groups'], np.load(parameters)['parameters'],
        a['takeoff_frame'], a['landing_frame'], observations['reference_frame'][0],
        policy_config['env'].get('anchor_obs', False), policy_config['env'].get('future_frames', 10), a.get('contact_pulses', False))
    computed_frames = [int(preview.frame_at(i*.02)[0]) for i in range(len(observations['policy']))]
    if not np.array_equal(computed_frames, observations['reference_frame']):
        raise ValueError('Preview reference clock differs from native capture')
    actual = mujoco.MjData(model)
    actual_heights = []
    for q, v in zip(physics['qpos'], physics['qvel']):
        actual.qpos[:] = q; actual.qvel[:] = v; mujoco.mj_forward(model, actual)
        mat = actual.geom_xmat.reshape(-1, 3, 3)
        z = (mat[preview.geom_ids, 2]*preview.vertices).sum(-1)+actual.geom_xpos[preview.geom_ids, 2]
        actual_heights.append([z[g].min() for g in preview.geometry_groups])
    actual_heights = np.asarray(actual_heights)
    errors = []; action_errors = []; goal_errors = []
    for start in range(len(trajectory['actions'])-args.horizon+1):
        index = start*preview.ratio
        q = physics['qpos'][index-1] if start else physics['initial_qpos']
        v = physics['qvel'][index-1] if start else physics['initial_qvel']
        predicted = preview.predict(q, v, observations['policy'][start], start*.02, horizon=args.horizon)
        states = np.c_[physics['qpos'][index:index+args.horizon*preview.ratio], physics['qvel'][index:index+args.horizon*preview.ratio]]
        delta = np.abs(predicted['states']-states)
        height_error = np.abs(predicted['heights']-actual_heights[index:index+args.horizon*preview.ratio])
        errors.append(np.c_[delta[:, :model.nq].max(-1), delta[:, model.nq:].max(-1), height_error.max(-1)])
        action_errors.append(np.abs(predicted['actions']-trajectory['actions'][start:start+args.horizon]).max(-1))
        goal_errors.append(np.abs(predicted['goals']-observations['goal'][start:start+args.horizon]).max(-1))
    errors = np.asarray(errors); action_errors = np.asarray(action_errors); goal_errors = np.asarray(goal_errors)
    report = {'scope': 'H control frames from true recorded boundaries, including predicted policy observation history, anchor goal and actions',
        'horizon_control_frames': args.horizon,
        'control_windows': len(errors), 'maximum_action_error_by_control_frame': action_errors.max(0).tolist(),
        'native_model_sha256': digest(args.trace/'native-model.mjb'),
        'maximum_goal_error_by_control_frame': goal_errors.max(0).tolist(),
        'maximum_errors_by_horizon': [{'horizon_s': h*.02,
            'qpos': float(errors[:, :h*preview.ratio, 0].max()), 'qvel': float(errors[:, :h*preview.ratio, 1].max()),
            'geometry_height_m': float(errors[:, :h*preview.ratio, 2].max())}
            for h in sorted({h for h in [1,2,4,8,16,32,args.horizon] if h<=args.horizon})],
        'reference_clock_matched': True, 'approved_for_controller_use': False,
        'input_sha256': {str(f): digest(f) for f in [args.ckpt, source, parameters,
            args.trace/'baseline-physics.npz', args.trace/'baseline-observations.npz']}}
    (args.out/'result.json').write_text(json.dumps(report, indent=2))
    np.savez(args.out/'errors.npz', state_and_geometry=errors, actions=action_errors, goals=goal_errors)
    (args.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    import supermimic.data.native_policy_preview as module
    (args.out/'native_policy_preview.py').write_bytes(Path(module.__file__).read_bytes())
    print(json.dumps(report))


if __name__ == '__main__':
    main()
