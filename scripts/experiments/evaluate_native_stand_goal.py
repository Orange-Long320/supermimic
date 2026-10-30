"""Diagnostic motion-goal standing override using the unchanged evaluator and policy.

This is a runtime override experiment, not an accepted or deployable checkpoint.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('evaluator', 'checkpoint', 'npz-dir', 'standing-reference', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--seconds', type=float, default=66.)
    p.add_argument('--num-envs', type=int, default=8)
    a = p.parse_args()
    expected = Path('/home/2025-Long.xh/HHD/venvs/supermimic').resolve()
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if Path(sys.prefix).resolve() != expected or not gpu or len(gpu.split(',')) != 1:
        p.error('Use the dedicated server environment and explicitly select one GPU')
    if a.out.exists() or a.seconds <= 0 or a.num_envs < 2:
        p.error('Require a new output path, positive duration and at least two environments')
    for path in (a.evaluator, a.checkpoint, a.standing_reference, a.npz_dir / 'walk.npz'):
        if not path.is_file():
            p.error(f'Missing input: {path}')
    processes = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-compute-apps=pid',
                                         '--format=csv,noheader'], text=True).strip()
    if processes:
        raise RuntimeError('Requested diagnostic GPU is occupied; no simulation started')
    import numpy as np
    import torch
    from supermimic.envs.mdp.motion_library import MotionLibrary
    from supermimic.kinematics.rotations import heading_yaw, quat_apply, quat_from_yaw, quat_mul
    from supermimic.world_model.heads import anchor_relative_obs
    import supermimic.policy.standing as standing_module

    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    expected_sha = checkpoint.get('standing_reference', {}).get('sha256')
    reference_sha = sha(a.standing_reference)
    if expected_sha != reference_sha:
        raise ValueError('Standing reference does not match checkpoint')
    config = checkpoint['config']
    if not config['env'].get('anchor_obs'):
        raise ValueError('This diagnostic requires the checkpoint anchor observation')
    with np.load(a.standing_reference, allow_pickle=False) as ref:
        for key in ('joint_vel', 'body_lin_vel_w', 'body_ang_vel_w'):
            if not np.isfinite(ref[key]).all() or np.abs(ref[key]).max() > 1e-6:
                raise ValueError(f'Standing reference must have zero {key}')
        for key in ('joint_pos', 'body_pos_w', 'body_quat_w'):
            if not np.isfinite(ref[key]).all() or np.max(np.abs(ref[key] - ref[key][:1])) > 1e-6:
                raise ValueError(f'Standing reference must have static {key}')
        if len(ref['joint_pos']) < config['env']['future_frames']:
            raise ValueError('Standing reference shorter than the goal window')
    spec = importlib.util.spec_from_file_location('native_stand_fixed_evaluator', a.evaluator)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    original = standing_module.standing_action
    library = None
    samples = 0
    delta_sq = delta_max = 0.
    gate_zero_checks = 0
    goal_dimension = None

    def override(policy, observations, command, context=None, disable_feedback=False, disable_moving_feedback=False):
        nonlocal library, samples, delta_sq, delta_max, gate_zero_checks, goal_dimension
        if context is None or disable_feedback or not disable_moving_feedback:
            raise ValueError('Diagnostic requires standing context, standing feedback and disabled movement feedback')
        cmd, env = context.command, context.env
        feedback = context.features(command)
        skill = policy.standing_skill
        gate = skill.gate(command)[:, None]
        base = policy.decoder(policy.encoders['velocity'](observations, command), observations)
        if library is None:
            library = MotionLibrary([str(a.standing_reference)], cmd.library.body_names,
                cmd.library.default_joint_pos, device=observations.device)
        if not (gate > 0).any():
            # This exact action is the original movement path with feedback disabled.
            gate_zero_checks += len(observations)
            return base
        frame = torch.zeros(len(observations), dtype=torch.long, device=observations.device)
        goal = library.goal_window(frame, config['env']['future_frames'])
        reference = library.ref(frame)
        bodies = library.bodies(frame)
        anchor_index = list(library.body_names).index(cmd.cfg.anchor_body_name)
        target_anchor = bodies.pos_w[:, anchor_index]
        target_quat = bodies.quat_w[:, anchor_index]
        rotation = quat_from_yaw(context.anchor_yaw - heading_yaw(reference.root_quat_w))
        target_root = torch.cat((context.anchor_xy, reference.root_pos_w[:, 2:3]), dim=-1)
        target_anchor = target_root + quat_apply(rotation, target_anchor - reference.root_pos_w)
        target_quat = quat_mul(rotation, target_quat)
        robot_anchor = cmd.robot_anchor_pos_w - env.scene.env_origins
        relative = anchor_relative_obs(robot_anchor, cmd.robot_anchor_quat_w, target_anchor, target_quat)
        goal = torch.cat((goal, relative), dim=-1)
        goal_dimension = goal.shape[-1]
        if goal_dimension != config['policy']['goal_dim']:
            raise ValueError('Native goal dimension does not match checkpoint')
        native = policy(observations, goal)
        old = skill(observations, feedback)
        diff = native - old
        samples += diff.numel()
        delta_sq += float(diff.square().sum())
        delta_max = max(delta_max, float(diff.abs().max()))
        return (1 - gate) * base + gate * native

    standing_module.standing_action = override
    try:
        evaluator.main(ckpt=a.checkpoint, out=a.out, seconds=a.seconds, num_envs=a.num_envs,
                       seed=a.seed, npz_dir=a.npz_dir, ieee=True, disable_moving_feedback=True)
    finally:
        standing_module.standing_action = original
    result = json.loads(a.out.read_text())
    result['native_standing_override'] = {
        'mode': 'native_motion_goal_smooth_gate_diagnostic', 'accepted': False,
        'standing_reference_sha256': reference_sha, 'checkpoint_sha256': sha(a.checkpoint),
        'evaluator_sha256': sha(a.evaluator), 'entry_sha256': sha(__file__),
        'gpu': gpu, 'goal_dimension': goal_dimension, 'future_frames': config['env']['future_frames'],
        'same_state_standing_action_difference_rms': (delta_sq / samples) ** .5 if samples else None,
        'same_state_standing_action_difference_max': delta_max if samples else None,
        'same_state_action_elements': samples, 'full_movement_base_rows': gate_zero_checks,
        'scope': 'experimental runtime override; no checkpoint adoption',
    }
    result['scope'] += ' Native motion standing override diagnostic; not an accepted policy.'
    a.out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
