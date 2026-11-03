"""Offline fixed-input sensitivity at a recorded X2 handover; never controls hardware.

Geometry, zero-history, and zero-anchor variants are counterfactuals, not a
replay of robot dynamics.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from supermimic.deploy.numpy_policy import NumpyPolicy
from supermimic.deploy.pelvis_state import UrdfPelvisMount, pelvis_angular_velocity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observation', type=Path, required=True)
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('Output already exists')
    saved = json.loads(a.observation.read_text())
    candidate = json.loads((a.release / 'startup-candidate.json').read_text())
    names = candidate['joint_names']
    n, steps = len(names), 10
    if n != 31:
        raise ValueError('Expected 31 named joints')
    obs = np.asarray(saved['obs'], dtype=np.float32)
    goal = np.asarray(saved['goal'], dtype=np.float32)
    default = np.asarray(saved['default'])
    scale = np.asarray(saved['scale'])
    if obs.shape != (steps * (3*n + 6),) or goal.shape != (729,):
        raise ValueError('Unexpected observation/goal dimensions')
    if not np.allclose(default, candidate['action_default']) or not np.allclose(scale, candidate['action_scale']):
        raise ValueError('Recorded default/scale differ from release')
    model_path = a.release / 'numpy-policy-11500.npz'
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    policy = NumpyPolicy(model_path, expected_sha256=model_sha)
    baseline = policy.infer(obs, goal)
    replay_error = float(np.max(np.abs(baseline - np.asarray(saved['action']))))
    if replay_error > 1e-4:
        raise ValueError(f'Recorded action not reproduced: {replay_error}')

    yaw, pitch, roll = (names.index(x) for x in ('waist_yaw_joint', 'waist_pitch_joint', 'waist_roll_joint'))
    waist_idx = [yaw, pitch, roll]
    mount = UrdfPelvisMount(a.release / 'robot-pc2.urdf',
        ('waist_yaw_joint', 'waist_pitch_joint', 'waist_roll_joint'),
        [.0239465, -.000228682, .04171], np.eye(3))
    corrected = obs.copy()
    q = obs[:steps*n].reshape(steps, n) + default
    qd = obs[steps*n:2*steps*n].reshape(steps, n)
    omega = obs[2*steps*n:2*steps*n+steps*3].reshape(steps, 3)
    gravity = obs[2*steps*n+steps*3:2*steps*n+steps*6].reshape(steps, 3)
    corrected_omega = corrected[2*steps*n:2*steps*n+steps*3].reshape(steps, 3)
    corrected_gravity = corrected[2*steps*n+steps*3:2*steps*n+steps*6].reshape(steps, 3)
    for i in range(steps):
        _, rpi, relative_omega = mount.mount(q[i, waist_idx], qd[i, waist_idx])
        corrected_omega[i] = pelvis_angular_velocity(omega[i], rpi, relative_omega)
        corrected_gravity[i] = rpi @ gravity[i]

    zero_previous = obs.copy()
    zero_previous[-steps*n:] = 0.
    combined = corrected.copy()
    combined[-steps*n:] = 0.
    zero_xy_goal = goal.copy()
    zero_xy_goal[-9:-7] = 0.
    lower = np.asarray(candidate['pc2_urdf_lower'])
    upper = np.asarray(candidate['pc2_urdf_upper'])
    variants = {}
    for name, x, g in [('recorded', obs, goal), ('model_pelvis_observation', corrected, goal),
                       ('zero_previous_history', zero_previous, goal), ('both', combined, goal),
                       ('zero_anchor_xy', obs, zero_xy_goal)]:
        action = policy.infer(x, g)
        target = default + scale * action
        excess = np.maximum(lower - target, target - upper)
        indices = np.argsort(-np.abs(target - np.asarray(saved['q'])))[:5]
        variants[name] = {'max_action_change_from_recorded': float(np.max(np.abs(action-baseline))),
            'out_of_range_count': int(np.count_nonzero(excess > 0)),
            'max_range_excess_rad': float(max(0., np.max(excess))),
            'largest_target_minus_measured_rad': {names[i]: float(target[i]-saved['q'][i]) for i in indices}}
    result = {'scope': __doc__, 'recorded_action_max_error': replay_error,
        'model_sha256': model_sha, 'observation_sha256': hashlib.sha256(a.observation.read_bytes()).hexdigest(),
        'goal_held_fixed_except_zero_anchor_xy': True, 'mount_assumption': 'URDF waist FK and zero sensor mount rotation; uncalibrated',
        'zero_history_assumption': 'hypothetical zero previous actions; not recorded policy history',
        'original_gravity_last': gravity[-1].tolist(), 'model_pelvis_gravity_last': corrected_gravity[-1].tolist(),
        'original_omega_last': omega[-1].tolist(), 'model_pelvis_omega_last': corrected_omega[-1].tolist(),
        'variants': variants, 'hardware_control': False}
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
