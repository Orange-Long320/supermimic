"""Prepare a dense rest-start diagnostic with source-confirmed planted feet.

This is an offline kinematic prefix, never training approval or a full climb.
Arm joints follow the smooth source exactly; root, legs and waist satisfy
explicit foot-pose equalities and the original 0.9 joint bounds.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['reference', 'calibration', 'model', 'support', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--seconds', type=float, default=1.2)
    p.add_argument('--blend-seconds', type=float, default=2 / 3)
    a = p.parse_args()
    if a.out.exists() or not 0 < a.blend_seconds < a.seconds:
        p.error('New output and positive bounded blend required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Run in the dedicated server CPU environment')
    import mujoco
    import numpy as np
    from scipy.interpolate import CubicSpline
    from scipy.optimize import Bounds, minimize
    from scipy.spatial.transform import Rotation, RotationSpline
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra', check_assets=True)
    from supermimic.robots.dimensions import PROFILE
    from supermimic.data.physics_motion import differentiate_qpos

    m = mujoco.MjModel.from_binary_path(str(a.model))
    if (m.nq, m.nv, m.nu) != (38, 37, 31):
        raise ValueError('Original native actuated X2 required')
    with np.load(a.reference) as z:
        source = z['qpos'].copy()
        fps = float(np.asarray(z['fps']).ravel()[0])
    with np.load(a.calibration) as z:
        entry = z['qpos'].copy()
    support = json.loads(a.support.read_text())
    for side in ['left', 'right']:
        if not support['enabled'] or not any(start == 0 and end >= a.seconds * fps
                                             for start, end, _ in support['windows'][side]):
            raise ValueError('Both feet must remain source-confirmed supports for this prefix')
    if a.seconds * fps >= len(source) - 1:
        raise ValueError('Prefix exceeds original source')
    names = [m.joint(j).name.removeprefix('robot/') for j in range(1, 32)]
    if names != PROFILE['joint_names']:
        raise ValueError('Unexpected joint order')
    joints = [j + 1 for j, name in enumerate(names)
              if any(part in name for part in ['hip_', 'knee_', 'ankle_', 'waist_'])]
    dofs = np.r_[np.arange(6), m.jnt_dofadr[joints]].astype(int)
    qpcols = m.jnt_qposadr[joints]
    middle = m.jnt_range[1:].mean(axis=1)
    half = np.diff(m.jnt_range[1:], axis=1).ravel() * .45
    lower, upper = middle - half, middle + half
    times = np.arange(len(source)) / fps
    numeric = CubicSpline(times, source[:, np.r_[0:3, 7:38]], axis=0)
    orientation = RotationSpline(times, Rotation.from_quat(source[:, [4, 5, 6, 3]]))
    rotation_delta = (Rotation.from_quat(entry[[4, 5, 6, 3]]) * orientation(0).inv()).as_rotvec()
    numeric_delta = entry[np.r_[0:3, 7:38]] - numeric(0)
    d = mujoco.MjData(m)
    d.qpos[:] = entry
    mujoco.mj_forward(m, d)
    feet = [m.body('robot/' + side + '_ankle_roll_link').id for side in ['left', 'right']]
    anchors = [(d.xpos[b].copy(), Rotation.from_matrix(d.xmat[b].reshape(3, 3))) for b in feet]
    a.out.mkdir(parents=True)
    (a.out / 'as-run.py').write_bytes(Path(__file__).read_bytes())
    rows, poses = [], []
    dt = float(m.opt.timestep)
    velocity = np.zeros(m.nv)
    for index, t in enumerate(np.arange(round(a.seconds / dt) + 1) * dt):
        u = min(t / a.blend_seconds, 1.)
        # Local phase has zero first/second derivatives at entry and joins
        # the original clock with phase'=1, phase''=0. No global time scaling.
        phase = a.blend_seconds * (6*u**3 - 8*u**4 + 3*u**5) if t < a.blend_seconds else t
        weight = 1 - (10*u**3 - 15*u**4 + 6*u**5)
        base = np.zeros(m.nq)
        base[np.r_[0:3, 7:38]] = numeric(phase) + weight * numeric_delta
        base[3:7] = (Rotation.from_rotvec(weight * rotation_delta) * orientation(phase)).as_quat()[[3, 0, 1, 2]]
        if index == 0:
            np.testing.assert_allclose(base, entry, atol=1e-12)
            q = entry.copy()
            error = 0.
            success = True
        else:
            lo = np.r_[np.full(3, -.15), np.full(3, -.3), lower[joints_to_indices(joints)] - base[qpcols]]
            hi = np.r_[np.full(3, .15), np.full(3, .3), upper[joints_to_indices(joints)] - base[qpcols]]
            if (lo >= hi).any():
                raise ValueError('Empty original joint domain')

            def pose(x):
                velocity[:] = 0.
                velocity[dofs] = x
                q = base.copy()
                mujoco.mj_integratePos(m, q, velocity, 1.)
                return q

            def constraints(x):
                d.qpos[:] = pose(x)
                mujoco.mj_kinematics(m, d)
                return np.concatenate([np.r_[d.xpos[b] - position,
                    (Rotation.from_matrix(d.xmat[b].reshape(3, 3)) * rotation.inv()).as_rotvec()]
                    for b, (position, rotation) in zip(feet, anchors)])

            initial = np.zeros(m.nv)
            mujoco.mj_differentiatePos(m, initial, 1., base, poses[-1])
            initial = np.clip(initial[dofs], lo + 1e-10, hi - 1e-10)
            weights = np.r_[np.full(3, 10.), np.ones(3), np.full(len(joints), .1)]
            fit = minimize(lambda x: .5 * np.dot(weights * x, x), initial,
                jac=lambda x: weights * x, method='SLSQP', bounds=Bounds(lo, hi),
                constraints=[{'type': 'eq', 'fun': constraints}],
                options={'maxiter': 60, 'ftol': 1e-12})
            q, error, success = pose(fit.x), float(abs(constraints(fit.x)).max()), bool(fit.success)
            if error > 1e-7:
                raise RuntimeError(f'Foot equality rejected at {t}s: {error}, {fit.message}')
        soft = float(np.maximum(np.maximum(lower - q[7:], q[7:] - upper), 0).max())
        if soft > 1e-8:
            raise RuntimeError(f'Original0.9 bounds rejected at {t}s: {soft}')
        poses.append(q)
        rows.append({'time_s': float(t), 'foot_pose_error_max': error, 'solver_success': success})
        if index % 40 == 0:
            print(json.dumps(rows[-1]), flush=True)
    poses = np.asarray(poses)
    qvel = differentiate_qpos(m, poses, 1 / dt)
    # The analytic rest boundary is explicit; numeric derivatives are recorded
    # separately to detect excessive discretization/IK noise, never hidden.
    numeric_entry_velocity = qvel[0].copy()
    qvel[0] = 0.
    np.savez_compressed(a.out / 'reference.npz', qpos=poses, qvel=qvel, fps=np.array([1/dt]),
                        source_start_frame=np.array(0))
    speed = np.array([PROFILE['velocity_limit'][n] for n in names])
    report = {'configuration': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [a.reference, a.calibration, a.model, a.support]},
        'source_start_frame': 0, 'frames': len(poses), 'fps': 1/dt,
        'numeric_entry_velocity_max': float(abs(numeric_entry_velocity).max()),
        'joint_speed_ratio': float((abs(qvel[:, 6:]) / speed).max()),
        'arms_follow_smooth_source_exactly': True, 'source_confirmed_foot_poses_fixed': True,
        'maximum_foot_pose_error': max(r['foot_pose_error_max'] for r in rows),
        'complete_climb_verified': False, 'training_approved': False,
        'scope': __doc__, 'rows': rows}
    (a.out / 'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}), flush=True)


def joints_to_indices(joints):
    return [j - 1 for j in joints]


if __name__ == '__main__':
    main()
