"""Offline walking IMU error audit against official-simulator odometry.

No ROS, training, commands, or truth feedback. Replays the current horizontal
Kalman/ZUPT equations and compares native and snapshot double integration.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def rows(path):
    return [json.loads(line) for line in path.open()]


def stamps(data, origin):
    return np.array([(r['stamp'][0] * 10**9 + r['stamp'][1] - origin) * 1e-9
                     for r in data])


def integrate(t, acceleration):
    dt = np.diff(t)[:, None]
    velocity = np.vstack([np.zeros(2), np.cumsum(
        .5 * (acceleration[:-1] + acceleration[1:]) * dt, axis=0)])
    position = np.vstack([np.zeros(2), np.cumsum(
        .5 * (velocity[:-1] + velocity[1:]) * dt, axis=0)])
    return position, velocity, np.zeros(len(t), dtype=bool)


def kalman(clock, acc, gyro, world, bias):
    # Matches the legacy producer at ded494a: px,py,vx,vy,bx,by; no foot updates.
    x = np.r_[np.zeros(4), bias]
    P = np.diag([.01, .01, .1, .1, .1, .1])
    H = np.zeros((2, 6)); H[0, 2] = H[1, 3] = 1.
    R = np.eye(2) * .02**2
    Q = np.diag([1e-8, 1e-8, 4e-5, 4e-5, 1e-8, 1e-8])
    identity = np.eye(6)
    states, updates = [], []
    since = None
    for i, now in enumerate(clock):
        dt = .002 if i == 0 else float(np.clip(now-clock[i-1], 1e-4, .05))
        F = np.eye(6)
        F[0, 2] = F[1, 3] = dt
        F[0, 4] = F[1, 5] = -.5*dt**2
        F[2, 4] = F[3, 5] = -dt
        B = np.zeros((6, 2))
        B[0, 0] = B[1, 1] = .5*dt**2
        B[2, 0] = B[3, 1] = dt
        x = F@x+B@world[i, :2]
        P = F@P@F.T+Q*(dt/.02)
        still = abs(np.linalg.norm(acc[i])-9.81) <= .5 and np.linalg.norm(gyro[i]) <= .08
        since = (since if since is not None else now) if still else None
        update = still and now-since >= .2
        if update:
            K = np.linalg.solve(H@P@H.T+R, (P@H.T).T).T
            x -= K@(H@x)
            P = (identity-K@H)@P@(identity-K@H).T+K@R@K.T
        states.append(x.copy()); updates.append(update)
    states = np.array(states)
    return states[:, :2], states[:, 2:4], np.array(updates)


def evaluate(trial, out):
    imu, odom = rows(Path(str(trial)+'-record')/'imu.jsonl'), rows(Path(str(trial)+'-record')/'odom.jsonl')
    policy = rows(trial/'policy/frames.jsonl')
    result = json.loads((trial/'policy/result.json').read_text())
    origin = imu[0]['stamp'][0]*10**9 + imu[0]['stamp'][1]
    t, ot = stamps(imu, origin), stamps(odom, origin)
    if np.any(np.diff(t) <= 0) or np.any(np.diff(ot) <= 0):
        raise ValueError('Unordered or duplicate acquisitions; refusing silent sort/dedup')
    acc = np.array([r['a'] for r in imu])
    gyro = np.array([r['w'] for r in imu])
    quat = np.array([r['q_xyzw'] for r in imu])  # Recorder stores ROS fields in scipy order.
    op = np.array([r['position'] for r in odom])
    receive = np.array([r['recv'] for r in imu])
    if not all(np.isfinite(a).all() for a in (t, ot, acc, gyro, quat, op, receive)):
        raise ValueError('Nonfinite input')
    # Every policy root is an unchanged simulator odometry sample. Exact matching
    # maps the walk transition to sensor time without fitting an arbitrary offset.
    roots = {tuple(r['position']): i for i, r in enumerate(odom)}
    first = next(r for r in policy if r['mode'] == 'walk')
    walk_start = ot[roots[tuple(first['root'])]]
    stop = next((r for r in policy if r['mode'] == 'stand' and r['elapsed'] > first['elapsed']), None)
    walk_end = ot[roots[tuple(stop['root'])]] if stop else ot[-1]
    bias_mask = (t >= walk_start-3.5) & (t <= walk_start-1.5)
    world = Rotation.from_quat(quat).apply(acc)
    if bias_mask.sum() < 500:
        raise ValueError('Insufficient pre-walk calibration')
    bias = world[bias_mask, :2].mean(axis=0)
    use = (t >= walk_start-1.) & (t <= min(walk_end+3., ot[-1])) & (t >= ot[0])
    nt, na, ng, nw, nr = t[use], acc[use], gyro[use], world[use], receive[use]
    horizon = min(30., walk_end-walk_start, nt[-1]-walk_start)
    grid = np.arange(nt[0], nt[-1], .02)
    idx = np.searchsorted(nt, grid, side='right')-1
    variants = {
        'native_raw': (nt, integrate(nt, nw[:, :2])),
        'native_static_bias': (nt, integrate(nt, nw[:, :2]-bias)),
        'snapshot_50hz_static_bias': (grid, integrate(grid, nw[idx, :2]-bias)),
        'snapshot_50hz_raw': (grid, integrate(grid, nw[idx, :2])),
        'kalman_header': (nt, kalman(nt, na, ng, nw, bias)),
        'kalman_header_zero_init': (nt, kalman(nt, na, ng, nw, np.zeros(2))),
        'kalman_receive': (nt, kalman(nr, na, ng, nw, bias)),
    }
    # Position derivative over a centered 100 ms interval, used only for scoring.
    def interp(times, values, at):
        return np.column_stack([np.interp(np.atleast_1d(at), times, values[:, j])
                                for j in range(values.shape[1])])
    initial_truth_v = ((interp(ot, op[:, :2], nt[0]+.05)
                       -interp(ot, op[:, :2], nt[0]-.05))/.1)[0]
    raw_p, raw_v, raw_z = variants['native_raw'][1]
    # Explicit oracle diagnostic: isolates propagation from the assumption v0=0.
    # This is not a deployable estimator and does not alter the primary variants.
    variants['native_truth_v0_diagnostic'] = (nt, (
        raw_p+(nt-nt[0])[:, None]*initial_truth_v,
        raw_v+initial_truth_v, raw_z))
    common = np.arange(0., horizon, .02)
    at = walk_start+common
    truth = interp(ot, op[:, :2], at)-interp(ot, op[:, :2], walk_start)
    truth_v = (interp(ot, op[:, :2], at+.05)-interp(ot, op[:, :2], at-.05))/.1
    report = {'trial': str(trial), 'policy_failure': result['failure'],
              'policy_control_seconds': result['control_seconds'],
              'walk_seconds_sensor_clock': float(walk_end-walk_start),
              'walk_origin_s': float(walk_start), 'initial_speed_m_s': float(np.linalg.norm(truth_v[0])),
              'integration_initial_truth_velocity_xy': initial_truth_v.tolist(),
              'bias_xy_m_s2': bias.tolist(), 'calibration_samples': int(bias_mask.sum()),
              'calibration_gyro_max': float(np.linalg.norm(gyro[bias_mask], axis=1).max()),
              'calibration_passes_current_still_fraction': float(np.mean(
                  (np.linalg.norm(gyro[bias_mask], axis=1) <= .08)
                  & (abs(np.linalg.norm(acc[bias_mask], axis=1)-9.81) <= .5))),
              'native_rate_hz': float(1/np.median(np.diff(nt))),
              'header_dt_max_ms': float(np.diff(nt).max()*1000),
              'receive_dt_max_ms': float(np.diff(nr).max()*1000),
              'quaternion_norm_max_error': float(abs(np.linalg.norm(quat, axis=1)-1).max()),
              'horizon_s': float(horizon), 'variants': {}}
    arrays = {'seconds': common, 'truth_xy': truth, 'truth_velocity_xy': truth_v}
    for name, (vt, (pos, vel, zupt)) in variants.items():
        est = interp(vt, pos, at)-interp(vt, pos, walk_start)
        velocity = interp(vt, vel, at)
        error = np.linalg.norm(est-truth, axis=1)
        speed_error = np.linalg.norm(velocity-truth_v, axis=1)
        checkpoints = []
        for sec in (5., 10., 20., 30.):
            if sec > horizon+1e-6:
                continue
            absolute = walk_start+sec
            delta = interp(vt, pos, absolute)-interp(vt, pos, walk_start)
            actual = interp(ot, op[:, :2], absolute)-interp(ot, op[:, :2], walk_start)
            vv = (interp(ot, op[:, :2], absolute+.05)-interp(ot, op[:, :2], absolute-.05))/.1
            checkpoints.append({'seconds': sec, 'position_error_m': float(np.linalg.norm(delta-actual)),
                                'velocity_error_m_s': float(np.linalg.norm(interp(vt, vel, absolute)-vv)),
                                'truth_net_m': float(np.linalg.norm(actual))})
        walking = (vt >= walk_start) & (vt <= walk_start+horizon)
        update_at = vt[walking & zupt]
        update_speed = np.linalg.norm((interp(ot, op[:, :2], update_at+.05)
                        -interp(ot, op[:, :2], update_at-.05))/.1, axis=1)
        report['variants'][name] = {'checkpoints': checkpoints,
            'truth_velocity_initialization_diagnostic': name == 'native_truth_v0_diagnostic',
            'position_rmse_m': float(np.sqrt(np.mean(error**2))),
            'position_max_m': float(error.max()), 'velocity_rmse_m_s': float(np.sqrt(np.mean(speed_error**2))),
            'walking_zupt_count': int(zupt[walking].sum()), 'walking_samples': int(walking.sum()),
            'zupt_after_first_5s': int((update_at > walk_start+5.).sum()),
            'zupt_with_truth_speed_over_0p1': int((update_speed > .1).sum())}
        arrays[name+'_xy'] = est
        arrays[name+'_error'] = error
    np.savez_compressed(out/(trial.name+'.npz'), **arrays)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    report = {'scope': 'Simulation-only offline audit; primary variants have no truth corrections; '
              'native_truth_v0_diagnostic alone uses one truth velocity initialization; no added noise; '
              'reported IMU attitude used; 2 s pre-walk bias mean; zero initial velocity; '
              'horizontal displacement since walk onset; fixed current ZUPT thresholds',
              'trials': [evaluate(a.run/f'walk-r{i}', a.out) for i in (1, 2, 3)]}
    (a.out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
