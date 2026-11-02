"""Evaluate paired simulated IMU producers; odometry is diagnostic truth only."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def read(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def stamp(rows):
    return np.array([r['stamp'][0] + r['stamp'][1] * 1e-9 for r in rows])


def interp(t, values, at):
    return np.column_stack([np.interp(at, t, values[:, i]) for i in range(values.shape[1])])


def stats(x):
    return {'median': float(np.median(x)), 'p95': float(np.percentile(x, 95)),
            'max': float(np.max(x))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    truth = read(a.run / 'trial-record/odom.jsonl')
    imu = read(a.run / 'trial-record/imu.jsonl')
    tt, it = stamp(truth), stamp(imu)
    if min(len(tt), len(it)) < 3 or np.any(np.diff(tt) <= 0) or np.any(np.diff(it) <= 0):
        raise ValueError('Insufficient or nonmonotonic truth/IMU timestamps')
    pos = np.array([r['position'] for r in truth])
    tq = Rotation.from_quat([r['q_xyzw'] for r in truth])
    valid = (it >= tt[0]) & (it <= tt[-1])
    iq = Rotation.from_quat([r['q_xyzw'] for r in imu])[valid]
    angles = (Slerp(tt, tq)(it[valid]).inv() * iq).magnitude()
    report = {'scope': 'Simulation only; odometry is independent diagnostic truth, never estimator input',
              'imu_header_interval_ms': stats(np.diff(it) * 1e3),
              'imu_header_gaps_over_3ms': int(np.sum(np.diff(it) > .003)),
              'imu_pelvis_orientation_error_rad': stats(angles), 'variants': {}}
    variants = {}
    for name in ('pelvis', 'chest'):
        rows = [r for r in read(a.run / name / 'packets.jsonl') if 'stamp_ns' in r]
        if len(rows) < 2:
            raise ValueError(f'{name}: insufficient producer output')
        t = np.array([r['stamp_ns'] * 1e-9 for r in rows])
        if np.any(np.diff(t) <= 0):
            raise ValueError(f'{name}: repeated or nonmonotonic output')
        variants[name] = (t, np.array([r['position'] for r in rows]),
                          np.array([r['velocity'] for r in rows]))
    start = max(tt[0]+.1, *(v[0][0] for v in variants.values()))
    end = min(tt[-1]-.1, *(v[0][-1] for v in variants.values()))
    if end-start < 5:
        raise ValueError('Insufficient shared time window')
    grid = np.arange(start, end, .02)
    actual = interp(tt, pos, grid) - interp(tt, pos, np.array([start]))
    vel = (interp(tt, pos, grid+.05)-interp(tt, pos, grid-.05))/.1
    report['common_duration_s'] = float(end-start)
    # Central attitude difference provides a separate local-frame gyro reference.
    gt = it[(it >= tt[0]+.01) & (it <= tt[-1]-.01)][::5]
    q0, q1 = Slerp(tt, tq)(gt-.01), Slerp(tt, tq)(gt+.01)
    omega = (q0.inv()*q1).as_rotvec()/.02
    gyro = interp(it, np.array([r['w'] for r in imu]), gt)
    world_omega = Slerp(tt, tq)(gt).apply(omega)
    report['gyro_against_odometry_finite_difference'] = {
        'interval_s': .02, 'local_frame_error_rad_s': stats(np.linalg.norm(gyro-omega, axis=1)),
        'world_frame_error_rad_s': stats(np.linalg.norm(gyro-world_omega, axis=1)),
        'limitation': 'Odometry and IMU header physical sampling alignment is not independently established.'}
    report['truth_initial_velocity_m_s'] = vel[0].tolist()
    init_grid = grid[grid < start+3]
    init_v = vel[:len(init_grid)]
    report['truth_first_3s'] = {'speed_m_s': stats(np.linalg.norm(init_v, axis=1)),
                                'displacement_m': float(np.linalg.norm(actual[len(init_grid)-1]))}
    for name, (t, position, velocity) in variants.items():
        pred = interp(t, position, grid)-interp(t, position, np.array([start]))
        pv = interp(t, velocity, grid)
        pe, ve = pred-actual, pv-vel
        rmse = lambda x: float(np.sqrt(np.mean(x*x)))
        dz = float(vel[0, 2]-pv[0, 2])
        report['variants'][name] = {'packets': len(t), 'initial_velocity_error_m_s': ve[0].tolist(),
            'horizontal_position_rmse_m': rmse(np.linalg.norm(pe[:, :2], axis=1)),
            'horizontal_velocity_rmse_m_s': rmse(np.linalg.norm(ve[:, :2], axis=1)),
            'vertical_position_rmse_m': rmse(pe[:, 2]), 'vertical_velocity_rmse_m_s': rmse(ve[:, 2]),
            'final_vertical_position_error_m': float(pe[-1, 2]),
            'truth_initial_vertical_velocity_diagnostic': {'adjustment_m_s': dz,
                'vertical_position_rmse_m': rmse(pe[:, 2]+dz*(grid-start)),
                'final_vertical_position_error_m': float(pe[-1, 2]+dz*(grid[-1]-start))}}
    a.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
