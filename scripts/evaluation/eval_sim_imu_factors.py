"""Truth-assisted inertial integration diagnostics, never an estimator input."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def read(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def ts(rows):
    return np.array([r['stamp'][0]*10**9+r['stamp'][1] for r in rows], dtype=np.int64)*1e-9


def ip(t, x, at):
    return np.column_stack([np.interp(at, t, x[:, i]) for i in range(x.shape[1])])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    imu, truth, joints = [read(a.run / 'trial-record' / n) for n in ('imu.jsonl', 'odom.jsonl', 'joints.jsonl')]
    t, tt = ts(imu), ts(truth)
    recv = np.array([r['recv'] for r in imu])
    mask = (t >= tt[0]+.1) & (t <= tt[-1]-.1)
    t, recv = t[mask], recv[mask]
    q = Rotation.from_quat(np.array([r['q_xyzw'] for r in imu])[mask])
    force = q.apply(np.array([r['a'] for r in imu])[mask])
    truth_pos = np.array([r['position'] for r in truth])
    truth_q = Rotation.from_quat([r['q_xyzw'] for r in truth])
    lever = np.array([.0239465, -.000228682, .04171])
    site = truth_pos + truth_q.apply(lever)
    site_p = ip(tt, site, t)
    initial_v = (ip(tt, site, np.array([t[0]+.05]))-ip(tt, site, np.array([t[0]-.05])))[0]/.1
    report = {'scope': 'Truth-assisted offline diagnostic: initial IMU-site velocity supplied by odometry; no feedback thereafter',
              'duration_s': float(t[-1]-t[0]), 'gravity_assumed_m_s2': 9.81, 'variants': {}, 'recorded_intervals': {}}
    final_v = (ip(tt, site, np.array([t[-1]+.05]))-ip(tt, site, np.array([t[-1]-.05])))[0]/.1
    accel0 = force.copy(); accel0[:, 2] -= 9.81
    elapsed = t[-1]-t[0]
    mean_acc = np.sum(.5*(accel0[1:]+accel0[:-1])*np.diff(t)[:, None], axis=0)/elapsed
    apparent_bias = mean_acc-(final_v-initial_v)/elapsed
    report['truth_derived_apparent_world_acceleration_bias_m_s2'] = apparent_bias.tolist()
    report['bias_limitation'] = 'Includes timing, truth differentiation, lever-arm, noise and model errors; not an identified sensor bias.'
    for name, clock, gravity in [('header_g981', t, 9.81), ('receive_g981', recv, 9.81), ('header_g980', t, 9.80), ('header_truth_fitted_constant_bias', t, 9.81)]:
        dt = np.diff(clock)
        if np.any(dt <= 0):
            raise ValueError('nonmonotonic clock')
        accel = force.copy(); accel[:, 2] -= gravity
        if name == 'header_truth_fitted_constant_bias':
            accel -= apparent_bias
        dv = .5*(accel[1:]+accel[:-1])*dt[:, None]
        velocity = np.vstack([initial_v, initial_v+np.cumsum(dv, axis=0)])
        displacement = np.vstack([np.zeros(3), np.cumsum(.5*(velocity[1:]+velocity[:-1])*dt[:, None], axis=0)])
        e = displacement-(site_p-site_p[0])
        report['variants'][name] = {'horizontal_position_rmse_m': float(np.sqrt(np.mean(np.sum(e[:, :2]**2, axis=1)))),
            'vertical_position_rmse_m': float(np.sqrt(np.mean(e[:, 2]**2))), 'final_error_m': e[-1].tolist()}
    for group in ('leg', 'waist', 'head', 'arm'):
        jt = ts([r for r in joints if r['group'] == group]); delta = np.diff(jt)
        report['recorded_intervals'][group] = {'samples': len(jt), 'p99_ms': float(np.percentile(delta, 99)*1000),
            'max_ms': float(np.max(delta)*1000), 'intervals_over_1_5ms': int(np.sum(delta > .0015))}
    a.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
