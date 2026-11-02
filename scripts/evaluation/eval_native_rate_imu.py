"""Compare native-rate IMU strapdown against the 50 Hz snapshot chain.

Both variants integrate the SAME recorded session; they differ only in which
samples the integrator sees. Variant A uses every native sample with its own
dt. Variant B emulates the deployed chain: keep only the latest sample at each
20 ms control tick. Ground truth comes from the simulator odometry, used for
evaluation only and never fed to either integrator.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

G = 9.81


def stamp_seconds(record):
    return record['stamp'][0] + record['stamp'][1] * 1e-9


def load_imu(path):
    rows = [json.loads(s) for s in path.open()]
    ts = np.array([stamp_seconds(r) for r in rows])
    acc = np.array([r['a'] for r in rows])
    quat = np.array([r['q_xyzw'] for r in rows])
    keep = np.isfinite(acc).all(axis=1) & np.isfinite(quat).all(axis=1) & np.isfinite(ts)
    return ts[keep], acc[keep], quat[keep]


def load_odom(path):
    rows = [json.loads(s) for s in path.open()]
    ts = np.array([stamp_seconds(r) for r in rows])
    pos = np.array([r['position'] for r in rows])
    keep = np.isfinite(pos).all(axis=1) & np.isfinite(ts)
    return ts[keep], pos[keep]


def dedupe_sorted(ts, *arrays):
    order = np.argsort(ts, kind='stable')
    ts = ts[order]
    arrays = [a[order] for a in arrays]
    keep = np.r_[True, np.diff(ts) > 0.]
    return (ts[keep], *[a[keep] for a in arrays])


def integrate(ts, acc, quat):
    """Trapezoidal strapdown integration; returns per-sample time, position, velocity."""
    world = Rotation.from_quat(quat).apply(acc)
    world[:, 2] -= G
    v = np.zeros(3)
    p = np.zeros(3)
    out_t = [ts[0]]
    out_p = [p.copy()]
    out_v = [v.copy()]
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i - 1]
        half = .5 * (world[i - 1] + world[i]) * dt
        v_next = v + half
        p = p + .5 * (v + v_next) * dt
        v = v_next
        out_t.append(ts[i])
        out_p.append(p.copy())
        out_v.append(v.copy())
    return np.array(out_t), np.array(out_p), np.array(out_v)


def snapshot_samples(ts, acc, quat, grid):
    """Latest sample at or before each grid tick: exactly what the chain forwards."""
    idx = np.searchsorted(ts, grid, side='right') - 1
    ok = idx >= 0
    return grid[ok], acc[idx[ok]], quat[idx[ok]]


def xy_error(est_t, est_p, truth_t, truth_p, t_start):
    tx = np.interp(est_t, truth_t, truth_p[:, 0]) - np.interp(t_start, truth_t, truth_p[:, 0])
    ty = np.interp(est_t, truth_t, truth_p[:, 1]) - np.interp(t_start, truth_t, truth_p[:, 1])
    return np.linalg.norm(np.column_stack([est_p[:, 0] - tx, est_p[:, 1] - ty]), axis=1)


def summary(err):
    return {'p50': float(np.percentile(err, 50)), 'p95': float(np.percentile(err, 95)),
            'max': float(err.max()), 'final': float(err[-1])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('imu', 'odom', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--warmup', type=float, default=1.0)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('Expected a new output file')

    ts, acc, quat = dedupe_sorted(*load_imu(a.imu))
    ots, opos = dedupe_sorted(*load_odom(a.odom))
    if len(ts) < 500 or len(ots) < 100:
        raise ValueError('Not enough samples; is the simulator publishing native IMU?')

    dt = np.diff(ts)
    rate = 1. / float(np.median(dt))
    mag = np.linalg.norm(acc, axis=1)
    grid = np.arange(ts[0], ts[-1], .02)
    sts, sacc, squat = snapshot_samples(ts, acc, quat, grid)

    t_start = ts[0] + a.warmup
    an_t_all, an_p, an_v = integrate(ts, acc, quat)
    bt_t_all, bt_p, bt_v = integrate(sts, sacc, squat)
    an_err = xy_error(an_t_all, an_p, ots, opos, t_start)
    bt_err = xy_error(bt_t_all, bt_p, ots, opos, t_start)
    end = min(an_t_all[-1], bt_t_all[-1])
    an_sel = (an_t_all >= t_start) & (an_t_all <= end)
    bt_sel = (bt_t_all >= t_start) & (bt_t_all <= end)
    grid_t = np.linspace(t_start, end, 2000)
    path = np.diff(np.column_stack([np.interp(grid_t, ots, opos[:, 0]),
                                    np.interp(grid_t, ots, opos[:, 1])]), axis=0)
    net = np.column_stack([np.interp(end, ots, opos[:, 0]) - np.interp(t_start, ots, opos[:, 0]),
                           np.interp(end, ots, opos[:, 1]) - np.interp(t_start, ots, opos[:, 1])])

    # Foot-strike impact peaks: what fraction of peak magnitude survives 50 Hz sampling.
    peaks = []
    for i in range(2, len(ts) - 2):
        if mag[i] > 15. and mag[i] >= mag[i - 1] and mag[i] >= mag[i + 1]:
            if peaks and ts[i] - peaks[-1] < .1:
                continue
            peaks.append(i)
    captured = []
    for i in peaks:
        near = np.abs(sts - ts[i]) <= .03
        if near.any():
            captured.append(float(np.linalg.norm(sacc[near], axis=1).max() / mag[i]))

    report = {
        'native_rate_hz': rate,
        'native_samples': int(len(ts)),
        'native_dt_p50_ms': float(np.percentile(dt, 50) * 1e3),
        'native_dt_p99_ms': float(np.percentile(dt, 99) * 1e3),
        'native_dt_max_ms': float(dt.max() * 1e3),
        'native_gaps_over_20ms': int((dt > .02).sum()),
        'odom_rate_hz': 1. / float(np.median(np.diff(ots))),
        'compared_seconds': float(end - t_start),
        'truth_path_m': float(np.linalg.norm(path, axis=1).sum()),
        'truth_net_m': float(np.linalg.norm(net)),
        'native_integration_error_m': summary(an_err[an_sel]),
        'snapshot_50hz_error_m': summary(bt_err[bt_sel]),
        'speed_native_p50_p95_max': [float(x) for x in np.percentile(np.linalg.norm(an_v[an_sel], axis=1), (50, 95, 100))],
        'speed_snapshot_p50_p95_max': [float(x) for x in np.percentile(np.linalg.norm(bt_v[bt_sel], axis=1), (50, 95, 100))],
        'impact_peaks': len(peaks),
        'impact_capture_ratio_p50': float(np.percentile(captured, 50)) if captured else None,
        'impact_capture_ratio_p05': float(np.percentile(captured, 5)) if captured else None,
        'accel_magnitude_p50_p99_max': [float(x) for x in np.percentile(mag, (50, 99, 100))],
        'scope': ('Recorded official simulator IMU at native rate; truth is simulator odometry used for '
                  'evaluation only; offline integration, no control loop, no contact model'),
    }
    a.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
