"""Score recorded ROS producer outputs against simulator pelvis odometry."""
import argparse
import json
from pathlib import Path
import numpy as np


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    truth = load(a.run/'trial-record/odom.jsonl')
    old, new = (load(a.run/name/'packets.jsonl') for name in ('old', 'new'))
    old = [r for r in old if 'stamp_ns' in r]
    new = [r for r in new if 'stamp_ns' in r]
    if not old or not new or not truth:
        raise ValueError('Missing producer or odometry samples')
    origin = int(truth[0]['stamp'][0])*10**9+int(truth[0]['stamp'][1])
    ts = np.array([(int(x['stamp'][0])*10**9+int(x['stamp'][1])-origin)*1e-9 for x in truth])
    op = np.array([x['position'] for x in truth])
    streams = {}
    for name, rows in [('old', old), ('new', new)]:
        t = np.array([(r['stamp_ns']-origin)*1e-9 for r in rows])
        if np.any(np.diff(t) <= 0) or np.any(np.diff(ts) <= 0):
            raise ValueError('Non-monotonic or duplicated timestamps')
        streams[name] = (t, np.array([r['position'] for r in rows]),
                         np.array([r['velocity'] for r in rows]))
    start = max(ts[0]+.1, *(v[0][0] for v in streams.values()))
    end = min(ts[-1]-.1, *(v[0][-1] for v in streams.values()))
    if end-start < 5:
        raise ValueError('Insufficient shared time window')
    grid = np.arange(start, end, .02)
    interp = lambda t, v, x: np.column_stack([np.interp(x, t, v[:, k]) for k in range(v.shape[1])])
    actual = interp(ts, op[:, :2], grid)-interp(ts, op[:, :2], np.array([start]))
    actual_v = (interp(ts, op[:, :2], grid+.05)-interp(ts, op[:, :2], grid-.05))/.1
    report = {'scope': 'Horizontal pelvis displacement and velocity; no truth feedback; initial displacement aligned for scoring; ROS outputs use same simulated chest samples',
              'common_duration_s': float(end-start), 'variants': {}}
    for name, (t, p, v) in streams.items():
        pred = interp(t, p[:, :2], grid)-interp(t, p[:, :2], np.array([start]))
        pv = interp(t, v[:, :2], grid)
        pe = np.linalg.norm(pred-actual, axis=1)
        ve = np.linalg.norm(pv-actual_v, axis=1)
        report['variants'][name] = {'samples': int(len(t)), 'position_rmse_m': float(np.sqrt(np.mean(pe**2))),
            'velocity_rmse_m_s': float(np.sqrt(np.mean(ve**2))), 'last_position_error_m': float(pe[-1]),
            'last_velocity_error_m_s': float(ve[-1]),
            'checkpoints': [{'seconds': sec, 'position_error_m': float(pe[np.searchsorted(grid, start+sec)]),
                             'velocity_error_m_s': float(ve[np.searchsorted(grid, start+sec)])}
                            for sec in (5, 10, 20, 30, 40) if start+sec < grid[-1]]}
    a.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
