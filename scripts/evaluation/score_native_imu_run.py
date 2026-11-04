"""Post-run native IMU scoring only; never import or modify a running estimator."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=34.)
    p.add_argument('--max-error', type=float, default=.5)
    p.add_argument('--out', type=Path, help='Optional new score file; never overwrite an earlier score')
    a = p.parse_args()
    out = a.out or a.run/'score.json'
    if out.exists(): raise ValueError('Do not overwrite existing score')
    result = json.loads((a.run/'result.json').read_text())
    frames = [json.loads(x) for x in (a.run/'frames.jsonl').open()]
    with (a.run/'truth-imu.jsonl').open() as f: origin = np.array(json.loads(next(f))['position'])
    estimated = np.array([x['estimated_root'] for x in frames])
    truth = np.array([x['root'] for x in frames])
    error = estimated-truth
    # Estimated XY has an arbitrary local origin; height is above the declared
    # flat floor. Register only the scoring coordinate origin, not a filter.
    error[:, :2] += origin[:2]
    if not np.isfinite(error).all(): raise ValueError('Nonfinite trajectory')
    xy = np.linalg.norm(error[:, :2], axis=1)
    valid = (result['sim_seconds'] >= a.seconds-.02 and result['failure'] is None
             and result['resets'] == 0 and result['state_source'] == 'imu')
    max_3d = float(np.linalg.norm(error, axis=1).max())
    report = {'scope': 'post-run scoring only; no estimator feedback', 'seconds': result['sim_seconds'],
              'frames': len(frames), 'resets': result['resets'], 'failure': result['failure'],
              'max_horizontal_error_m': float(xy.max()), 'final_horizontal_error_m': float(xy[-1]),
              'max_position_error_3d_m': max_3d,
              'max_absolute_frame_horizontal_error_m': float(np.linalg.norm(estimated[:, :2]-truth[:, :2], axis=1).max()),
              'true_net_motion_m': float(np.linalg.norm(truth[-1, :2]-origin[:2])),
              'min_true_root_height_m': float(truth[:, 2].min()),
              'threshold_m': a.max_error, 'completed_valid_duration': valid,
              'pass': bool(valid and xy.max() < a.max_error and max_3d < a.max_error)}
    out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if not report['pass']: raise SystemExit(2)


if __name__ == '__main__': main()
