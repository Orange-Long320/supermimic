"""Summarize offline LIO stationarity diagnostics without declaring acceptance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def window_metrics(a):
    p = a[:, 1:4] - a[0, 1:4]
    d = np.linalg.norm(p, axis=1)
    q = a[:, 4:8]
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    angles = np.rad2deg(2 * np.arccos(np.clip(np.abs(q @ q[0]), 0., 1.)))
    return {'samples': len(a), 'span_s': float(a[-1, 0]-a[0, 0]),
            'net_translation_m': float(d[-1]), 'max_translation_m': float(d.max()),
            'translation_delta_xyz_m': p[-1].tolist(),
            'final_attitude_change_deg': float(angles[-1]),
            'max_attitude_change_deg': float(angles.max()),
            'output_gaps_over_150ms': int(np.sum(np.diff(a[:, 0]) > .15)),
            'output_nonincreasing_stamps': int(np.sum(np.diff(a[:, 0]) <= 0)),
            'output_gap_max_s': float(np.diff(a[:, 0]).max()) if len(a)>1 else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stationary-confirmed', action='store_true',
                        help='record an explicit user confirmation of stationary capture')
    args = parser.parse_args()
    results = {}
    for name in args.runs:
        path = args.root / name
        report = json.loads((path/'report.json').read_text())
        a = np.loadtxt(path/'trajectory.csv', delimiter=',', skiprows=1, ndmin=2)
        complete = (report['error'] is None and report.get('finite') and
                    report.get('final_scan_lag_s') is not None and
                    abs(report['final_scan_lag_s']) <= .001)
        results[name] = {'last_scan_covered': bool(complete),
                         'scan_output_coverage_complete': bool(complete and np.all(
                             (np.diff(a[:, 0]) > 0) & (np.diff(a[:, 0]) <= .15))),
                         'input_sent': report['sent'],
                         'native_counts': report.get('native_counts'),
                         'native_input_delivery_complete': (
                             bool(report.get('native_cloud_callbacks_match_sent') and
                                  report.get('native_imu_callbacks_match_sent'))
                             if 'native_counts' in report else None),
                         'odom_recording_matches_native_publications': report.get(
                             'odom_received_matches_native_publications'),
                         'replay_rate': report['replay_rate'], 'all': window_metrics(a)}
        for key, subset in [('after_first_5s', a[a[:, 0] >= a[0, 0]+5]),
                            ('last_30s', a[a[:, 0] >= a[-1, 0]-30])]:
            valid = len(subset)>1 and (key != 'last_30s' or a[-1, 0]-a[0, 0]>=30)
            results[name][key] = window_metrics(subset) if valid else None
    output = {'stationary_capture_confirmed_by_user': args.stationary_confirmed,
              'independent_motion_ground_truth': False,
              'dynamic_accuracy_validated': False, 'acceptance_declared': False,
              'runs': results,
              'scope': 'descriptive stationary output drift under the user-confirmed condition; '
                       'no parameter selection or claimed dynamic calibration'}
    with args.out.open('x') as stream:
        stream.write(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
