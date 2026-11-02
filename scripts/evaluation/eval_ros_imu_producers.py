"""Score recorded ROS producer outputs against simulator pelvis odometry."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


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
    duplicate_counts = {}
    for name, rows in [('old', old), ('new', new)]:
        t = np.array([(r['stamp_ns']-origin)*1e-9 for r in rows])
        if np.any(np.diff(t) < 0) or np.any(np.diff(ts) <= 0):
            raise ValueError('Timestamp rollback or invalid truth timestamps')
        # Repeated output timestamps are recorded as a producer defect; retain
        # the last output solely to make interpolation well-defined.
        keep = np.r_[np.diff(t) != 0, True]
        duplicate_counts[name] = int((~keep).sum())
        streams[name] = (t[keep], np.array([r['position'] for r in rows])[keep],
                         np.array([r['velocity'] for r in rows])[keep])
    start = max(ts[0]+.1, *(v[0][0] for v in streams.values()))
    end = min(ts[-1]-.1, *(v[0][-1] for v in streams.values()))
    if end-start < 5:
        raise ValueError('Insufficient shared time window')
    grid = np.arange(start, end, .02)
    interp = lambda t, v, x: np.column_stack([np.interp(x, t, v[:, k]) for k in range(v.shape[1])])
    actual = interp(ts, op[:, :2], grid)-interp(ts, op[:, :2], np.array([start]))
    actual_v = (interp(ts, op[:, :2], grid+.05)-interp(ts, op[:, :2], grid-.05))/.1
    policy = load(a.run/'trial/policy/frames.jsonl')
    truth_by_root = {tuple(x['position']): ts[i] for i, x in enumerate(truth)}
    events = [(truth_by_root[tuple(r['root'])], r['mode']) for r in policy
              if tuple(r['root']) in truth_by_root]
    event_t = np.array([e[0] for e in events])
    if not events or np.any(np.diff(event_t) < 0):
        raise ValueError('Policy mode cannot be mapped to truth time')
    modes = np.array([e[1] for e in events])
    select = np.searchsorted(event_t, grid, side='right')-1
    labels = np.where(select >= 0, modes[np.maximum(select, 0)], 'before_policy')
    report = {'scope': 'Horizontal pelvis displacement and velocity; no truth feedback; initial displacement aligned for scoring; ROS outputs use same simulated chest samples',
              'common_duration_s': float(end-start), 'duplicate_output_timestamps': duplicate_counts,
              'policy_frames_matched': len(events), 'policy_frames_total': len(policy), 'variants': {}}
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
                            for sec in (5, 10, 20, 30, 40, 50) if start+sec < grid[-1]]}
        report['variants'][name]['modes'] = {
            mode: {'samples': int(np.sum(labels == mode)),
                   'position_rmse_m': float(np.sqrt(np.mean(pe[labels == mode]**2))),
                   'velocity_rmse_m_s': float(np.sqrt(np.mean(ve[labels == mode]**2)))}
            for mode in sorted(set(labels)) if np.any(labels == mode)}
        if p.shape[1] == 3 and v.shape[1] == 3:
            truth3 = interp(ts, op, grid); truth3 -= interp(ts, op, np.array([start]))
            truthv3 = (interp(ts, op, grid+.05)-interp(ts, op, grid-.05))/.1
            estimate3 = interp(t, p, grid)-interp(t, p, np.array([start]))
            error3 = np.linalg.norm(estimate3-truth3, axis=1)
            verr3 = np.linalg.norm(interp(t, v, grid)-truthv3, axis=1)
            report['variants'][name].update(position_3d_rmse_m=float(np.sqrt(np.mean(error3**2))),
                velocity_3d_rmse_m_s=float(np.sqrt(np.mean(verr3**2))),
                final_position_3d_error_m=float(error3[-1]), final_velocity_3d_error_m_s=float(verr3[-1]),
                final_vertical_position_error_m=float(estimate3[-1, 2]-truth3[-1, 2]))
            # Oracle diagnostic only: a single truth vertical velocity at the
            # common start, with no subsequent feedback. The producer has no
            # vertical velocity updates, so this is an exact linear adjustment.
            initial_z_error = (truthv3[0, 2]-interp(t, v, np.array([start]))[0, 2])
            corrected_z = estimate3[:, 2]+initial_z_error*(grid-start)
            report['variants'][name]['truth_initial_vertical_velocity_diagnostic'] = {
                'initial_velocity_adjustment_m_s': float(initial_z_error),
                'final_vertical_position_error_m': float(corrected_z[-1]-truth3[-1, 2]),
                'vertical_position_rmse_m': float(np.sqrt(np.mean((corrected_z-truth3[:, 2])**2)))}
            selected = [r for r in new if 'quat_wxyz' in r]
            st = np.array([(r['stamp_ns']-origin)*1e-9 for r in selected])
            sq = Rotation.from_quat(np.array([r['quat_wxyz'] for r in selected])[:, [1,2,3,0]])
            tq = Rotation.from_quat([r['q_xyzw'] for r in truth])
            valid = (st >= ts[0]) & (st <= ts[-1])
            angles = (Slerp(ts, tq)(st[valid]).inv()*sq[valid]).magnitude()
            report['variants'][name]['orientation_error_rad'] = {'median': float(np.median(angles)),
                'p95': float(np.percentile(angles,95)), 'max': float(np.max(angles))}
            measured_w = np.array([r['angular_velocity_body'] for r in selected])[valid]
            truth_w = interp(ts, np.array([r['angular'] for r in truth]), st[valid])
            werr = np.linalg.norm(measured_w-truth_w, axis=1)
            report['variants'][name]['angular_velocity_error_rad_s'] = {
                'median': float(np.median(werr)), 'p95': float(np.percentile(werr,95)),
                'max': float(np.max(werr))}
    a.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
