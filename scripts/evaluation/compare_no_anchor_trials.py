"""Compare recorded simulator rollouts on paired, equally long time windows.

Run on the server. No simulation, training, contact inference or acceptance gate.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_trial(path):
    rows = [json.loads(s) for s in (path/'frames.jsonl').read_text().splitlines()]
    if len(rows) < 2:
        raise ValueError('Need at least two recorded frames')
    t = np.array([r['elapsed'] for r in rows])
    if np.any(np.diff(t) <= 0):
        raise ValueError('Non-monotonic timestamps')
    q = np.array([r['q'] for r in rows])
    w, x, y, z = q.T
    yaw = np.unwrap(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
    modes = np.array([r['mode'] for r in rows])
    bounds = np.r_[0, np.flatnonzero(modes[1:] != modes[:-1])+1, len(rows)]
    segments = [(str(modes[a]), float(t[a]), float(t[b-1]), a, b)
                for a, b in zip(bounds[:-1], bounds[1:])]
    return dict(path=path, rows=rows, t=t, pos=np.array([r['root'] for r in rows]),
                yaw=yaw, segments=segments,
                result=json.loads((path/'result.json').read_text()))


def sample(d, start, duration):
    grid = np.arange(int(np.floor(duration/.02))+1)*.02
    times = start+grid
    pos = np.stack([np.interp(times, d['t'], d['pos'][:, i]) for i in range(3)], -1)
    yaw = np.interp(times, d['t'], d['yaw'])
    return grid, pos, yaw


def metrics(d, start, duration):
    grid, pos, yaw = sample(d, start, duration)
    displacement = np.linalg.norm(pos[:, :2]-pos[0, :2], axis=-1)
    # A 0.2-second displacement derivative reduces frame-to-frame odom jitter.
    if len(grid) < 11:
        return {'observed_s': float(grid[-1]), 'insufficient_for_speed': True}
    vel = (pos[10:, :2]-pos[:-10, :2])/.2
    rate = (yaw[10:]-yaw[:-10])/.2
    mid_yaw = yaw[5:-5]
    local_x = vel[:, 0]*np.cos(mid_yaw)+vel[:, 1]*np.sin(mid_yaw)
    local_y = -vel[:, 0]*np.sin(mid_yaw)+vel[:, 1]*np.cos(mid_yaw)
    speed = np.linalg.norm(vel, axis=-1)
    quiet = (speed < .05) & (np.abs(rate) < .1)
    # Descriptive root-stop criterion, not a balance/contact acceptance gate.
    starts = np.flatnonzero(np.convolve(quiet.astype(int), np.ones(50, dtype=int), 'valid') == 50) if len(quiet) >= 50 else []
    return dict(observed_s=float(grid[-1]), net_displacement_m=float(displacement[-1]),
                max_excursion_m=float(displacement.max()),
                sampled_path_length_m=float(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=-1).sum()),
                median_body_vx_mps=float(np.median(local_x)), median_body_vy_mps=float(np.median(local_y)),
                median_speed_mps=float(np.median(speed)), residual_speed_last_1s_mps=float(np.median(speed[-50:])),
                median_yaw_rate_radps=float(np.median(rate)), yaw_change_deg=float(np.degrees(yaw[-1]-yaw[0])),
                root_stop_time_s=float(grid[5+starts[0]]) if len(starts) else None,
                minimum_height_m=float(pos[:, 2].min()))


def compare(base, candidate):
    # Pair by transition order and require the requested skill to match.
    stops = []
    a = [s for s in base['segments'][1:] if s[0] == 'stand']
    b = [s for s in candidate['segments'][1:] if s[0] == 'stand']
    for i, (old, new) in enumerate(zip(a, b)):
        before_old = base['rows'][old[3]-1]['mode']
        before_new = candidate['rows'][new[3]-1]['mode']
        if before_old != before_new:
            raise ValueError('Release skill sequence differs')
        duration = min(5., old[2]-old[1], new[2]-new[1])
        row = dict(release=i+1, skill=before_old, baseline_release_s=old[1], candidate_release_s=new[1])
        for label, d, s in [('baseline', base, old), ('candidate', candidate, new)]:
            row[label] = metrics(d, s[1], duration)
            events = [e for e in d['result']['phase_events'] if e.get('to') == 'stand' and e.get('from') != 'stand']
            row[label]['selected_reference_frame'] = events[i]['next_frame_at_selection'] if i < len(events) else None
            # Planned path length is not subtracted from actual displacement.
            idx = (d['t'] >= s[1]) & (d['t'] <= s[1]+row[label]['observed_s'])
            goals = np.array([r['goal_current'] for r, keep in zip(d['rows'], idx) if keep])
            nj = len(d['rows'][0]['joint_pos'])
            row[label]['planned_horizontal_path_m'] = float(np.linalg.norm(goals[:, 2*nj+6:2*nj+8], axis=-1).sum()*.02)
        stops.append(row)
    motions = []
    a = [s for s in base['segments'] if s[0] != 'stand']
    b = [s for s in candidate['segments'] if s[0] != 'stand']
    for i, (old, new) in enumerate(zip(a, b)):
        if old[0] != new[0]:
            raise ValueError('Motion skill sequence differs')
        duration = min(old[2]-old[1], new[2]-new[1])-.5
        if duration < .3:
            continue
        motions.append(dict(segment=i+1, skill=old[0], start_excluded_s=.5,
                            baseline=metrics(base, old[1]+.5, duration),
                            candidate=metrics(candidate, new[1]+.5, duration)))
    long_stand = None
    final_old, final_new = base['segments'][-1], candidate['segments'][-1]
    # Reaching a stand segment early does not establish final-hold coverage.
    last_release = max(t for t, mode in base['result']['schedule'] if mode == 'stand')
    if final_old[0] == final_new[0] == 'stand' and final_new[1] >= last_release-.15:
        duration = min(final_old[2]-final_old[1], final_new[2]-final_new[1])-5.
        if duration > .3:
            long_stand = dict(settling_excluded_s=5., baseline=metrics(base, final_old[1]+5., duration),
                              candidate=metrics(candidate, final_new[1]+5., duration))
    return dict(stops=stops, motion_segments=motions, long_stand=long_stand)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('baseline','candidate','out'):
        p.add_argument('--'+key, type=Path, required=True)
    args=p.parse_args()
    if args.out.exists():
        raise ValueError('Output exists')
    a,b=read_trial(args.baseline),read_trial(args.candidate)
    if a['result']['schedule'] != b['result']['schedule']:
        raise ValueError('Requested schedules differ')
    report=compare(a,b)
    report['provenance']={name:dict(path=str(d['path']),frames_sha256=sha(d['path']/'frames.jsonl'),
                                  failure=d['result']['failure'],control_seconds=d['result']['control_seconds'],
                                  resets=d['result']['resets_after_start'],anchor_obs=d['result'].get('anchor_obs'))
                          for name,d in [('baseline',a),('candidate',b)]}
    report['method']=dict(resample_dt_s=.02, velocity_derivative_span_s=.2,
                          root_stop_speed_mps=.05, root_stop_yaw_rate_radps=.1,root_stop_hold_s=1.,
                          contact_force_available=False,
                          scope='Paired deterministic rollout, not acceptance or generalization; path length includes odometry jitter')
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('stops','motion_segments')},indent=2))


if __name__ == '__main__':
    main()
