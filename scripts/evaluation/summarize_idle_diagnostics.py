"""Summarize complete holds without mixing later walking into failed trials."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def summarize(folder, hold_seconds=10.):
    result = json.loads((folder/'policy/result.json').read_text())
    frames = folder/'policy/frames.jsonl'
    rows = [json.loads(line) for line in frames.open()]
    t = np.array([r['elapsed'] for r in rows])
    mode = np.array([r['mode'] for r in rows])
    pos = np.array([r['root'] for r in rows])
    qvel = np.array([r['joint_vel'] for r in rows])
    yaw = Rotation.from_quat(np.array([r['q'] for r in rows])[:,[1,2,3,0]]).as_euler('xyz')[:,2]
    starts = np.flatnonzero((mode[:-1] == 'stand') & (mode[1:] == 'walk'))+1
    ends = np.flatnonzero((mode[:-1] == 'walk') & (mode[1:] == 'stand'))+1
    speed = np.diff(pos[:,:2],axis=0)/np.diff(t)[:,None]
    vx = np.sum(speed*np.c_[np.cos(yaw[:-1]), np.sin(yaw[:-1])],axis=1)
    movement = []
    for start in starts:
        stops = ends[ends > start]
        if not len(stops):
            continue
        end = stops[0]
        mask = (t[:-1] >= t[start]+1.) & (t[1:] < t[end])
        movement.append(float(np.average(vx[mask],weights=np.diff(t)[mask])))
    holds = []
    for start in ends:
        next_motion = np.flatnonzero(mode[start:] != 'stand')
        end = start+int(next_motion[0]) if len(next_motion) else len(rows)
        h = np.flatnonzero((t >= t[start]+5.) & (t < t[start]+5.+hold_seconds) & (np.arange(len(t)) < end))
        item = dict(release_s=float(t[start]), observed_stand_after_release_s=float(t[end-1]-t[start]),
                    requested_hold_seconds=hold_seconds, full_ten_second_hold=False, full_requested_hold=False)
        if t[end-1] >= t[start]+4.:
            i = int(np.searchsorted(t, t[start]+4.))
            item['stop_displacement_4s_m'] = float(np.linalg.norm(pos[i,:2]-pos[start,:2]))
        if len(h) > 1:
            distance = np.linalg.norm(pos[h,:2]-pos[h[0],:2],axis=1)
            dq = np.sqrt(np.mean(qvel[h]**2,axis=1))
            item.update(hold_seconds=float(t[h[-1]]-t[h[0]]),
                        full_ten_second_hold=bool(t[h[-1]] >= t[start]+14.95),
                        full_requested_hold=bool(t[h[-1]] >= t[start]+5.+hold_seconds-.05),
                        hold_max_displacement_m=float(distance.max()), hold_net_displacement_m=float(distance[-1]),
                        hold_peak_joint_speed_rms_rad_s=float(dq.max()),
                        hold_peak_joint_motion_after_release_s=float(t[h[dq.argmax()]]-t[start]))
        holds.append(item)
    valid = bool(result['failure'] is None and len(ends) == 3 and mode[-1] == 'stand'
                 and holds[-1]['full_requested_hold'])
    return dict(failure=result['failure'], seconds=result['control_seconds'], frames=len(rows),
                resets=result['resets_after_start'], max_overlap=result['max_future_overlap_error'],
                movement_forward_speed_m_s=movement, holds=holds, complete_terminal_hold=valid,
                terminal_hold=holds[-1] if valid else None,
                planner=result['planner'], ablation=result.get('planner_ablation'),
                anchor_mode=result.get('idle_anchor_mode','fixed'),
                velocity_feedback=result.get('idle_velocity_feedback',False),
                source_frames_sha256=hashlib.sha256(frames.read_bytes()).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--hold-seconds', type=float, default=10.)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    if a.hold_seconds <= 0:
        raise ValueError('Positive hold window required')
    results = {}
    for run in a.runs:
        comparison = json.loads((run/'comparison.json').read_text())
        for trial in comparison['trials']:
            path = run/trial['planner']
            if (path/'policy/result.json').exists():
                results[run.resolve().name+'/'+trial['planner']] = summarize(path, a.hold_seconds)
    a.out.write_text(json.dumps(dict(trials=results,
        scope='Read-only recorded-state diagnosis. Failed or truncated trials cannot qualify for the terminal hold metric. Geometry/contact-force and multi-seed acceptance are separate.'), indent=2)+'\n')
    for name, r in results.items():
        hold = r['terminal_hold']
        print(json.dumps(dict(trial=name, failure=r['failure'],
                              terminal_hold_max_cm=None if hold is None else 100*hold['hold_max_displacement_m'],
                              terminal_hold_peak_dq=None if hold is None else hold['hold_peak_joint_speed_rms_rad_s'])))


if __name__ == '__main__':
    main()
