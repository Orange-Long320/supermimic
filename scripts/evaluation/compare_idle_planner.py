"""Compare recorded Walk/Idle trials, including retained walking and equal holds."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    outputs = {}
    for planner in ('baseline', 'context-idle'):
        folder = a.run/planner
        audit = json.loads((folder/'audit.json').read_text())
        result = json.loads((folder/'policy/result.json').read_text())
        rows = [json.loads(line) for line in (folder/'policy/frames.jsonl').open()]
        times = np.array([r['elapsed'] for r in rows])
        positions = np.array([r['root'] for r in rows])
        modes = np.array([r['mode'] for r in rows])
        yaw = Rotation.from_quat(np.array([r['q'] for r in rows])[:, [1, 2, 3, 0]]).as_euler('xyz')[:, 2]
        starts = np.flatnonzero((modes[:-1] == 'stand') & (modes[1:] == 'walk'))+1
        ends = np.flatnonzero((modes[:-1] == 'walk') & (modes[1:] == 'stand'))+1
        if len(starts) != 3 or len(ends) != 3 or len(audit['stops']) != 3:
            raise ValueError('Expected three complete Walk/Idle transitions')
        velocity = np.diff(positions[:, :2], axis=0)/np.diff(times)[:, None]
        forward = np.c_[np.cos(yaw[:-1]), np.sin(yaw[:-1])]
        vx = np.sum(velocity*forward, axis=1)
        movement = []
        for start, end in zip(starts, ends):
            mask = (times[:-1] >= times[start]+1.) & (times[1:] < times[end])
            movement.append(dict(duration_s=float(times[end]-times[start]),
                                 mean_forward_speed_after_first_second_m_s=float(np.average(vx[mask], weights=np.diff(times)[mask]))))
        # Compare equal 10-second windows starting 5 seconds after the last release.
        begin = times[ends[-1]]+5.
        ids = np.flatnonzero((times >= begin) & (times < begin+10.))
        if len(ids) < 2 or times[ids[-1]] < begin+9.95:
            raise ValueError('Incomplete common hold window')
        displacement = np.linalg.norm(positions[ids, :2]-positions[ids[0], :2], axis=1)
        outputs[planner] = dict(failure=result['failure'], resets=result['resets_after_start'],
            control_seconds=result['control_seconds'], future_overlap_error=result['max_future_overlap_error'],
            movement=movement, stops=audit['stops'],
            common_hold=dict(requested_seconds=10., measured_seconds=float(times[ids[-1]]-times[ids[0]]),
                             net_displacement_m=float(displacement[-1]), max_displacement_m=float(displacement.max())),
            planned_transitions=[e for e in result['phase_events'] if e.get('kind') == 'context_transition'],
            frames_sha256=hashlib.sha256((folder/'policy/frames.jsonl').read_bytes()).hexdigest())
    old, new = outputs['baseline'], outputs['context-idle']
    improved_stops = [b['displacement_4s_m'] < x['displacement_4s_m'] for x, b in zip(old['stops'], new['stops'])]
    retained_movement = all(b['mean_forward_speed_after_first_second_m_s'] >= .9*x['mean_forward_speed_after_first_second_m_s']
                            for x, b in zip(old['movement'], new['movement']))
    better_hold = new['common_hold']['max_displacement_m'] <= old['common_hold']['max_displacement_m']
    favorable = bool(all(improved_stops) and retained_movement and better_hold)
    report = dict(trials=outputs, stop_distance_improved_by_trial=improved_stops,
                  movement_within_ten_percent_of_baseline=retained_movement,
                  common_hold_maximum_drift_not_worse=better_hold,
                  conclusion=('Initial metrics favorable; acceptance still unverified.' if favorable
                              else 'Initial comparison does not show consistent improvement; candidate not adopted.'),
                  physics_accepted=False,
                  scope='One continuous sequence per planner with equal inputs/initial setup; different cycle joins and wall-clock ROS timing, not a causal stop-entry-only ablation; geometric foot gaps are not contact force')
    a.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({name:dict(stops_4s_m=[s['displacement_4s_m'] for s in trial['stops']],
                               forward_m_s=[m['mean_forward_speed_after_first_second_m_s'] for m in trial['movement']],
                               common_hold=trial['common_hold']) for name, trial in outputs.items()}, indent=2))


if __name__ == '__main__':
    main()
