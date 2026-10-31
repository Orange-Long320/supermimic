"""CPU reference-only checks for the Walk/Idle planner; no physical acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'profile', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    from supermimic.deploy.skill_goal import SkillGoal
    spec = json.loads(a.profile.read_text())
    default = np.array([spec['default_joint_pos'][n] for n in spec['joint_names']])
    rows = []
    for duration in (3.6, 4., 4.4):
        goal = SkillGoal(a.dataset, a.profile, [0, 0, .65], [1, 0, 0, 0], planner='context-idle')
        schedule = [(0., 'stand'), (3., 'walk'), (3.+duration, 'stand'),
                    (10.+duration, 'walk'), (13.+duration, 'stand')]
        previous = None
        overlap = 0.
        times = []
        features = []
        for step in range(round((20.+duration)/.02)):
            mode = next(m for t, m in reversed(schedule) if step*.02 >= t)
            g = goal.scheduler.window()[0]
            begin = time.perf_counter()
            goal.goal([0, 0, g[-1]], [1, 0, 0, 0], g[:len(default)]+default, mode,
                      joint_velocity=g[len(default):2*len(default)])
            times.append(time.perf_counter()-begin)
            window = goal.scheduler.window()
            if previous is not None:
                overlap = max(overlap, float(np.max(np.abs(previous[1:]-window[:-1]))))
            previous = window.copy()
            features.append(window[0].copy())
            goal.advance()
        f = np.asarray(features)
        n = len(default)
        limits = goal.scheduler.joint_limits
        hard = float(np.maximum(np.maximum(limits[:, 0]-f[:, :n], f[:, :n]-limits[:, 1]), 0).max())
        speed = float(np.max(np.abs(np.diff(f[:, :n], axis=0)/.02)/goal.scheduler.velocity_limits))
        events = goal.scheduler.events
        transitions = [e for e in events if e['kind'] == 'context_transition']
        idle_error = float(np.max(np.abs(goal.scheduler.window()-goal.scheduler.stand)))
        report = dict(walk_duration_s=duration, max_overlap_error=overlap, hard_excess_rad=hard,
                      finite_difference_speed_ratio=speed, final_idle_error=idle_error,
                      max_goal_call_s=max(times), expected_transitions=4, events=events)
        report['passed'] = bool(overlap == 0 and hard <= 1e-4 and speed <= 1 and idle_error <= 1e-7
                                and len(transitions) == 4)
        np.savez_compressed(a.out/f'reference-{duration:.1f}.npz', features=f)
        rows.append(report)
    result = dict(passed=all(r['passed'] for r in rows), cases=rows,
                  dataset_manifest_sha256=hashlib.sha256((a.dataset/'manifest.json').read_bytes()).hexdigest(),
                  scope='Ideal reference feedback, CPU kinematics only; no simulation or contact/dynamics acceptance')
    (a.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'cases'}))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
