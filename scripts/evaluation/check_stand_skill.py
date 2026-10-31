"""Check standing feature compatibility and five continuous future windows on CPU."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('output exists')
    from supermimic.deploy.skill_goal import SkillGoal
    from supermimic.envs.mdp.motion_library import MotionLibrary
    spec = json.loads(a.profile.read_text())
    default = np.array([spec['default_joint_pos'][n] for n in spec['joint_names']])
    lib = MotionLibrary([a.dataset/'walk_stop.npz', a.dataset/'stand.npz'],
                        tuple(spec['tracked_bodies']), default, device='cpu')
    stand_diff = float((lib.goal_feat[lib.clips[0].end-1] - lib.goal_feat[lib.clips[1].start]).abs().max())
    if stand_diff > 1e-5:
        raise ValueError('stand feature differs from existing terminal reference')
    skills = {}
    for name in ('walk', 'backward', 'turn_left_inplace', 'turn_right_inplace', 'jog'):
        goal = SkillGoal(a.dataset, a.profile, [0, 0, .65], [1, 0, 0, 0])
        previous = None
        maximum = 0.
        for mode, count in ((name, 130), ('stand', 100)):
            for _ in range(count):
                goal.goal([0, 0, .65], [1, 0, 0, 0], default, mode)
                window = goal.scheduler.window().copy()
                if previous is not None:
                    maximum = max(maximum, float(np.max(abs(previous[1:] - window[:-1]))))
                previous = window
                goal.advance()
        stand_error = float(np.max(abs(previous - goal.scheduler.stand)))
        if maximum > 1e-6 or stand_error > 1e-5:
            raise ValueError('Reference transition check failed: ' + name)
        skills[name] = {'future_overlap_error': maximum, 'terminal_stand_error': stand_error}
    report = {'stand_feature_vs_existing_terminal_max_error': stand_diff, 'skills': skills,
              'scope': 'kinematic reference and scheduler check; not a physics or policy test'}
    a.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
