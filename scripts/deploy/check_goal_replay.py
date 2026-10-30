"""Compare standalone goal construction with recorded simulator tracker goals."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--trajectory', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('output must be new')
    from supermimic.deploy.skill_goal import SkillGoal
    with np.load(a.trajectory, allow_pickle=False) as z:
        states, expected, modes = z['states'], z['goals'], z['modes']
    goal = SkillGoal(a.dataset, a.profile, states[0, :3], states[0, 3:7])
    errors = []
    for state, target, mode in zip(states, expected, modes):
        actual = goal.goal(state[:3], state[3:7], state[7:7+goal.nj], str(mode))
        errors.append(np.abs(actual-target))
        goal.advance()
    errors = np.stack(errors)
    report = {'frames': len(errors), 'max_error': float(errors.max()),
              'future_max_error': float(errors[:, :-9].max()),
              'anchor_position_max_error': float(errors[:, -9:-6].max()),
              'anchor_orientation_max_error': float(errors[:, -6:].max()),
              'source': str(a.trajectory), 'scope': 'recorded simulator replay only'}
    a.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
