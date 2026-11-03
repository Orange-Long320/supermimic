"""Offline ablation: is the elbow command a sign-convention error?

At the nominal standing state the policy commands the elbows to -2.4 rad, which
is its full negative travel, while a standing skill should sit near the trained
offset (-0.6). Either the observation is off-distribution, or one joint's sign
convention differs between training and this deployment.

Each variant below changes exactly one hypothesis and reports the elbow action.

Read-only: no robot, no ROS, no motor path.
"""
import json
import sys

import numpy as np

STEPS = 10


def main():
    trial, release = sys.argv[1], sys.argv[2]
    plan = json.load(open(trial + '/contract/plan.json'))
    candidate = json.load(open(release + '/startup-candidate.json'))
    names = candidate['joint_names']
    n = len(names)
    default = np.asarray(candidate['action_default'], dtype=float)
    scale = np.asarray(candidate['action_scale'], dtype=float)
    lower = np.asarray(candidate['pc2_urdf_lower'], dtype=float)
    upper = np.asarray(candidate['pc2_urdf_upper'], dtype=float)

    from supermimic.deploy.numpy_policy import NumpyPolicy
    from supermimic.deploy.skill_goal import SkillGoal
    policy = NumpyPolicy(plan['numpy_policy_path'], expected_sha256=plan['numpy_policy_sha256'])
    tracker = SkillGoal(release + '/dataset', release + '/fk-only/profile.json',
                        candidate['startup']['root_position'], candidate['startup']['root_quat_wxyz'])
    goal = np.asarray(tracker.goal(candidate['startup']['root_position'],
                                   candidate['startup']['root_quat_wxyz'],
                                   candidate['startup']['position'], 'stand'), dtype=np.float32)

    elbow = [names.index('left_elbow_joint'), names.index('right_elbow_joint')]
    zeros = np.zeros(n)
    gravity = np.array([0., 0., -1.])

    def obs(q, qd, w, g, prev):
        def tile(x):
            return np.tile(np.asarray(x, dtype=float), (STEPS, 1)).ravel()
        return np.concatenate([tile(q), tile(qd), tile(w), tile(g), tile(prev)]).astype(np.float32)

    def variant(tag, g, prev=None):
        # At the nominal state the only non-zero inputs are gravity and the goal,
        # so sweeping the goal isolates what drives the elbow command.
        zeros_ = zeros if prev is None else prev
        action = policy.infer(obs(zeros, zeros, np.zeros(3), g, zeros_), goal_v)
        desired = default + scale * action
        el = [-float(action[i]) for i in elbow]
        print('%-40s elbow action L/R %+7.2f / %+7.2f   waist action %+6.2f   desired elbow %+6.2f'
              % (tag, el[0], el[1], float(action[names.index('waist_pitch_joint')]),
                 float(desired[elbow[0]])))
        return action

    fixture = json.load(open(plan['policy_fixture_path']))
    fgoal = np.asarray(fixture['goal'], dtype=np.float32)
    for tag, goal_v in (('goal = zero', np.zeros(729, dtype=np.float32)),
                        ('goal = 0.25 x real', (0.25 * goal).astype(np.float32)),
                        ('goal = real (deployment)', goal),
                        ('goal = fixture (synthetic)', fgoal)):
        variant(tag, gravity)

    print()
    variant('real goal, gravity sign flipped', -gravity)
    variant('real goal, gravity zeroed', np.zeros(3))

    print()
    for target in (-0.6, -1.0, -1.5, -2.0, -2.356):
        needed = (target - default[elbow[0]]) / scale[elbow[0]]
        print('elbow target %+6.3f rad needs action %+6.2f   (range allows down to %+6.2f)'
              % (target, needed, (lower[elbow[0]] - default[elbow[0]]) / scale[elbow[0]]))


if __name__ == '__main__':
    main()
