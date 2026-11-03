"""Offline: evaluate the policy with the goal the deployment actually builds.

The fixture ships a synthetic 729-d goal, and the network is strongly goal
sensitive, so a nominal-state test with that goal proves little. This builds the
real SkillGoal from the release dataset/profile at the declared startup pose and
evaluates the 31-dim action it implies.

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

    q0 = np.asarray(candidate['startup']['position'], dtype=float)
    goal = np.asarray(tracker.goal(candidate['startup']['root_position'],
                                   candidate['startup']['root_quat_wxyz'], q0, 'stand'), dtype=np.float32)
    fixture = json.load(open(plan['policy_fixture_path']))
    fgoal = np.asarray(fixture['goal'], dtype=np.float32)

    def obs_term_major(q, qd, w, g, prev):
        def tile(x):
            return np.tile(np.asarray(x, dtype=float), (STEPS, 1))
        return np.concatenate([tile(q).ravel(), tile(qd).ravel(), tile(w).ravel(),
                               tile(g).ravel(), tile(prev).ravel()]).astype(np.float32)

    def run(tag, obs, g):
        action = policy.infer(obs, g)
        desired = default + scale * action
        out = (desired < lower) | (desired > upper)
        i = names.index('waist_pitch_joint')
        print('%-30s max|action| %6.3f  max|desired| %6.3f  out-of-range %2d  waist desired %+7.3f'
              % (tag, float(np.max(np.abs(action))), float(np.max(np.abs(desired))), int(out.sum()),
                 float(desired[i])))
        if out.any():
            print('    ', ', '.join('%s %+.2f' % (names[j], desired[j]) for j in np.flatnonzero(out)[:8]))

    print('goal: real |max| %.3f mean %+.4f | fixture |max| %.3f mean %+.4f | cosine %.3f'
          % (float(np.max(np.abs(goal))), float(np.mean(goal)),
             float(np.max(np.abs(fgoal))), float(np.mean(fgoal)),
             float(np.dot(goal, fgoal) / (np.linalg.norm(goal) * np.linalg.norm(fgoal) + 1e-12))))
    print('goal == fixture? %s' % bool(np.allclose(goal, fgoal)))

    zeros = np.zeros(n)
    gravity = np.array([0., 0., -1.])
    run('nominal obs + real goal', obs_term_major(zeros, zeros, np.zeros(3), gravity, zeros), goal)
    run('nominal obs + fixture goal', obs_term_major(zeros, zeros, np.zeros(3), gravity, zeros), fgoal)

    # The state the handover would actually start from: measured pose, standing tilt.
    qi = np.asarray(q0, dtype=float) - default
    prev = qi / scale
    run('startup offsets + real goal', obs_term_major(qi, zeros, np.zeros(3), gravity, prev), goal)


if __name__ == '__main__':
    main()
