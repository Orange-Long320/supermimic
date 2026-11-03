"""Offline: is the policy's waist pitch command sane at the nominal standing state?

If the policy asks for near-default actions at the state it is supposed to stand
in, then the -1.96 rad waist target seen on hardware comes from an off-nominal
observation (or goal), not from a broken policy or a wrong action scale.

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
    fixture = json.load(open(plan['policy_fixture_path']))

    from supermimic.deploy.numpy_policy import NumpyPolicy
    policy = NumpyPolicy(plan['numpy_policy_path'], expected_sha256=plan['numpy_policy_sha256'])
    goal = np.asarray(fixture['goal'], dtype=np.float32)

    def term_major(q_hist, qd_hist, w_hist, g_hist, prev_hist):
        return np.concatenate([np.asarray(x).ravel() for x in (q_hist, qd_hist, w_hist, g_hist, prev_hist)]).astype(np.float32)

    def build(q, qd, w, g, prev):
        q = np.tile(np.asarray(q, dtype=float), (STEPS, 1))
        qd = np.tile(np.asarray(qd, dtype=float), (STEPS, 1))
        w = np.tile(np.asarray(w, dtype=float), (STEPS, 1))
        g = np.tile(np.asarray(g, dtype=float), (STEPS, 1))
        prev = np.tile(np.asarray(prev, dtype=float), (STEPS, 1))
        return term_major(q, qd, w, g, prev)

    def show(tag, obs):
        action = policy.infer(obs, goal)
        desired = default + scale * action
        out = (desired < lower) | (desired > upper)
        i = names.index('waist_pitch_joint')
        print('%-28s max|action| %6.3f  max|desired| %6.3f  out-of-range %2d  waist action %+7.3f desired %+7.3f'
              % (tag, float(np.max(np.abs(action))), float(np.max(np.abs(desired))),
                 int(out.sum()), float(action[i]), float(desired[i])))
        if out.any():
            worst = np.flatnonzero(out)
            print('    out of range:', ', '.join('%s %+.3f' % (names[j], desired[j]) for j in worst[:6]))
        return action, desired

    # 1. The exact nominal standing state the policy is trained to hold.
    zeros = np.zeros(n)
    gravity = np.array([0., 0., -1.])
    show('nominal stand, zero prev', build(zeros, zeros, np.zeros(3), gravity, zeros))
    show('nominal stand, fixture goal only', np.asarray(fixture['obs'], dtype=np.float32))

    # 2. The squat observed on hardware, with the previous action the controller
    #    seeds from (start - default) / scale.
    knees = names.index('left_knee_joint'), names.index('right_knee_joint')
    ankles = names.index('left_ankle_pitch_joint'), names.index('right_ankle_pitch_joint')
    q_squat = zeros.copy()
    q_squat[list(knees)] = 0.36
    q_squat[list(ankles)] = -0.44
    prev_squat = q_squat / scale
    show('squat offsets, seeded prev', build(q_squat, zeros, np.zeros(3), gravity, prev_squat))

    # 3. How large is the goal input that a correct handover would use?
    print('\nfixture goal |max| %.3f  mean %.3f' % (float(np.max(np.abs(goal))), float(np.mean(goal))))
    print('waist_pitch: default %+.4f scale %.4f range [%+.4f, %+.4f]'
          % (default[names.index('waist_pitch_joint')], scale[names.index('waist_pitch_joint')],
             lower[names.index('waist_pitch_joint')], upper[names.index('waist_pitch_joint')]))


if __name__ == '__main__':
    main()
