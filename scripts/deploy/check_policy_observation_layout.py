"""Offline check: is the 990-value observation assembled in the layout the policy expects?

The controller builds the observation term-major (every q, then every qd, then
every omega, every gravity, every previous action). A per-step layout is the
other plausible convention. Both are 990 values, so the only way to tell them
apart is to ask the policy, whose fixture fixes the correct layout.

Read-only: no robot, no ROS, no motor path.
"""
import json
import sys

import numpy as np

STEPS = 10
N = 31


def parts_step_major(obs):
    """Split a step-major observation into per-step blocks."""
    o = np.asarray(obs, dtype=np.float32)
    blocks = []
    i = 0
    for _ in range(STEPS):
        q = o[i:i + N]; i += N
        qd = o[i:i + N]; i += N
        w = o[i:i + 3]; i += 3
        g = o[i:i + 3]; i += 3
        prev = o[i:i + N]; i += N
        blocks.append((q, qd, w, g, prev))
    return blocks, i


def term_major_from_blocks(blocks):
    return np.concatenate([
        np.concatenate([b[0] for b in blocks]),
        np.concatenate([b[1] for b in blocks]),
        np.concatenate([b[2] for b in blocks]),
        np.concatenate([b[3] for b in blocks]),
        np.concatenate([b[4] for b in blocks]),
    ])


def blocks_from_term_major(obs):
    o = np.asarray(obs, dtype=np.float32)
    i = 0
    q = o[i:i + STEPS * N].reshape(STEPS, N); i += STEPS * N
    qd = o[i:i + STEPS * N].reshape(STEPS, N); i += STEPS * N
    w = o[i:i + STEPS * 3].reshape(STEPS, 3); i += STEPS * 3
    g = o[i:i + STEPS * 3].reshape(STEPS, 3); i += STEPS * 3
    prev = o[i:i + STEPS * N].reshape(STEPS, N); i += STEPS * N
    return [(q[k], qd[k], w[k], g[k], prev[k]) for k in range(STEPS)], i


def main():
    trial = sys.argv[1]
    release = sys.argv[2]
    plan = json.load(open(trial + '/contract/plan.json'))
    candidate = json.load(open(release + '/startup-candidate.json'))
    profile = json.load(open(release + '/fk-only/profile.json'))
    names = candidate['joint_names']
    default = np.asarray(candidate['action_default'], dtype=float)
    scale = np.asarray(candidate['action_scale'], dtype=float)
    fixture = json.load(open(plan['policy_fixture_path']))

    from supermimic.deploy.numpy_policy import NumpyPolicy
    policy = NumpyPolicy(plan['numpy_policy_path'], expected_sha256=plan['numpy_policy_sha256'])

    obs = np.asarray(fixture['obs'], dtype=np.float32)
    goal = np.asarray(fixture['goal'], dtype=np.float32)
    expected = np.asarray(fixture['expected'], dtype=float)

    def report(tag, action):
        desired = default + scale * action
        i = names.index('waist_pitch_joint')
        return dict(tag=tag, waist_action=float(action[i]), waist_desired=float(desired[i]),
                    max_abs_action=float(np.max(np.abs(action))),
                    max_abs_desired=float(np.max(np.abs(desired))),
                    waist_limit=float(profile['stiffness'].get('waist_pitch_joint', float('nan'))))

    print('joints', len(names), 'obs', obs.shape, 'expected', expected.shape)

    # 1. The fixture itself: the policy must reproduce the recorded output.
    fixture_action = policy.infer(obs, goal)
    print('fixture max |infer - expected| %.3e' % float(np.max(np.abs(fixture_action - expected))))

    # 2. Which layout is the fixture in? Ask the policy both ways and compare with
    #    the recorded output. Only the correct layout can reproduce it.
    blocks_step, consumed = parts_step_major(obs)
    print('step-major split consumed', consumed)
    term_version = term_major_from_blocks(blocks_step)
    action_term = policy.infer(term_version, goal)
    print('fixture reinterpreted as term-major -> max |infer - expected| %.3e'
          % float(np.max(np.abs(action_term - expected))))

    blocks_term, consumed_term = blocks_from_term_major(obs)
    print('term-major split consumed', consumed_term)
    step_version = np.concatenate([np.concatenate(b) for b in blocks_term])
    action_step = policy.infer(step_version, goal)
    print('fixture reinterpreted as step-major -> max |infer - expected| %.3e'
          % float(np.max(np.abs(action_step - expected))))

    print()
    for row in (report('fixture as recorded', fixture_action),
                report('fixture as step-major', action_step),
                report('fixture as term-major', action_term)):
        print('%-22s waist action %+8.3f  desired %+8.3f  max|action| %8.3f  max|desired| %8.3f'
              % (row['tag'], row['waist_action'], row['waist_desired'],
                 row['max_abs_action'], row['max_abs_desired']))
    print('waist_pitch in-range?', 'target %+0.4f' % default[names.index('waist_pitch_joint')])
    print('action_scale waist_pitch %.4f' % scale[names.index('waist_pitch_joint')])


if __name__ == '__main__':
    main()
