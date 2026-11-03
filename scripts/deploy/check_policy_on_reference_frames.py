"""Offline: run the policy on real reference-trajectory frames.

The dataset holds no recorded actions, so the test is: feed ten consecutive
reference frames as the observation, and compare the resulting target with the
reference joint angles themselves. On in-distribution data a tracking policy
should ask for something close to the pose it is looking at; a large gap means
the observation construction, the goal, or the exported scale does not match
what the policy was trained on.

Read-only: no robot, no ROS, no motor path.
"""
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation

STEPS = 10


def main():
    release = sys.argv[1]
    candidate = json.load(open(release + '/startup-candidate.json'))
    profile = json.load(open(release + '/fk-only/profile.json'))
    names = candidate['joint_names']
    n = len(names)
    default = np.asarray(candidate['action_default'], dtype=float)
    scale = np.asarray(candidate['action_scale'], dtype=float)
    lower = np.asarray(candidate['pc2_urdf_lower'], dtype=float)
    upper = np.asarray(candidate['pc2_urdf_upper'], dtype=float)

    import hashlib
    from supermimic.deploy.numpy_policy import NumpyPolicy
    from supermimic.deploy.skill_goal import SkillGoal

    npz = release + '/numpy-policy-11500.npz'
    sha = hashlib.sha256(open(npz, 'rb').read()).hexdigest()
    policy = NumpyPolicy(npz, expected_sha256=sha)

    data = np.load(release + '/dataset/walk_stop.npz', allow_pickle=False)
    q_all = data['joint_pos'].astype(float)
    qd_all = data['joint_vel'].astype(float)
    pos_all = data['body_pos_w'].astype(float)
    quat_all = data['body_quat_w'].astype(float)
    w_all = data['body_ang_vel_w'].astype(float)
    names_npz = [str(x) for x in data['joint_names']]
    assert names_npz == list(names), 'joint order mismatch'
    pelvis = [str(b) for b in data['body_names']].index('pelvis')

    tracker = SkillGoal(release + '/dataset', release + '/fk-only/profile.json',
                        pos_all[0, pelvis].tolist(), quat_all[0, pelvis].tolist())

    def gravity_of(quat_wxyz):
        return Rotation.from_quat(np.r_[quat_wxyz[1:], quat_wxyz[0]]).inv().apply([0., 0., -1.])

    def ang_vel_body(quat_wxyz, world):
        return Rotation.from_quat(np.r_[quat_wxyz[1:], quat_wxyz[0]]).inv().apply(world)

    def run(start, prev_mode):
        frames = range(start, start + STEPS)
        blocks = []
        for k in frames:
            quat = quat_all[k, pelvis]
            prev = ((q_all[k] - default) / scale) if prev_mode == 'pose' else np.zeros(n)
            blocks.append((q_all[k] - default, qd_all[k],
                           ang_vel_body(quat, w_all[k, pelvis]), gravity_of(quat), prev))
        obs = np.concatenate([b[t] for t in range(5) for b in blocks]).astype(np.float32)
        q_now = q_all[frames[-1]]
        quat_now = quat_all[frames[-1], pelvis]
        goal = np.asarray(tracker.goal(pos_all[frames[-1], pelvis], quat_now, q_now, 'stand'), dtype=np.float32)
        action = policy.infer(obs, goal)
        desired = default + scale * action
        gap = np.abs(desired - q_now)
        out = (desired < lower) | (desired > upper)
        i = names.index('waist_pitch_joint')
        print('  window %3d-%3d prev=%-5s |desired-q| p50 %6.3f max %6.3f  out-of-range %2d  waist %+6.3f'
              % (frames[0], frames[-1], prev_mode, float(np.median(gap)), float(gap.max()),
                 int(out.sum()), float(desired[i])))
        if out.any():
            worst = np.argsort(-gap)[:3]
            print('       worst joints:', ', '.join(
                '%s gap %+.2f desired %+.2f range [%+.2f,%+.2f]' % (names[j], desired[j] - q_now[j], desired[j], lower[j], upper[j])
                for j in worst))

    total = q_all.shape[0]
    windows = [0, total // 4, total // 2, total - STEPS]
    print('reference frames', total, ' windows tested', windows)
    for prev_mode in ('pose', 'zero'):
        print('prev action = %s' % ('(q-default)/scale' if prev_mode == 'pose' else 'zeros'))
        for w in windows:
            run(w, prev_mode)


if __name__ == '__main__':
    main()
