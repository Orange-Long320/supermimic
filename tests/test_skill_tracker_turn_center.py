"""The physics wrapper must register turn XY in environment-local coordinates."""
from types import SimpleNamespace

import numpy as np
import torch

from supermimic.control.skill_phase import SkillPhase
from supermimic.control.skill_tracker import SkillTracker
from supermimic.robots.dimensions import NJ


def test_turn_center_accounts_for_environment_origin_and_remains_fixed(monkeypatch):
    tracker = SkillTracker.__new__(SkillTracker)
    tracker.anchor_obs = True
    tracker.origin = np.array([5., 7., 0.])
    tracker.pos = np.array([8., -4., .65])
    tracker.yaw = 0.
    tracker.joint_default = np.zeros(NJ)
    tracker.anchor = 0
    tracker.model = object()
    tracker.ref = SimpleNamespace(qpos=np.zeros(7+NJ), xpos=np.zeros((1, 3)),
                                  xquat=np.array([[1., 0., 0., 0.]]))
    qpos = torch.zeros((1, 7+NJ))
    qpos[0, :3] = torch.tensor([15., 27., .65])
    tracker.env = SimpleNamespace(device='cpu', sim=SimpleNamespace(data=SimpleNamespace(qpos=qpos)))
    tracker.cmd = SimpleNamespace(robot_anchor_pos_w=qpos[:, :3],
                                  robot_anchor_quat_w=torch.tensor([[1., 0., 0., 0.]]))
    row = np.zeros(2*NJ+10, dtype=np.float32)
    row[2*NJ+2], row[-1] = -1., .65
    clips = {n: np.tile(row, (50, 1)) for n in ('walk', 'turn_left_inplace')}
    tracker.scheduler = SkillPhase(clips, clips, {n: 0 for n in clips}, NJ)

    def kinematics(model, data):
        data.xpos[0] = data.qpos[:3]

    monkeypatch.setattr('supermimic.control.skill_tracker.mujoco.mj_kinematics', kinematics)
    goal = tracker.goal('turn_left_inplace')
    np.testing.assert_array_equal(tracker.pos[:2], [10., 20.])
    np.testing.assert_allclose(goal[0, -9:-7].numpy(), [0., 0.])
    qpos[0, 0] += .25
    goal = tracker.goal('turn_left_inplace')
    np.testing.assert_array_equal(tracker.pos[:2], [10., 20.])
    np.testing.assert_allclose(goal[0, -9:-7].numpy(), [-.25, 0.])
