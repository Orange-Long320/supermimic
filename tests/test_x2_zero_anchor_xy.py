"""Check the experimental 729-dimensional standing goal intervention."""
import unittest
import numpy as np
from supermimic.deploy.early_handover import EarlyHandover, Phase


class Tracker:
    def __init__(self, goal): self.value = goal
    def goal(self, position, quat, joints, mode):
        if mode != 'stand': raise AssertionError('unexpected skill')
        return self.value
    def advance(self): pass


class Policy:
    def __init__(self): self.goal = None
    def infer(self, obs, goal): self.goal = np.asarray(goal).copy(); return np.zeros(31)


class ZeroAnchorTests(unittest.TestCase):
    def controller(self, goal, enabled, unclipped=()):
        z = np.zeros(31); ones = np.ones(31); policy = Policy(); tracker = Tracker(goal)
        c = EarlyHandover(z, z, z, ones, ones, ones, ones, ones, -ones, ones, -ones, ones, ones,
            max_velocity=ones, max_acceleration=ones, minimum_duration=.2, blend_seconds=.1,
            max_tick_gap=.031, max_tilt=.5, max_angular_speed=1., policy=policy,
            tracker_factory=lambda p, q: tracker, zero_stand_anchor_xy=enabled,
            unclipped_target_indices=unclipped)
        return c, policy

    @staticmethod
    def step(c, t):
        z=np.zeros(31)
        return c.step(t, [0,0,.6], [1,0,0,0], z, z, [0,0,0], [0,0,-1])

    def test_only_xy_of_stand_anchor_changes_and_input_is_not_mutated(self):
        goal = np.arange(729, dtype=np.float32)
        for enabled in (False, True):
            source = goal.copy(); c, policy = self.controller(source, enabled)
            for i in range(10): self.step(c, i*.02)
            c.request_handover(); self.step(c, .2)
            expected = goal.copy()
            if enabled: expected[-9:-7] = 0.
            np.testing.assert_array_equal(policy.goal, expected)
            np.testing.assert_array_equal(source, goal)

    def test_wrong_goal_shape_faults_before_policy_inference(self):
        c, policy = self.controller(np.arange(720), True)
        for i in range(10): self.step(c, i*.02)
        c.request_handover()
        with self.assertRaisesRegex(ValueError, '729-element'): self.step(c, .2)
        self.assertIsNone(policy.goal)
        self.assertEqual(c.phase, Phase.FAULT)

    def test_only_selected_joint_bypasses_range_clamp(self):
        c, policy = self.controller(np.arange(729), True, unclipped=(13,))
        action = np.zeros(31); action[13] = -5.; action[14] = -5.
        policy.infer = lambda obs, goal: action
        for i in range(10): self.step(c, i*.02)
        c.request_handover()
        for i in range(10, 22): out = self.step(c, i*.02)
        self.assertAlmostEqual(out['position'][13], -5.)
        self.assertAlmostEqual(out['position'][14], -1.)

    def test_bad_unclipped_indices_rejected(self):
        for indices in ((31,), (-1,), (13,13), (True,)):
            with self.assertRaisesRegex(ValueError,'indices'):
                self.controller(np.arange(729), True, unclipped=indices)

    def test_all_selected_joint_targets_are_unclipped(self):
        c, policy = self.controller(np.arange(729), True, unclipped=tuple(range(31)))
        action = np.zeros(31); action[13] = -5.; action[14] = 4.; action[0] = 3.
        policy.infer = lambda obs, goal: action
        for i in range(10): self.step(c, i*.02)
        c.request_handover()
        for i in range(10, 22): out = self.step(c, i*.02)
        np.testing.assert_allclose(out['position'][[0,13,14]],[3.,-5.,4.])


if __name__ == '__main__': unittest.main()
