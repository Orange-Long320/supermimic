import unittest
from dataclasses import replace
from supermimic.deploy.core import Contract, DeploymentError, JointSample, Mode, ShadowController, StartupPose, State


class Policy:
    def __init__(self): self.obs = None
    def infer(self, observation, goal):
        self.obs = observation
        return (0.1, -0.1)


class Goals:
    def goal(self, state, command): return (1., 2.)


class DeploymentCoreTest(unittest.TestCase):
    def setUp(self):
        self.spec = Contract(('a', 'b'), (0., 0.), (1., 1.), (-1., -1.), (1., 1.), (.2, .2), .1, .05, 1, 2)
        self.state = State((JointSample('b', 0., 2., 0), JointSample('a', 0., 1., 0)), (1., 1.), 1., (1., 0., 0., 0.), (0., 0., 0.))
        self.policy = Policy()
        self.controller = ShadowController(self.spec, self.policy, Goals())
        self.controller.start()

    def test_reorder_and_observation(self):
        self.assertEqual(self.controller.step(self.state, 'stand', 1.01), (.1, -.1))
        self.assertEqual(self.policy.obs, (0., 0., 1., 2., 0., 0., 0., 0., 0., -1., 0., 0.))

    def test_fault_latches_until_reset(self):
        with self.assertRaises(DeploymentError): self.controller.step(self.state, 'stand', 1.2)
        self.assertEqual(self.controller.mode, Mode.FAULT)
        with self.assertRaises(DeploymentError): self.controller.step(self.state, 'stand', 1.01)
        self.controller.stop()
        self.controller.start()
        self.controller.step(self.state, 'stand', 1.01)

    def test_term_major_history(self):
        spec = replace(self.spec, history=2, max_target_step=(.4, .4))
        controller = ShadowController(spec, self.policy, Goals()); controller.start()
        self.assertIsNone(controller.step(self.state, 'stand', 1.01))
        state = replace(self.state, joints=(JointSample('b', .2, 4., 0), JointSample('a', .1, 3., 0)))
        controller.step(state, 'stand', 1.01)
        self.assertEqual(self.policy.obs[:8], (0., 0., .1, .2, 1., 2., 3., 4.))

    def test_invalid_states_and_targets(self):
        bad = [replace(self.state, imu_time=1.2), replace(self.state, orientation_wxyz=(0., 0., 0., 0.)),
               replace(self.state, joints=(self.state.joints[0], self.state.joints[0])),
               replace(self.state, joints=(replace(self.state.joints[0], error_code=1), self.state.joints[1]))]
        for state in bad:
            self.controller.stop(); self.controller.start()
            with self.assertRaises(DeploymentError): self.controller.step(state, 'stand', 1.01)
        self.controller.stop(); self.controller.start()
        self.controller.policy.infer = lambda obs, goal: (float('nan'), 0.)
        with self.assertRaises(DeploymentError): self.controller.step(self.state, 'stand', 1.01)

    def test_pd_setpoint_bounds_do_not_change_measured_bounds(self):
        spec = replace(self.spec, target_lower=(-2., -2.), target_upper=(2., 2.), max_target_step=(2., 2.))
        policy = Policy(); policy.infer = lambda obs, goal: (1.2, 0.)
        controller = ShadowController(spec, policy, Goals()); controller.start()
        self.assertEqual(controller.step(self.state, 'stand', 1.01), (1.2, 0.))
        controller.stop(); controller.start()
        bad = replace(self.state, joints=(self.state.joints[0], replace(self.state.joints[1], position=1.2)))
        with self.assertRaisesRegex(DeploymentError, 'measured joint'):
            controller.step(bad, 'stand', 1.01)

    def test_omitted_pd_bounds_still_reject_same_setpoint(self):
        self.policy.infer = lambda obs, goal: (1.2, 0.)
        with self.assertRaises(DeploymentError):
            self.controller.step(self.state, 'stand', 1.01)
        for kwargs in ({'target_lower': (-2., -2.)},
                       {'target_lower': (-2.,), 'target_upper': (2.,)},
                       {'target_lower': (0., 0.), 'target_upper': (0., 0.)}):
            with self.assertRaises(DeploymentError): replace(self.spec, **kwargs)

    def startup(self):
        return StartupPose(('a', 'b'), (.4, -.4), (.02, .02), (.05, .05), (0., 0., -1.), .05)

    def ready_state(self):
        return replace(self.state, joints=(JointSample('b', -.4, 0., 0), JointSample('a', .4, 0., 0)))

    def test_startup_uses_reference_not_action_default(self):
        controller = ShadowController(self.spec, self.policy, Goals(), startup=self.startup()); controller.start()
        quiet_default = replace(self.state, joints=tuple(replace(j, velocity=0.) for j in self.state.joints))
        self.assertIsNone(controller.step(quiet_default, 'stand', 1.01))
        self.assertIsNone(self.policy.obs)
        self.assertEqual(len(controller.history), 0)
        # Output equal to the measured reference for this synthetic test.
        self.policy.infer = lambda obs, goal: (.4, -.4)
        self.assertEqual(controller.step(self.ready_state(), 'stand', 1.01), (.4, -.4))

    def test_startup_requires_contiguous_real_history(self):
        controller = ShadowController(replace(self.spec, history=2), self.policy, Goals(), startup=self.startup())
        controller.start(); ready = self.ready_state()
        self.assertIsNone(controller.step(ready, 'stand', 1.01))
        moving = replace(ready, joints=tuple(replace(j, velocity=.1) for j in ready.joints))
        self.assertIsNone(controller.step(moving, 'stand', 1.01))
        self.assertEqual(len(controller.history), 0)
        self.assertIsNone(controller.step(ready, 'stand', 1.01))
        self.policy.infer = lambda obs, goal: (.4, -.4)
        self.assertEqual(controller.step(ready, 'stand', 1.01), (.4, -.4))
        controller.stop(); controller.start()
        self.assertFalse(controller.startup_complete)

    def test_startup_rejects_tilt_and_nonstand_command(self):
        controller = ShadowController(self.spec, self.policy, Goals(), startup=self.startup()); controller.start()
        tilted = replace(self.ready_state(), orientation_wxyz=(.995004165278, .099833416647, 0., 0.))
        self.assertIsNone(controller.step(tilted, 'stand', 1.01))
        self.assertIsNone(self.policy.obs)
        with self.assertRaisesRegex(DeploymentError, 'startup requires stand'):
            controller.step(self.ready_state(), 'walk', 1.01)

    def test_startup_gate_does_not_replace_motion_tracking(self):
        self.policy.infer = lambda obs, goal: (.4, -.4)
        controller = ShadowController(self.spec, self.policy, Goals(), startup=self.startup()); controller.start()
        self.assertEqual(controller.step(self.ready_state(), 'stand', 1.01), (.4, -.4))
        moved = replace(self.ready_state(), joints=(JointSample('b', -.8, 0., 0), JointSample('a', .8, 0., 0)))
        self.assertEqual(controller.step(moved, 'walk', 1.01), (.4, -.4))

    def test_startup_invalid_reference_is_not_silently_adjusted(self):
        for startup in (replace(self.startup(), position=(2., 0.)),
                        replace(self.startup(), names=('b', 'a')),
                        replace(self.startup(), position_tolerance=(0., .02)),
                        replace(self.startup(), gravity=(0., 0., 0.))):
            with self.assertRaises(DeploymentError):
                ShadowController(self.spec, self.policy, Goals(), startup=startup)


if __name__ == '__main__': unittest.main()
