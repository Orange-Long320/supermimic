import unittest
from dataclasses import replace
from supermimic.deploy.core import Contract, DeploymentError, JointSample, Mode, ShadowController, State


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


if __name__ == '__main__': unittest.main()
