import unittest
import numpy as np
from supermimic.deploy.early_handover import EarlyHandover,Phase


class Policy:
    def infer(self,obs,goal):return np.array([.2])


class Tracker:
    def goal(self,position,quat,q,skill):return np.zeros(1)
    def advance(self):pass


class EarlyHandoverTests(unittest.TestCase):
    def make(self,max_step=.5):
        return EarlyHandover([0.],[.1],[0.],[1.],[20.],[2.],[40.],[4.],[-1.],[1.],[-2.],[2.],[max_step],
            max_velocity=[1.],max_acceleration=[1.],minimum_duration=.2,blend_seconds=.1,max_tick_gap=.031,
            max_tilt=.5,max_angular_speed=1.,policy=Policy(),tracker_factory=lambda p,q:Tracker())

    def tick(self,c,t,q=0.,w=(0.,0.,0.),gravity=(0.,0.,-1.)):
        return c.step(t,[0,0,.6],[1,0,0,0],[q],[0],w,gravity)

    def test_request_early_after_real_history_without_settling(self):
        c=self.make()
        with self.assertRaisesRegex(ValueError,'Ten fresh'):c.request_handover()
        for i in range(10):self.tick(c,i*.02,q=.3)
        self.assertEqual(c.phase,Phase.PREPARE);c.request_handover();prev=c.last_target.copy()
        command=self.tick(c,.20,q=.3);self.assertEqual(c.phase,Phase.HANDOVER)
        np.testing.assert_allclose(command['position'],prev)
        for i in range(11,17):command=self.tick(c,i*.02,q=.3)
        self.assertEqual(c.phase,Phase.ACTIVE);np.testing.assert_allclose(command['position'],[.2])

    def test_timing_and_limits_latch(self):
        c=self.make();self.tick(c,0.)
        with self.assertRaisesRegex(ValueError,'timing'):self.tick(c,.1)
        with self.assertRaisesRegex(ValueError,'latched'):self.tick(c,.12)
        c=self.make()
        with self.assertRaisesRegex(ValueError,'Measured joint'):self.tick(c,0.,q=1.1)

    def test_bad_policy_target_and_jump_are_faults(self):
        c=self.make();
        for i in range(10):self.tick(c,i*.02)
        c.request_handover();self.tick(c,.2);c.policy.infer=lambda obs,goal:np.array([99.])
        with self.assertRaisesRegex(ValueError,'command contract'):self.tick(c,.22)


if __name__=='__main__':unittest.main()
