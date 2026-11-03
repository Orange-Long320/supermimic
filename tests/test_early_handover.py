import unittest
import numpy as np
from supermimic.deploy.early_handover import EarlyHandover,Phase


class Policy:
    def infer(self,obs,goal):return np.array([.2])


class Tracker:
    def goal(self,position,quat,q,skill):return np.zeros(1)
    def advance(self):pass


class EarlyHandoverTests(unittest.TestCase):
    def make(self,max_step=.5,history_source='executed',max_tilt=.5,max_tilt_delta=None):
        return EarlyHandover([0.],[.1],[0.],[1.],[20.],[2.],[40.],[4.],[-1.],[1.],[-2.],[2.],[max_step],
            max_velocity=[1.],max_acceleration=[1.],minimum_duration=.2,blend_seconds=.1,max_tick_gap=.031,
            max_tilt=max_tilt,max_angular_speed=1.,policy=Policy(),tracker_factory=lambda p,q:Tracker(),
            history_source=history_source,max_tilt_delta=max_tilt_delta)

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

    def test_relative_tilt_guard_tolerates_static_mount_tilt(self):
        def gravity(tilt):return (0.,float(np.sin(tilt)),float(-np.cos(tilt)))
        c=self.make(max_tilt=.7,max_tilt_delta=.2)
        self.tick(c,0.,gravity=gravity(.36))
        self.assertEqual(c.phase,Phase.PREPARE)
        self.tick(c,.02,gravity=gravity(.40))
        self.assertEqual(c.phase,Phase.PREPARE)
        self.assertAlmostEqual(c.tilt_reference,.36,places=6)
        with self.assertRaisesRegex(ValueError,'Tilt'):self.tick(c,.04,gravity=gravity(.60))
        absolute=self.make(max_tilt=.35)
        with self.assertRaisesRegex(ValueError,'Tilt'):self.tick(absolute,0.,gravity=gravity(.36))

    def test_timing_and_limits_latch(self):
        c=self.make();self.tick(c,0.)
        with self.assertRaisesRegex(ValueError,'timing'):self.tick(c,.1)
        with self.assertRaisesRegex(ValueError,'latched'):self.tick(c,.12)
        c=self.make()
        with self.assertRaisesRegex(ValueError,'Measured joint'):self.tick(c,0.,q=1.1)

    def test_reported_31_489ms_gap_remains_a_latched_fault(self):
        c=self.make();self.tick(c,0.)
        with self.assertRaisesRegex(ValueError,'State timing gap'):
            self.tick(c,.031489)
        self.assertEqual(c.phase,Phase.FAULT)
        with self.assertRaisesRegex(ValueError,'latched'):
            self.tick(c,.051489)

    def test_finite_policy_target_is_clamped_but_nonfinite_faults(self):
        c=self.make();
        for i in range(10):self.tick(c,i*.02)
        c.request_handover();self.tick(c,.2);previous=c.last_target.copy();c.policy.infer=lambda obs,goal:np.array([99.])
        command=self.tick(c,.22)
        self.assertLessEqual(abs(command['position'][0]-previous[0]),.5)
        self.assertEqual(c.saturated,1)
        self.assertEqual(c.rate_limited,1)
        c.policy.infer=lambda obs,goal:np.array([np.nan])
        with self.assertRaisesRegex(ValueError,'Invalid policy output'):self.tick(c,.24)
        self.assertEqual(c.phase,Phase.FAULT)

    def test_policy_history_uses_clean_output_during_blending(self):
        c=self.make(history_source='policy')
        for i in range(10):self.tick(c,i*.02)
        c.request_handover();command=self.tick(c,.2)
        self.assertLess(command['position'][0],.1)
        np.testing.assert_allclose(c.previous,[.2])
        self.tick(c,.22)
        np.testing.assert_allclose(c.history[-1][4],[.2])


if __name__=='__main__':unittest.main()
