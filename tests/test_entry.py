import unittest
import numpy as np
from supermimic.deploy.entry import EntryController, EntryPhase, EntryThresholds, QuinticPose


class EntryTests(unittest.TestCase):
    def test_path_limits_and_zero_end_derivatives(self):
        p = QuinticPose([0., 1.], [2., -1.], [.3, .5], [.4, .4], minimum_duration=1.)
        rows = [p.sample(t) for t in np.linspace(0, p.duration, 1001)]
        self.assertTrue(np.all(np.max(np.abs(np.asarray([r[1] for r in rows])), axis=0) <= [.3000001, .5000001]))
        self.assertTrue(np.all(np.max(np.abs(np.asarray([r[2] for r in rows])), axis=0) <= .4000001))
        for t, target in ((0., [0., 1.]), (p.duration, [2., -1.])):
            q, v, a = p.sample(t); np.testing.assert_allclose(q, target)
            np.testing.assert_allclose(v, 0., atol=1e-12); np.testing.assert_allclose(a, 0., atol=1e-12)

    def controller(self):
        path = QuinticPose([0.], [0.], [1.], [1.], minimum_duration=.1)
        thresholds = EntryThresholds(.1, .1, .1, .1, .1, 2., .031, 3, .1)
        return EntryController(path, thresholds, [20.], [2.], [40.], [4.])

    def test_noisy_settle_resets_history_before_handover(self):
        c = self.controller()
        def step(t, v=0., policy=None): return c.step(t, [0.], [v], [0.]*3, [0., 0., -1.], policy_target=policy)
        for t in np.arange(0., .28, .02): step(float(t))
        self.assertEqual(c.phase, EntryPhase.HISTORY)
        step(.28, .2); self.assertEqual(c.phase, EntryPhase.SETTLE)
        t=.30
        while c.phase != EntryPhase.HANDOVER: step(t); t+=.02
        # The transition produces the same pose and gains; no new raw policy jump.
        np.testing.assert_allclose(c.last_command[0], [0.]); np.testing.assert_allclose(c.last_command[2], [20.])
        while c.phase != EntryPhase.ACTIVE: step(t, policy=[1.]); t+=.02
        np.testing.assert_allclose(c.last_command[0], [1.]); np.testing.assert_allclose(c.last_command[2], [40.])

    def test_gap_fault_is_latched(self):
        c=self.controller(); c.step(0., [0.], [0.], [0.]*3, [0., 0., -1.])
        with self.assertRaisesRegex(ValueError, 'gap'): c.step(.1, [0.], [0.], [0.]*3, [0., 0., -1.])
        with self.assertRaisesRegex(ValueError, 'latched'): c.step(.12, [0.], [0.], [0.]*3, [0., 0., -1.])

    def test_never_settled_times_out(self):
        c=self.controller()
        with self.assertRaisesRegex(ValueError, 'settle'):
            for t in np.arange(0., 2.5, .02): c.step(float(t), [1.], [0.], [0.]*3, [0., 0., -1.])


if __name__ == '__main__': unittest.main()
