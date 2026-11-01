import unittest
import numpy as np

from supermimic.deploy.entry import EntryController, EntryPhase, EntryThresholds, QuinticPose
from supermimic.deploy.independent_policy import IndependentPolicyCoordinator


class Policy:
    def __init__(self): self.observations=[]
    def infer(self,obs,goal): self.observations.append(np.asarray(obs).copy()); return np.array([1.])


class Tracker:
    def goal(self,position,quat,q,skill): return np.array([0.])
    def advance(self): pass


class IndependentPolicyTests(unittest.TestCase):
    def make(self):
        path=QuinticPose([0.],[0.],[1.],[1.],minimum_duration=.1)
        entry=EntryController(path,EntryThresholds(.1,.1,.1,.1,.1,2.,.031,3,.1),[20.],[2.],[40.],[4.])
        policy=Policy()
        return IndependentPolicyCoordinator(entry,[0.],[.5],policy,lambda p,q: Tracker()),policy

    def test_waits_for_user_handover_then_blends(self):
        c,policy=self.make()
        with self.assertRaisesRegex(ValueError,'not ready'): c.request_handover()
        t=0.
        for _ in range(40):
            command=c.step(t,[0,0,.6],[1,0,0,0],[0],[0],[0,0,0],[0,0,-1],support_ready=True); t+=.02
        self.assertEqual(c.entry.phase,EntryPhase.HISTORY); self.assertEqual(len(policy.observations),0)
        c.request_handover()
        command=c.step(t,[0,0,.6],[1,0,0,0],[0],[0],[0,0,0],[0,0,-1],support_ready=True);t+=.02
        self.assertEqual(c.entry.phase,EntryPhase.HANDOVER); np.testing.assert_allclose(command[0],[0.])
        for _ in range(6):
            command=c.step(t,[0,0,.6],[1,0,0,0],[0],[0],[0,0,0],[0,0,-1],support_ready=True); t+=.02
        self.assertEqual(c.entry.phase,EntryPhase.ACTIVE); np.testing.assert_allclose(command[0],[.5])
        self.assertEqual(policy.observations[0].shape,(3*(1+1+3+3+1),))

    def test_loss_of_quiet_resets_request_and_history(self):
        c,_=self.make();t=0.
        for _ in range(30): c.step(t,[0,0,.6],[1,0,0,0],[0],[0],[0,0,0],[0,0,-1],support_ready=True);t+=.02
        c.request_handover()
        c.step(t,[0,0,.6],[1,0,0,0],[0],[.2],[0,0,0],[0,0,-1],support_ready=True)
        self.assertEqual(c.entry.phase,EntryPhase.SETTLE);self.assertFalse(c.requested);self.assertEqual(len(c.history),0)

    def test_failed_inference_latches_fault(self):
        c,policy=self.make();t=0.
        for _ in range(30): c.step(t,[0,0,.6],[1,0,0,0],[0],[0],[0,0,0],[0,0,-1],support_ready=True);t+=.02
        c.request_handover()
        c.step(t,[0,0,.6],[1,0,0,0],[0],[0],[0,0,0],[0,0,-1],support_ready=True);t+=.02
        def bad(*args): raise RuntimeError('synthetic failure')
        policy.infer=bad
        with self.assertRaisesRegex(ValueError,'Policy or goal failed'):
            c.step(t,[0,0,.6],[1,0,0,0],[0],[0],[0,0,0],[0,0,-1],support_ready=True)
        self.assertEqual(c.entry.phase,EntryPhase.FAULT)


if __name__=='__main__': unittest.main()
