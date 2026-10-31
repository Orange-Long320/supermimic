import multiprocessing
import queue
import time
import unittest

from supermimic.deploy.supervision import MotorPowerConsent, TargetLeaseSupervisor


def producer(messages):
    for i in range(5):
        now=time.monotonic()
        messages.put({'names':['a'],'position':[.2],'velocity':[0.],'effort':[0.],
                      'stiffness':[20.],'damping':[2.],'generated':now,'state_sampled':now,'sequence':i})
        time.sleep(.02)


class SupervisionTests(unittest.TestCase):
    def test_deployment_approval_does_not_power_motors(self):
        sha='a'*64
        for consent in (MotorPowerConsent(False,sha,10.),MotorPowerConsent(True,'b'*64,10.),
                        MotorPowerConsent(True,sha,-500.),MotorPowerConsent(True,sha,12.)):
            with self.assertRaises(ValueError):consent.require(sha,11.)
        MotorPowerConsent(True,sha,10.).require(sha,11.)

    def test_source_age_not_hidden_by_a_new_target(self):
        s=TargetLeaseSupervisor(['a'],[3.],max_state_age=.05,max_target_age=.1)
        self.assertIsNone(s.tick(1.))
        packet={'names':['a'],'position':[1.],'velocity':[0.],'effort':[0.],
                'stiffness':[20.],'damping':[2.],'generated':1.,'state_sampled':.8,'sequence':0}
        self.assertTrue(s.accept(packet));out=s.tick(1.01)
        self.assertTrue(out['fallback']);self.assertEqual(out['stiffness'],[0.]);self.assertEqual(out['damping'],(3.,))
        packet.update(sequence=1,generated=1.02,state_sampled=1.02)
        self.assertFalse(s.accept(packet));self.assertTrue(s.tick(1.03)['fallback'])

    def test_independent_policy_process_exit_expires_lease(self):
        ctx=multiprocessing.get_context('spawn');messages=ctx.Queue();child=ctx.Process(target=producer,args=(messages,))
        supervisor=TargetLeaseSupervisor(['a'],[3.],max_state_age=.2,max_target_age=.2)
        child.start();active=False;fallback=None;deadline=time.monotonic()+5
        try:
            while time.monotonic()<deadline:
                try:supervisor.accept(messages.get(timeout=.01))
                except queue.Empty:pass
                output=supervisor.tick(time.monotonic())
                if output is not None:
                    if output['fallback']:fallback=output;break
                    active=True
            self.assertTrue(active);self.assertIsNotNone(fallback)
            self.assertEqual(fallback['stiffness'],[0.]);self.assertEqual(fallback['damping'],(3.,))
        finally:
            child.join(timeout=2)
            if child.is_alive():child.terminate();child.join()
            messages.close()


if __name__=='__main__':unittest.main()
