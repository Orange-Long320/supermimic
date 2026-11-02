import unittest
from supermimic.deploy.actuator_watchdog import ActuatorDeadlineWatchdog


class ActuatorWatchdogTests(unittest.TestCase):
    def make(self):
        w=ActuatorDeadlineWatchdog(command_timeout=.05,max_state_age=.05,max_round_trip=.025)
        w.begin('a'*32,1.,inhibit_observed=True)
        return w

    def accept(self,w,t=1.001,seq=0,state=0,age=.002,challenge=None):
        c=challenge or w.challenge(t-.0005)
        return w.accept(session=c.session,nonce=c.nonce,command_sequence=seq,
                        state_sequence=state,state_age=age,now=t)

    def test_no_invented_startup_or_automatic_rearm(self):
        w=self.make();self.assertTrue(w.tick(1.))
        self.assertTrue(self.accept(w));self.assertFalse(w.tick(1.01))
        self.assertTrue(w.tick(1.06));self.assertIsNone(w.challenge(1.061))
        with self.assertRaises(ValueError):w.begin('b'*32,1.07,inhibit_observed=True)

    def test_deadline_uses_receiver_clock_and_late_frame_cannot_revive(self):
        w=self.make();c=w.challenge(1.)
        self.assertFalse(self.accept(w,t=1.051,challenge=c))
        self.assertIn('deadline',w.reason)

    def test_challenges_cannot_keep_a_dead_sender_alive(self):
        w=self.make();c=w.challenge(1.)
        self.assertEqual(c,w.challenge(1.02))
        self.assertIsNone(w.challenge(1.026));self.assertTrue(w.tick(1.027))

    def test_replayed_state_and_old_nonce_latch(self):
        for stale_nonce in (False,True):
            w=self.make();old=w.challenge(1.);self.assertTrue(self.accept(w,challenge=old))
            fresh=w.challenge(1.01)
            ok=w.accept(session='a'*32,nonce=(old if stale_nonce else fresh).nonce,
                command_sequence=1,state_sequence=0,state_age=.001,now=1.012)
            self.assertFalse(ok);self.assertTrue(w.inhibit_required)

    def test_source_age_includes_network_delay(self):
        w=self.make();c=w.challenge(1.)
        self.assertFalse(self.accept(w,t=1.02,age=.035,challenge=c))
        self.assertIn('Source or transport',w.reason)

    def test_backward_clock_nonfinite_age_and_boolean_sequences_rejected(self):
        w=self.make();self.assertTrue(self.accept(w));self.assertTrue(w.tick(.99))
        for age,seq in ((float('nan'),1),(-.1,1),(0.,True)):
            w=self.make();self.assertFalse(self.accept(w,age=age,seq=seq))

    def test_fresh_monotonic_frames_renew_without_remote_timestamp(self):
        w=self.make()
        for i in range(100):
            c=w.challenge(1.+i*.02)
            self.assertTrue(w.accept(session=c.session,nonce=c.nonce,command_sequence=i,
                state_sequence=i,state_age=.002,now=1.001+i*.02))
            self.assertFalse(w.tick(1.002+i*.02))


if __name__=='__main__':unittest.main()
