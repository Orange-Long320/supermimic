import numpy as np

from supermimic.control.skill_phase import SkillPhase


def test_future_window_is_preserved_when_commands_change():
    nj = 2
    def clip(n, shift=0.):
        x = np.zeros((n, 2*nj+10), dtype=np.float32)
        x[:, 0] = np.linspace(0, shift, n)
        x[:, nj] = shift / ((n-1)*.02)
        x[:, 2*nj+2] = -1.
        x[:, -1] = .65
        return x
    gait = clip(80, .3)
    stop = clip(100, 0.)
    scheduler = SkillPhase({'walk': gait}, {'walk': stop}, {'walk': 10}, nj)
    for step in range(250):
        if step in (2, 8, 12, 70, 140):
            before = scheduler.window().copy()
            scheduler.select('walk' if step in (2, 12, 140) else 'stand')
            np.testing.assert_array_equal(scheduler.window(), before)
        before = scheduler.window().copy()
        scheduler.advance()
        np.testing.assert_array_equal(scheduler.window()[:-1], before[1:])
        assert np.isfinite(scheduler.window()).all()
