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


def test_configured_loop_restarts_before_slow_clip_tail():
    nj = 2
    gait = np.zeros((80, 2*nj+10), dtype=np.float32)
    gait[:, 0] = np.arange(80)/100.
    gait[:, 2*nj+2] = -1.
    stop = gait.copy()
    scheduler = SkillPhase({'walk':gait}, {'walk':stop}, {'walk':10}, nj, loops={'walk':(25, 55)})
    scheduler.select('walk')
    for _ in range(130):
        scheduler.advance()
    seams = [e for e in scheduler.events if e.get('kind') == 'loop_seam']
    assert len(seams) >= 2
    assert all(e['from_frame'] == 55 and e['to_frame'] == 25 for e in seams)


def test_jog_brake_fade_preserves_committed_window_and_other_features():
    nj = 2
    clip = np.zeros((100, 2*nj+10), dtype=np.float32)
    clip[:, 2*nj+2] = -1.
    clip[:, 2*nj+6] = 1.
    stops = clip.copy(); stops[-1, 2*nj+6] = 0.
    args = ({'walk': clip, 'jog': clip}, {'walk': stops, 'jog': stops}, {'walk': 0, 'jog': 0}, nj)
    baseline = SkillPhase(*args)
    candidate = SkillPhase(*args, jog_brake_fade=.4)
    for scheduler in (baseline, candidate):
        scheduler.select('jog')
        for _ in range(50): scheduler.advance()
        before = scheduler.window().copy()
        scheduler.select('stand')
        np.testing.assert_array_equal(scheduler.window(), before)
    b, c = np.stack(baseline.plan), np.stack(candidate.plan)
    np.testing.assert_allclose(b[:, :2*nj+6], c[:, :2*nj+6])
    np.testing.assert_allclose(b[:, 2*nj+9:], c[:, 2*nj+9:])
    assert 0 < c[0, 2*nj+6] < b[0, 2*nj+6]
    np.testing.assert_allclose(c[19:, 2*nj+6:2*nj+9], 0, atol=1e-7)
