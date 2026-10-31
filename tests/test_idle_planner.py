import numpy as np
import pytest

from supermimic.control.idle_planner import ContextIdlePlanner


def planner(limits=None, **kwargs):
    n = 2
    gait = np.zeros((100, 2*n+10), dtype=np.float32)
    gait[:, 0] = np.linspace(0, .6, 100)
    gait[:, n] = .6/(99*.02)
    gait[:, 2*n+2] = -1
    gait[:, -1] = .65
    stop = gait.copy()
    stop[:, 0] = np.maximum(.6-np.arange(100)*.01, 0)
    stop[:, n] = np.where(stop[:, 0] > 0, -.5, 0)
    return ContextIdlePlanner({'walk': gait}, {'walk': stop}, {'walk': 10}, n,
        loops={'walk': (25, 70)}, joint_limits=np.array([[-2, 2], [-2, 2]]) if limits is None else limits,
        velocity_limits=np.array([30, 30]),
        foot_features=lambda row: np.array([[row[0], .1, 0], [0, -.1, 0]]), **kwargs)


def test_interrupts_preserve_committed_window_and_reach_idle():
    p = planner()
    for step in range(350):
        p.set_context(p.window()[0, :2], p.window()[0, 2:4])
        if step in (0, 3, 7, 25, 200, 220):
            before = p.window().copy()
            p.select('walk' if step in (0, 7, 200) else 'idle')
            np.testing.assert_array_equal(p.window(), before)
        before = p.window().copy()
        p.advance()
        np.testing.assert_array_equal(p.window()[:-1], before[1:])
    np.testing.assert_allclose(p.window(), np.repeat(p.stand[None], p.horizon, axis=0))
    assert all(e['selected']['valid'] for e in p.events if e['kind'] == 'context_transition')


def test_measured_tracking_error_changes_selected_entry():
    a, b = planner(), planner()
    for p, q in ((a, [0, 0]), (b, [1, 0])):
        p.set_context(q, [0, 0])
        p.select('walk')
    assert a.events[-1]['selected']['entry_frame'] != b.events[-1]['selected']['entry_frame']


def test_no_admissible_transition_fails_without_rewriting_published_goals():
    narrow = np.array([[-.0001, .0001], [-.0001, .0001]])
    with pytest.raises(ValueError, match='No mechanically admissible sustained cycle'):
        planner(narrow)
    p = planner()
    p.joint_limits = narrow
    p.set_context([0, 0], [0, 0])
    before = p.window().copy()
    with pytest.raises(ValueError, match='No mechanically admissible'):
        p.select('walk')
    np.testing.assert_array_equal(p.window(), before)
    assert p.requested == 'stand'


def test_context_and_scope_validation():
    p = planner()
    with pytest.raises(ValueError, match='Measured joint context'):
        p.select('walk')
    with pytest.raises(ValueError, match='Invalid measured'):
        p.set_context([np.nan, 0], [0, 0])
    with pytest.raises(ValueError, match='Walk and Idle only'):
        p.select('jog')


def test_repeated_cycle_seams_are_mechanically_screened():
    p = planner()
    p.set_context([0, 0], [0, 0])
    p.select('walk')
    for _ in range(300):
        p.advance()
    seams = [e for e in p.events if e['kind'] == 'loop_seam']
    assert len(seams) >= 3
    assert all(e['valid'] and e['joint_speed_ratio'] <= 1 and e['hard_excess_rad'] <= 1e-4 for e in seams)


def test_both_original_paths_exactly_reproduce_original_scheduler():
    from supermimic.control.skill_phase import SkillPhase
    p = planner(baseline_motion=True, baseline_idle=True)
    original = SkillPhase(p.gaits, p.stops, p.brake_start, p.nj, loops=dict(p.loops))
    for step in range(500):
        if step in (0, 180, 300, 350):
            mode = 'walk' if step in (0, 300) else 'stand'
            p.select(mode)
            original.select(mode)
        np.testing.assert_array_equal(p.window(), original.window())
        p.advance()
        original.advance()
