import numpy as np
from supermimic.deploy.sim_foot_support import SimFootSupport


def sample(support, t, left_height, right_height, left_v, right_v):
    support.candidates = lambda *_: {'left': (left_height, np.asarray(left_v, float)),
                                     'right': (right_height, np.asarray(right_v, float))}
    return support.observe(t, None, None, None, None)


def new_support():
    support = SimFootSupport.__new__(SimFootSupport)
    support.side = support.side_since = support.accepted_relative_height = None
    return support


def test_consistent_near_level_switch_preserves_hold():
    s = new_support(); v = [0., 0., 0.]
    assert sample(s, 0., 0., .001, v, v)[1] == 'hold'
    assert sample(s, .02, .001, 0., v, v)[1] == 'hold'
    assert sample(s, .041, 0., .001, v, v)[1] == 'accepted'
    assert sample(s, .06, .001, 0., v, v)[1] == 'accepted'


def test_inconsistent_switch_restarts_hold_then_rejects():
    s = new_support(); zero = [0., 0., 0.]; moving = [.4, 0., 0.]
    assert sample(s, 0., 0., .02, zero, moving)[1] == 'hold'
    assert sample(s, .05, 0., .02, zero, moving)[1] == 'accepted'
    assert sample(s, .06, .001, 0., zero, moving)[1] == 'hold'
    assert sample(s, .11, .001, 0., zero, moving)[1] == 'disagree'
