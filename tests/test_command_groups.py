import numpy as np
import pytest
from supermimic.data.command_groups import expand_group_parameters


@pytest.mark.parametrize('initial', [False, True])
def test_appended_groups_preserve_every_original_knot_and_clock(initial):
    old = [[('left_roll', 1), ('right_roll', 1)], [('knee', 1)]]
    new = old+[[('right_roll', 1)], [('yaw', 1)]]
    p = np.random.default_rng(2).normal(size=(5, 1+(3+int(initial))*len(old)))
    p[:, 0] = .52
    q = expand_group_parameters(p, old, new)
    np.testing.assert_array_equal(q[:, 0], p[:, 0])
    for knot in range(3+int(initial)):
        a = p[:, 1+knot*len(old):1+(knot+1)*len(old)]
        b = q[:, 1+knot*len(new):1+(knot+1)*len(new)]
        np.testing.assert_array_equal(a, b[:, :len(old)])
        assert not b[:, len(old):].any()
        # The appended right-roll channel spans independent leg roll without changing the old command.
        np.testing.assert_array_equal(a@np.array([[1, 1, 0, 0], [0, 0, 1, 0]]),
                                      b@np.array([[1, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]))


def test_migration_rejects_group_reordering_and_wrong_dimensions():
    old = [[('a', 1)], [('b', 1)]]
    with pytest.raises(ValueError): expand_group_parameters(np.zeros(9), old, old[::-1])
    with pytest.raises(ValueError): expand_group_parameters(np.zeros(10), old, old)
