import pytest

from supermimic.data.flight_budget import rising_ballistic_apex


def test_launch_speed_does_not_replace_launch_height():
    old = rising_ballistic_apex(.7915133, .547715, 9.81)
    delayed = rising_ballistic_apex(.7465498, .806931, 9.81)
    assert old == pytest.approx(.8068034)
    assert delayed == pytest.approx(.7797373)
    assert old > delayed


def test_descending_state_has_no_remaining_rise():
    assert rising_ballistic_apex(.8, -2, 9.81) == .8
    with pytest.raises(ValueError):
        rising_ballistic_apex(.8, 1, 0)
