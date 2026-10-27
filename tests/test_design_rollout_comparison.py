"""A variable stopping time must remain visible in repeatability diagnostics."""
import numpy as np
import pytest
from scripts.data.optimize_webster_policy_cpu import compare_rollout_positions


def test_unequal_durations_are_reported_even_when_the_shared_states_match():
    first=np.zeros((3,2,4));second=np.zeros((4,2,4))
    error,per_step,equal=compare_rollout_positions(first,second)
    assert error==0 and np.array_equal(per_step,np.zeros(3))
    assert not equal


def test_equal_durations_compare_all_states_and_reject_incompatible_batches():
    first=np.zeros((3,2,4));second=first.copy();second[-1,-1,-1]=.2
    error,per_step,equal=compare_rollout_positions(first,second)
    assert equal and error==.2
    np.testing.assert_array_equal(per_step,[0,0,.2])
    with pytest.raises(ValueError,match='matching environment'):
        compare_rollout_positions(first,np.zeros((3,3,4)))
