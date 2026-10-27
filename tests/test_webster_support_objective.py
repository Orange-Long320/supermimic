"""An initial raised leg cannot substitute for a single-leg final push."""
from scripts.data.optimize_webster_policy_cpu import support_metrics


def row(left,right):
    return {"feet":{"left":left,"right":right}}


def test_early_single_support_then_double_leg_jump_is_rejected():
    rows=[row(0.,.1) for _ in range(8)]+[row(0.,0.) for _ in range(8)]+[row(.04,.08) for _ in range(3)]
    single,valid,deficit=support_metrics(rows,16)
    assert len(single)==8
    assert not valid
    assert deficit>0


def test_continuous_left_support_through_final_push_is_accepted():
    rows=[row(0.,.1) for _ in range(12)]+[row(.04,.12) for _ in range(3)]
    _,valid,deficit=support_metrics(rows,12)
    assert valid
    assert deficit==0


def test_late_free_foot_contact_invalidates_single_leg_takeoff():
    rows=[row(0.,.1) for _ in range(12)]+[row(.04,.12) for _ in range(3)]
    rows[10]=row(0.,0.)
    _,valid,deficit=support_metrics(rows,12)
    assert not valid
    assert deficit>0


def test_balanced_single_support_without_flight_is_not_takeoff():
    _,valid,deficit=support_metrics([row(0.,.1) for _ in range(12)],None)
    assert not valid
    assert deficit==0
