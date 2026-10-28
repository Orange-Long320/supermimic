from copy import deepcopy
from supermimic.data.design_limits import contact_repair_priority


def report(foot=.008,hard=0.,apex=.837,cost=100.):
    return {'cost':cost,'physical_candidate_passed':False,'takeoff_candidate_passed':False,
            'objective_event':{'ballistic_apex_estimate_m':apex},
            'single_left_support_at_launch':True,'consecutive_ascending_air_frames':3,
            'maxima':{'hard':hard,'foot_penetration':foot,'minimum_nonfoot_height':.04,'speed':.6,'effort':1.}}


def test_contact_deficit_cannot_buy_more_height_or_a_lower_score():
    safe=report(foot=.0049,cost=200.)
    bad=report(foot=.008,apex=.85,cost=1.)
    assert contact_repair_priority(safe,.83)<contact_repair_priority(bad,.83)
    better=report(foot=.0065,cost=150.)
    assert contact_repair_priority(better,.83)<contact_repair_priority(bad,.83)


def test_repair_retains_rising_single_leg_flight_and_hard_limits():
    base=report()
    for change in ['height','support','flight','hard']:
        bad=deepcopy(base);bad['cost']=1.;bad['maxima']['foot_penetration']=.001
        if change=='height':bad['objective_event']['ballistic_apex_estimate_m']=.82
        if change=='support':bad['single_left_support_at_launch']=False
        if change=='flight':bad['consecutive_ascending_air_frames']=0
        if change=='hard':bad['maxima']['hard']=.001
        assert contact_repair_priority(base,.83)<contact_repair_priority(bad,.83)
