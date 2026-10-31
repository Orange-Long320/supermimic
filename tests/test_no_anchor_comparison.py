import importlib.util
from pathlib import Path

import numpy as np

spec=importlib.util.spec_from_file_location('comparison',Path(__file__).resolve().parents[1]/'scripts/evaluation/compare_no_anchor_trials.py')
comparison=importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def trajectory(duration, velocity):
    t=np.arange(round(duration/.02)+1)*.02
    pos=np.c_[velocity*t,np.zeros(len(t)),np.ones(len(t))*.65]
    return dict(t=t,pos=pos,yaw=np.zeros(len(t)),rows=[],
                segments=[('stand',0.,float(t[-1]),0,len(t))],result={'schedule':[[0.,'stand']]})


def test_known_constant_translation_and_body_frame():
    d=trajectory(5.,.2)
    a=comparison.metrics(d,0.,5.)
    assert np.isclose(a['max_excursion_m'],1.)
    assert np.isclose(a['sampled_path_length_m'],1.)
    assert np.isclose(a['median_body_vx_mps'],.2)
    assert a['root_stop_time_s'] is None
    d['yaw'][:]=np.pi/2
    a=comparison.metrics(d,0.,5.)
    assert abs(a['median_body_vx_mps'])<1e-9
    assert np.isclose(a['median_body_vy_mps'],-.2)


def test_long_standing_uses_common_duration_after_settling():
    report=comparison.compare(trajectory(35.,.001),trajectory(20.,.001))
    stand=report['long_stand']
    assert stand['baseline']['observed_s']==stand['candidate']['observed_s']==15.
    for label in ('baseline','candidate'):
        assert np.isclose(stand[label]['net_displacement_m'],.015)
