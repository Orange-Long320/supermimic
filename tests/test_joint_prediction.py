"""Endpoint-only predictions can miss a stop crossed before reversal."""
import numpy as np
import pytest
from supermimic.policy.joint_prediction import position_acceleration_bounds


@pytest.mark.parametrize('direction',[-1.,1.])
def test_joint_reversal_does_not_cross_the_stop_between_endpoints(direction):
    q=direction*.8;v=direction*8.;h=.1
    endpoint_acc=2*(direction-q-h*v)/h**2
    t=np.linspace(0,h,10001)
    old=q+t*v+.5*endpoint_acc*t*t
    assert abs(old[-1])==pytest.approx(1.)
    assert max(abs(old))>1.06
    low,high=position_acceleration_bounds(q,v,-1.,1.,h)
    acceleration=high if direction>0 else low
    path=q+t*v+.5*acceleration*t*t
    assert max(abs(path))<=1.+1e-12


def test_bounds_cover_random_forward_and_reversing_paths():
    rng=np.random.default_rng(471)
    q=rng.uniform(-.95,.95,100);v=rng.uniform(-4,4,100);h=.08
    lo,hi=position_acceleration_bounds(q,v,-1.,1.,h)
    assert (lo<=hi).all()
    t=np.linspace(0,h,2001)[:,None]
    for acceleration in [lo,hi,(lo+hi)/2]:
        path=q+t*v+.5*acceleration*t*t
        assert np.max(abs(path))<=1.+1e-10
