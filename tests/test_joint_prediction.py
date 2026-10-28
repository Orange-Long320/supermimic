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


def test_fast_braking_can_replan_before_reaching_the_opposite_stop():
    from supermimic.policy.joint_prediction import replanned_acceleration_bounds
    q=np.array([.23258]);v=np.array([2.597]);lo=np.array([-.241799]);hi=-lo
    minimum,maximum,h=replanned_acceleration_bounds(q,v,lo,hi,np.array([14.456]),.005)
    assert .01<=h<.08
    a=maximum
    t=np.linspace(0,h,1001)[:,None]
    continuous=q+t*v+.5*t*t*a
    assert continuous.min()>=lo[0]-1e-10 and continuous.max()<=hi[0]+1e-10
    for _ in range(round(h/.005)):
        v=v+.005*a;q=q+.005*v
        assert (q>=lo-1e-10).all() and (q<=hi+1e-10).all()
        assert (abs(v)<=14.456).all()


@pytest.mark.parametrize('direction',[-1.,1.])
def test_replanned_stop_keeps_numerical_headroom_for_the_next_step(direction):
    from supermimic.policy.joint_prediction import replanned_acceleration_bounds
    # The native waist stop was exceeded by 4.2e-11 rad with effectively
    # zero velocity. Do not relax the stop; require an interior sampled state.
    bound=.468692;dt=.005;q=np.array([direction*(bound-1e-10)]);v=np.array([direction*1e-8])
    low,high,_=replanned_acceleration_bounds(q,v,-bound,bound,np.array([30.]),dt)
    acceleration=high if direction>0 else low
    following_velocity=v+dt*acceleration
    following=q+dt*following_velocity+direction*1e-10
    assert np.max(abs(following))<bound-5e-7
    next_low,next_high,_=replanned_acceleration_bounds(following,following_velocity,-bound,bound,np.array([30.]),dt)
    assert (next_low<=next_high).all()
