"""Continuous joint-position bounds under the local constant-acceleration model."""
import numpy as np


def position_acceleration_bounds(position,velocity,lower,upper,horizon):
    """Bound the entire interval, including a reversal before its endpoint.

    This is a local prediction constraint, not a guarantee against changes in
    acceleration or contact geometry after issuing the motor command.
    """
    q,v,lo,hi=np.broadcast_arrays(position,velocity,lower,upper)
    if horizon<=0 or np.any(q<=lo) or np.any(q>=hi):
        raise ValueError('Continuous prediction requires an interior state and a positive horizon')
    distance_lower=q-lo;distance_upper=hi-q
    minimum=2*(-distance_lower-horizon*v)/horizon**2
    maximum=2*(distance_upper-horizon*v)/horizon**2
    before_lower=(v<0)&(2*distance_lower<-horizon*v)
    before_upper=(v>0)&(2*distance_upper<horizon*v)
    minimum=np.where(before_lower,v*v/(2*distance_lower),minimum)
    maximum=np.where(before_upper,-v*v/(2*distance_upper),maximum)
    return minimum,maximum


def replanned_acceleration_bounds(position,velocity,lower,upper,velocity_limit,timestep,horizon=.08):
    """Keep constant-acceleration predictions local during rapid braking.

    A long constant-acceleration interval may require both braking at the near
    stop and avoiding the opposite stop after reversing. Replanning can change
    acceleration, so reduce that interval to at least two actual physics steps.
    Preserve continuous bounds and add semi-implicit integration sample bounds.
    This does not alter physical limits or guarantee changing-contact dynamics.
    """
    if timestep<=0 or horizon<2*timestep:raise ValueError('Prediction needs at least two positive physics steps')
    q=np.asarray(position);v=np.asarray(velocity);limit=np.asarray(velocity_limit)
    while True:
        minimum,maximum=position_acceleration_bounds(q,v,lower,upper,horizon)
        times=np.arange(1,int(np.ceil(horizon/timestep))+1)*timestep
        # Include the endpoint if it falls between discrete simulation samples.
        times=times[times<=horizon+1e-12]
        denominator=.5*times[:,None]*(times[:,None]+timestep)
        minimum=np.maximum(minimum,((np.asarray(lower)-q-times[:,None]*v)/denominator).max(axis=0))
        maximum=np.minimum(maximum,((np.asarray(upper)-q-times[:,None]*v)/denominator).min(axis=0))
        minimum=np.maximum(minimum,(-limit-v)/horizon)
        maximum=np.minimum(maximum,(limit-v)/horizon)
        if np.all(minimum<=maximum):return minimum,maximum,horizon
        if horizon<=2*timestep+1e-12:raise RuntimeError('Joint predictive bounds infeasible even at two physics steps')
        horizon=max(2*timestep,horizon/2)
