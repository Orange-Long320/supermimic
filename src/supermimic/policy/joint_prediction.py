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
