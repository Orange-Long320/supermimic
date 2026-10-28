"""Explicit migration of command coefficients when residual groups are appended."""
import numpy as np


def expand_group_parameters(parameters, source_groups, target_groups):
    """Preserve a clock, three knots and optional initial knot; append zero groups.

    Group definitions are required so dimension changes cannot silently reorder
    actuator commands. This transforms commands, never physical state arrays.
    """
    canonical = lambda groups: tuple(tuple((str(n), float(w)) for n, w in group) for group in groups)
    old, new = canonical(source_groups), canonical(target_groups)
    if not old or new[:len(old)] != old:
        raise ValueError('Target groups must append to the exact source group definitions')
    p = np.asarray(parameters)
    if p.ndim < 1 or not np.isfinite(p).all():
        raise ValueError('Expected finite command parameters')
    initial = p.shape[-1] == 1+4*len(old)
    if not initial and p.shape[-1] != 1+3*len(old):
        raise ValueError('Expected a clock, three command knots and optional initial knot')
    q = np.zeros((*p.shape[:-1], 1+(3+int(initial))*len(new)), dtype=p.dtype)
    q[..., 0] = p[..., 0]
    source = p[..., 1:1+3*len(old)].reshape(*p.shape[:-1], 3, len(old))
    target = q[..., 1:1+3*len(new)].reshape(*q.shape[:-1], 3, len(new))
    target[..., :len(old)] = source
    if initial:
        q[..., 1+3*len(new):1+3*len(new)+len(old)] = p[..., 1+3*len(old):]
    return q
