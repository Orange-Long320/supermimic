"""Small contact-wrench LP shared by reference diagnostics and CPU design."""
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, QhullError


def convex_contact_points(points):
    """Keep extreme points without shrinking a rigid body's contact wrench cone.

    For each fixed friction ray, its generalized force is affine in the contact
    position. Interior points therefore add no new wrench; arbitrary index
    subsampling can remove extreme wrenches and incorrectly report infeasibility.
    Degenerate patches are reduced in their affine span, without perturbation.
    """
    points=np.unique(np.asarray(points,dtype=float).reshape(-1,3),axis=0)
    if len(points)<=4:return points
    centred=points-points.mean(axis=0)
    _,singular,axes=np.linalg.svd(centred,full_matrices=False)
    tolerance=singular[0]*max(centred.shape)*np.finfo(float).eps
    rank=int(np.count_nonzero(singular>tolerance))
    if rank==0:return points[:1]
    projected=centred@axes[:rank].T
    if rank==1:return points[np.unique([projected[:,0].argmin(),projected[:,0].argmax()])]
    try:return points[ConvexHull(projected).vertices]
    except QhullError:return points  # Preserve every physical point on failure.


def solve_contact_effort(basis, required, limits):
    """Minimize peak normalized motor effort with exact base-wrench balance."""
    number = basis.shape[1]
    return linprog(np.r_[np.zeros(number), 1.],
                   A_ub=np.r_[np.c_[-basis[6:], -limits], np.c_[basis[6:], -limits]],
                   b_ub=np.r_[-required[6:], required[6:]],
                   A_eq=np.c_[basis[:6], np.zeros(6)], b_eq=required[:6],
                   bounds=[(0, None)] * (number + 1), method="highs")
