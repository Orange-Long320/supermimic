"""Small contact-wrench LP shared by reference diagnostics and CPU design."""
import numpy as np
from scipy.optimize import linprog


def solve_contact_effort(basis, required, limits):
    """Minimize peak normalized motor effort with exact base-wrench balance."""
    number = basis.shape[1]
    return linprog(np.r_[np.zeros(number), 1.],
                   A_ub=np.r_[np.c_[-basis[6:], -limits], np.c_[basis[6:], -limits]],
                   b_ub=np.r_[-required[6:], required[6:]],
                   A_eq=np.c_[basis[:6], np.zeros(6)], b_eq=required[:6],
                   bounds=[(0, None)] * (number + 1), method="highs")
