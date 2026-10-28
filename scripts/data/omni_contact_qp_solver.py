"""Solve a saved sparse QP in the existing isolated Omni environment.

No robot/model imports. Exchanges only arrays, keeping the physics environment
and Omni's solver dependencies separate. Solver success is not physical success.
"""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
    import numpy as np
    import scipy.sparse as sp
    import osqp
    z=np.load(a.directory/'vectors.npz');P=sp.load_npz(a.directory/'P.npz');A=sp.load_npz(a.directory/'A.npz')
    solver=osqp.OSQP();solver.setup(P=P,q=z['q'],A=A,l=z['lower'],u=z['upper'],
        eps_abs=1e-7,eps_rel=1e-7,max_iter=20000,polishing=True,verbose=False)
    result=solver.solve(raise_error=False)
    if result.x is not None:
        value=A@result.x;violation=float(max(np.max(np.maximum(z['lower']-value,0)),np.max(np.maximum(value-z['upper'],0))))
        np.savez_compressed(a.directory/'solution.npz',x=result.x)
    else:violation=None
    (a.directory/'solver.json').write_text(json.dumps(dict(osqp_version=osqp.__version__,status=result.info.status,
        status_value=int(result.info.status_val),iterations=int(result.info.iter),primal_residual=float(result.info.prim_res),
        dual_residual=float(result.info.dual_res),maximum_linear_constraint_violation=violation),indent=2))


if __name__=='__main__':main()
