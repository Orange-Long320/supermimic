"""Solve a saved sparse QP in the existing isolated Omni environment.

No robot/model imports. Exchanges only arrays, keeping the physics environment
and Omni's solver dependencies separate. Solver success is not physical success.
"""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--backend',choices=['osqp','clarabel'],default='osqp');a=p.parse_args()
    import numpy as np
    import scipy.sparse as sp
    z=np.load(a.directory/'vectors.npz');P=sp.load_npz(a.directory/'P.npz');A=sp.load_npz(a.directory/'A.npz')
    if a.backend=='osqp':
        import osqp
        solver=osqp.OSQP();solver.setup(P=P,q=z['q'],A=A,l=z['lower'],u=z['upper'],
            eps_abs=1e-7,eps_rel=1e-7,max_iter=20000,polishing=True,verbose=False)
        result=solver.solve(raise_error=False);solution=result.x
        report=dict(backend='osqp',version=osqp.__version__,status=result.info.status,
            status_value=int(result.info.status_val),iterations=int(result.info.iter),
            primal_residual=float(result.info.prim_res),dual_residual=float(result.info.dual_res))
    else:
        import cvxpy as cp
        import clarabel
        full=P+P.T-sp.diags(P.diagonal(),format='csc');x=cp.Variable(A.shape[1])
        eq=np.isfinite(z['lower'])&np.isfinite(z['upper'])&(z['lower']==z['upper'])
        lower=np.isfinite(z['lower'])&~eq;upper=np.isfinite(z['upper'])&~eq
        problem=cp.Problem(cp.Minimize(.5*cp.quad_form(x,cp.psd_wrap(full))+z['q']@x),
            [A[eq]@x==z['lower'][eq],A[lower]@x>=z['lower'][lower],A[upper]@x<=z['upper'][upper]])
        problem.solve(solver='CLARABEL',tol_gap_abs=1e-8,tol_gap_rel=1e-8,tol_feas=1e-8,max_iter=100,verbose=False)
        solution=x.value
        report=dict(backend='clarabel',version=clarabel.__version__,cvxpy=cp.__version__,status=problem.status,
            status_value=1 if problem.status=='optimal' else 2 if problem.status=='optimal_inaccurate' else 3,
            iterations=problem.solver_stats.num_iters,solve_seconds=problem.solver_stats.solve_time)
    if solution is not None:
        value=A@solution;violation=float(max(np.max(np.maximum(z['lower']-value,0)),np.max(np.maximum(value-z['upper'],0))))
        np.savez_compressed(a.directory/'solution.npz',x=solution)
    else:violation=None
    report['maximum_linear_constraint_violation']=violation
    (a.directory/'solver.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
