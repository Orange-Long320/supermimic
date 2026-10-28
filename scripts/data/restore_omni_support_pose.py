"""Bounded nonlinear feasibility restoration for a failed Omni support pose.

Diagnostic seeds only. Mechanical joint bounds remain hard. Collision and
contact-task errors are optimized internally, then checked independently at
their original tolerances; solver termination alone never approves a pose.
"""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--failed',type=Path,required=True)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--support',type=Path,required=True)
    p.add_argument('--frame',type=int,required=True)
    p.add_argument('--sticking-side',choices=['left','right'],default='right',help='Explicit pivot side from the failed frame diagnostic')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('Output exists')
    import mujoco
    import numpy as np
    from scipy.optimize import least_squares
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra',check_assets=True)
    from supermimic.robots.dimensions import PROFILE
    m=mujoco.MjModel.from_xml_path(str(a.scene));d=mujoco.MjData(m)
    with np.load(a.failed,allow_pickle=False) as z:base=z['qpos'].copy()
    if base.shape!=(45,) or (m.nq,m.nv)!=(45,43):raise ValueError('Expected X2 plus locked terrain pose')
    support=json.loads(a.support.read_text())['windows']
    targets={}
    for side,windows in support.items():
        active=[z for start,end,z in windows if start<=a.frame<=end]
        if active:targets[side]=active[0]
    if not targets:raise ValueError('No source-confirmed flat support at this frame')
    corners={s:[m.body(f'{s}_omni_corner_{k}').id for k in range(4)] for s in ('left','right')}
    obstacles=[g for g in range(m.ngeom) if m.geom(g).name=='ground' or m.geom(g).name.startswith('multi_boxes_')]
    robot=[g for g in range(m.ngeom) if g not in obstacles and (m.geom_contype[g] or m.geom_conaffinity[g])]
    lo,hi=m.jnt_range[1:32].T
    if not m.jnt_limited[1:32].all():raise ValueError('Explicit mechanical bounds required')
    lower=np.r_[np.full(3,-.5),np.full(3,-.6),lo-base[7:38]]
    upper=np.r_[np.full(3,.5),np.full(3,.6),hi-base[7:38]]
    speed=np.array([PROFILE['velocity_limit'][m.joint(j).name] for j in range(1,32)])
    d.qpos[:]=base;mujoco.mj_forward(m,d)
    # The known frame-60 mask is L=False,R=True. The explicitly selected side
    # isolates whether locking every sole corner is the obstruction.
    sticking_xy=d.xpos[corners[a.sticking_side],:2].copy()
    toe_xy=d.xpos[m.body(a.sticking_side+'_omni_toe').id,:2].copy()
    witness=np.zeros(6);full_velocity=np.zeros(m.nv)
    def evaluate(delta,mode,forward=False):
        q=base.copy();full_velocity[:]=0.;full_velocity[:37]=delta;mujoco.mj_integratePos(m,q,full_velocity,1.)
        d.qpos[:]=q
        if forward:mujoco.mj_forward(m,d)
        else:mujoco.mj_kinematics(m,d)
        distances=np.array([mujoco.mj_geomDistance(m,d,g,o,.02,witness) for g in robot for o in obstacles])
        flat=np.concatenate([d.xpos[corners[side],2]-height for side,height in targets.items()])
        if mode=='all_corners':xy=(d.xpos[corners[a.sticking_side],:2]-sticking_xy).ravel()
        elif mode=='toe_only':xy=d.xpos[m.body(a.sticking_side+'_omni_toe').id,:2]-toe_xy
        else:xy=np.empty(0)
        residual=np.r_[np.maximum(-distances-.0008,0.)*1000,
            np.maximum(abs(flat)-.001,0.)*1000,np.maximum(abs(xy)-.001,0.)*1000,delta*.01]
        metrics={'max_penetration_m':float(max(0.,-distances.min())),'maximum_flat_height_error_m':float(abs(flat).max()),
            'maximum_sticking_xy_error_m':float(abs(xy).max()) if len(xy) else 0.,
            'joint_limit_violation_rad':float(np.maximum(np.maximum(lo-q[7:38],q[7:38]-hi),0).max())}
        return residual,metrics,q
    a.out.mkdir(parents=True);(a.out/'as-run.py').write_bytes(Path(__file__).read_bytes());results=[]
    initial=np.clip(np.zeros(37),lower+1e-10,upper-1e-10)
    for mode in ('all_corners','toe_only','no_xy_diagnostic'):
        def derivative(x):
            columns=[]
            for j in range(37):
                plus=x.copy();minus=x.copy();plus[j]=min(upper[j],x[j]+1e-6);minus[j]=max(lower[j],x[j]-1e-6)
                columns.append((evaluate(plus,mode)[0]-evaluate(minus,mode)[0])/(plus[j]-minus[j]))
            return np.stack(columns,axis=1)
        fit=least_squares(lambda x:evaluate(x,mode)[0],initial,bounds=(lower,upper),
            jac=derivative,max_nfev=120,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        _,metrics,q=evaluate(fit.x,mode,forward=True)
        feasible=metrics['max_penetration_m']<=.00085 and metrics['maximum_flat_height_error_m']<=.00105 and metrics['maximum_sticking_xy_error_m']<=.00105 and metrics['joint_limit_violation_rad']<=1e-8
        result={'mode':mode,'solver_success':bool(fit.success),'solver_status':int(fit.status),'nfev':fit.nfev,
            'cost':float(fit.cost),'geometry_and_task_feasible':bool(feasible),'metrics':metrics,
            'max_joint_change_rad':float(abs(q[7:38]-base[7:38]).max()),
            'minimum_joint_speed_transition_seconds':float((abs(q[7:38]-base[7:38])/speed).max()),
            'training_approved':False}
        np.savez_compressed(a.out/(mode+'.npz'),qpos=q,source_qpos=base,delta=fit.x)
        results.append(result);(a.out/'progress.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(result),flush=True)
    (a.out/'report.json').write_text(json.dumps({'frame':a.frame,'sticking_side':a.sticking_side,'source_support_targets':targets,'results':results,
        'scope':'Nonlinear local geometry/task restoration only; no force, temporal continuity or controller certificate. Diagnostic modes are not automatically applied to references.'},indent=2)+'\n')


if __name__=='__main__':main()
