"""Fit an actual native discrete fixed point; no inverse-force certificate assumed."""
import argparse
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['model','initial','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('New output required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Explicit CPU-only execution required')
    import mujoco
    import numpy as np
    from scipy.optimize import least_squares
    m=mujoco.MjModel.from_binary_path(str(a.model));q0=np.load(a.initial)['qpos'].copy()
    initial=mujoco.MjData(m);initial.qpos[:]=q0;mujoco.mj_inverse(m,initial)
    joints=m.actuator_trnid[:,0].astype(int);qc=m.jnt_qposadr[joints];vc=m.jnt_dofadr[joints]
    u0=q0[qc]+initial.qfrc_inverse[vc]/m.actuator_gainprm[:,0]
    ankles=[m.joint('robot/'+s+'_ankle_'+axis+'_joint').id for s in ['left','right'] for axis in ['pitch','roll']]
    columns=m.jnt_qposadr[ankles];vcols=np.r_[2,3,4,m.jnt_dofadr[ankles]].astype(int)
    scales=np.r_[.001,.02,.02,np.full(4,.03),np.full(m.nu,.05)]
    a.out.mkdir(parents=True);(a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    calls=0
    def state(x):
        q=q0.copy();velocity=np.zeros(m.nv);velocity[vcols]=x[:7]*scales[:7];mujoco.mj_integratePos(m,q,velocity,1.)
        return q,u0+x[7:]*scales[7:]
    def residual(x,regularize=True):
        nonlocal calls
        q,u=state(x);d=mujoco.MjData(m);d.qpos[:]=q;d.ctrl[:]=u;mujoco.mj_forward(m,d);mujoco.mj_step(m,d);calls+=1
        r=d.qvel/.001
        return np.r_[r,x*1e-6] if regularize else r
    def jac(x):
        cols=[]
        for i in range(len(x)):
            delta=np.zeros(len(x));delta[i]=1e-3
            cols.append((residual(x+delta)-residual(x-delta))/.002)
        return np.stack(cols,axis=1)
    x=np.zeros(len(scales));history=[]
    fit=least_squares(residual,x,jac=jac,bounds=(-np.ones(len(x)),np.ones(len(x))),max_nfev=100,ftol=1e-12,xtol=1e-12,gtol=1e-12)
    q,u=state(fit.x);np.savez_compressed(a.out/'equilibrium.npz',qpos=q,control=u)
    d=mujoco.MjData(m);d.qpos[:]=q;d.ctrl[:]=u;mujoco.mj_forward(m,d);rows=[];states=[];controls=[];root=int(m.jnt_bodyid[0])
    for step in range(round(5/m.opt.timestep)):
        d.ctrl[:]=u;controls.append(u.copy());mujoco.mj_step(m,d);mujoco.mj_forward(m,d);states.append(np.r_[d.qpos.copy(),d.qvel.copy()])
        hard=float(np.maximum(np.maximum(m.jnt_range[1:,0]-d.qpos[7:],d.qpos[7:]-m.jnt_range[1:,1]),0).max())
        drift=float(np.linalg.norm(d.qpos[:3]-q[:3]));up=float(d.xmat[root].reshape(3,3)[2,2])
        rows.append({'time_s':(step+1)*m.opt.timestep,'hard_violation_rad':hard,'root_drift_m':drift,'root_up':up})
        if hard>.001 or drift>.15 or up<.9:break
    np.savez_compressed(a.out/'hold.npz',state=np.array(states),controls=np.array(controls),dt=m.opt.timestep)
    report={'configuration':{k:str(v) for k,v in vars(a).items()},'solver_success':bool(fit.success),'nfev':fit.nfev,'native_calls':calls,
            'one_step_velocity_max':float(abs(residual(fit.x,False)).max()*.001),'seconds':len(states)*m.opt.timestep,
            'max_root_drift_m':max(r['root_drift_m'] for r in rows),'max_hard_violation_rad':max(r['hard_violation_rad'] for r in rows),
            'minimum_root_up':min(r['root_up'] for r in rows),'approved_for_training':False,'full_transition_verified':False}
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
