"""Continuous native CPU servo diagnostic; no state resets after initialization.

This is a bounded PD/feedforward tracking check, not a learned-policy evaluation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import mujoco
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--start-frame',type=int,default=0)
    p.add_argument('--seconds',type=float,default=5.)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    m=mujoco.MjModel.from_binary_path(str(args.model))
    actual=mujoco.MjData(m);desired=mujoco.MjData(m)
    z=np.load(args.reference);q=z['qpos'];v=z['qvel']
    if q.ndim!=2 or q.shape[1]!=m.nq or v.shape!=(len(q),m.nv):
        raise ValueError('Reference dimensions do not match native model')
    fps=float(z['fps'].reshape(-1)[0]) if 'fps' in z else 50.
    if not 0<=args.start_frame<len(q):raise ValueError('Invalid start frame')
    joints=m.actuator_trnid[:,0].astype(int)
    qcols=m.jnt_qposadr[joints];vcols=m.jnt_dofadr[joints]
    kp=m.actuator_gainprm[:,0].copy();kd=-m.actuator_biasprm[:,2].copy()
    if np.any(kp<=0) or not np.all(m.actuator_forcelimited):
        raise ValueError('Expected native limited position actuators')
    actual.qpos[:]=q[args.start_frame];actual.qvel[:]=v[args.start_frame]
    mujoco.mj_forward(m,actual)
    acc=np.zeros_like(v)
    tangent=np.zeros(m.nv);backward=np.zeros(m.nv)
    for i in range(1,len(q)-1):
        mujoco.mj_differentiatePos(m,tangent,1/fps,q[i],q[i+1])
        mujoco.mj_differentiatePos(m,backward,-1/fps,q[i],q[i-1])
        acc[i]=(tangent-backward)*fps
    root=int(m.jnt_bodyid[0]);rows=[];states=[];status='completed'
    maximum_joint_error=0.;maximum_hard_violation=0.;maximum_effort_ratio=0.
    lo,hi=m.jnt_range[1:].T
    for step in range(round(args.seconds/m.opt.timestep)):
        t=step*m.opt.timestep
        f=min(args.start_frame+t*fps,len(q)-1)
        a=int(f);b=min(a+1,len(q)-1);blend=f-a
        target=q[a].copy()
        mujoco.mj_differentiatePos(m,tangent,1.,q[a],q[b])
        mujoco.mj_integratePos(m,target,tangent,blend)
        target_v=(1-blend)*v[a]+blend*v[b]
        target_a=(1-blend)*acc[a]+blend*acc[b]
        desired.qpos[:]=target;desired.qvel[:]=target_v;desired.qacc[:]=target_a
        mujoco.mj_inverse(m,desired)
        feedforward=desired.qfrc_inverse[vcols]
        actual.ctrl[:]=target[qcols]+(kd*target_v[vcols]+feedforward)/kp
        # Native actuator force limits remain active. Only joint motors act.
        mujoco.mj_step(m,actual)
        hard=float(np.maximum(np.maximum(lo-actual.qpos[7:],actual.qpos[7:]-hi),0).max())
        ratio=float((abs(actual.actuator_force)/np.maximum(abs(m.actuator_forcerange).max(axis=1),1e-9)).max())
        error=float(abs(actual.qpos[qcols]-target[qcols]).max())
        maximum_hard_violation=max(maximum_hard_violation,hard)
        maximum_effort_ratio=max(maximum_effort_ratio,ratio)
        maximum_joint_error=max(maximum_joint_error,error)
        mujoco.mj_forward(m,actual)
        up=float(actual.xmat[root].reshape(3,3)[2,2])
        rows.append({'time_s':t+m.opt.timestep,'root_height_m':float(actual.qpos[2]),
                     'root_up_z':up,'hard_violation_rad':hard,'joint_error_max_rad':error})
        states.append(np.r_[actual.qpos.copy(),actual.qvel.copy()])
        if not np.isfinite(states[-1]).all() or actual.qpos[2]<.22 or up<.3:
            status='lost_balance';break
    np.savez(args.out/'trajectory.npz',state=np.asarray(states),dt=np.array(m.opt.timestep))
    report={'configuration':{k:str(x) if isinstance(x,Path) else x for k,x in vars(args).items()},
            'pid':os.getpid(),'python':sys.executable,'mujoco_version':mujoco.__version__,
            'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),
            'reference_sha256':hashlib.sha256(args.reference.read_bytes()).hexdigest(),
            'status':status,'seconds_executed':len(rows)*m.opt.timestep,
            'maximum_hard_violation_rad':maximum_hard_violation,
            'maximum_effort_ratio':maximum_effort_ratio,
            'maximum_joint_error_rad':maximum_joint_error,
            'last':rows[-1] if rows else None,'rows':rows,
            'policy_evaluation':False,'state_resets_after_initialization':0,
            'approved_for_training':False,
            'limitations':'A single PD plus inverse-feedforward controller. Failure does not establish trajectory impossibility; completion does not certify contact/limit/steady-state gates.'}
    (args.out/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:x for k,x in report.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
