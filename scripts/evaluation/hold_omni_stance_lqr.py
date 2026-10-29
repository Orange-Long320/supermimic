"""Native motor-only whole-state feedback diagnostic for a fixed stance.

Linearizes the unchanged discrete MuJoCo plant. No root wrench, resets,
physics edits, policy training, or full motion approval.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['attribution','targets','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--seconds',type=float,default=5.)
    p.add_argument('--finite-horizon-steps',type=int,help='Use finite-horizon Riccati recursion when fixed contact leaves invariant world-frame modes')
    p.add_argument('--initial-root-velocity',nargs=3,type=float,default=[0.,0.,0.],help='Initial-only world translation velocity disturbance')
    a=p.parse_args()
    if a.out.exists() or not 2<=a.seconds<=10:p.error('New output and 2–10 second duration required')
    if a.finite_horizon_steps is not None and not 100<=a.finite_horizon_steps<=2000:p.error('Finite horizon must be 100–2000 native steps')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Explicit CPU-only server execution required')
    import mujoco
    import numpy as np
    from scipy.linalg import solve_discrete_are
    a.out.mkdir(parents=True);(a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    (a.out/'launch.json').write_text(json.dumps({'pid':os.getpid(),'configuration':{k:str(v) for k,v in vars(a).items()}},indent=2))
    report={'scope':__doc__,'policy_evaluation':False,'full_transition_verified':False,'tasks':{}}
    for name,row in json.loads(a.attribution.read_text()).items():
        folder=a.out/name;folder.mkdir();m=mujoco.MjModel.from_binary_path(row['model']);d=mujoco.MjData(m)
        if m.na or m.nu!=31:raise ValueError('Expected original X2 position actuators without activation states')
        source=a.targets/name/'terminal-candidates.npz';(folder/'terminal-candidates.npz').write_bytes(source.read_bytes())
        q=np.load(source)['candidate'];d.qpos[:]=q;d.qvel[:]=0;d.qacc[:]=0;mujoco.mj_inverse(m,d)
        joints=m.actuator_trnid[:,0].astype(int);qcols=m.jnt_qposadr[joints];vcols=m.jnt_dofadr[joints]
        u=q[qcols]+d.qfrc_inverse[vcols]/m.actuator_gainprm[:,0];d.ctrl[:]=u;mujoco.mj_forward(m,d)
        nx=2*m.nv;A=np.zeros((nx,nx));B=np.zeros((nx,m.nu))
        mujoco.mjd_transitionFD(m,d,1e-6,True,A,B,None,None)
        Q=np.diag(np.r_[np.full(3,2000.),np.full(3,1000.),np.full(m.nv-6,100.),np.full(6,20.),np.full(m.nv-6,.2)])
        R=np.eye(m.nu)
        if a.finite_horizon_steps is None:
            P=solve_discrete_are(A,B,Q,R)
        else:
            P=Q.copy()
            for _ in range(a.finite_horizon_steps):
                gain=np.linalg.solve(R+B.T@P@B,B.T@P@A)
                P=Q+A.T@P@(A-B@gain);P=(P+P.T)/2
        K=np.linalg.solve(R+B.T@P@B,B.T@P@A)
        radius=float(max(abs(np.linalg.eigvals(A-B@K))))
        if not np.isfinite(K).all() or radius>=1+1e-7:raise ValueError('Linear feedback failed local stability check')
        np.savez_compressed(folder/'linear-feedback.npz',A=A,B=B,K=K,Q=Q,R=R,nominal_control=u,reference_qpos=q)
        actual=mujoco.MjData(m);actual.qpos[:]=q;actual.qvel[:]=0;actual.qvel[:3]=a.initial_root_velocity;mujoco.mj_forward(m,actual)
        tangent=np.zeros(m.nv);states=[];controls=[];status='completed';root=int(m.jnt_bodyid[0]);rows=[]
        for step in range(round(a.seconds/m.opt.timestep)):
            mujoco.mj_differentiatePos(m,tangent,1.,q,actual.qpos)
            actual.ctrl[:]=u-K@np.r_[tangent,actual.qvel]
            controls.append(actual.ctrl.copy());mujoco.mj_step(m,actual);mujoco.mj_forward(m,actual)
            states.append(np.r_[actual.qpos.copy(),actual.qvel.copy()])
            hard=float(np.maximum(np.maximum(m.jnt_range[1:,0]-actual.qpos[7:],actual.qpos[7:]-m.jnt_range[1:,1]),0).max())
            up=float(actual.xmat[root].reshape(3,3)[2,2]);drift=float(np.linalg.norm(actual.qpos[:3]-q[:3]))
            rows.append({'time_s':(step+1)*m.opt.timestep,'hard_violation_rad':hard,'root_up':up,'root_drift_m':drift})
            if hard>.001 or up<.9 or drift>.15 or not np.isfinite(states[-1]).all():status='safety_stop';break
        np.savez_compressed(folder/'candidate-hold.npz',state=np.array(states),controls=np.array(controls),dt=m.opt.timestep)
        item={'status':status,'seconds':len(states)*m.opt.timestep,'linear_closed_loop_radius':radius,
              'initial_root_velocity':a.initial_root_velocity,
              'maximum_hard_violation_rad':max(x['hard_violation_rad'] for x in rows),
              'maximum_root_drift_m':max(x['root_drift_m'] for x in rows),'minimum_root_up':min(x['root_up'] for x in rows),
              'feedforward_mode':'inverse','control_mode':'native-discrete-LQR','state_resets_after_initialization':0,
              'model_sha256':hashlib.sha256(Path(row['model']).read_bytes()).hexdigest(),'approved_for_training':False}
        report['tasks'][name]=item;(a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({name:item}),flush=True)


if __name__=='__main__':main()
