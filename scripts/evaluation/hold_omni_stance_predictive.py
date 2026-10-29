"""Bounded motor feedback using the unchanged native one-step plant.

Only control derivatives are used; state derivatives across nonsmooth contact
are avoided. All applied forces are original joint actuators. Server CPU only.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['attribution','targets','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--seconds',type=float,default=5.)
    p.add_argument('--task')
    p.add_argument('--leg-controls-only',action='store_true',help='Restrict balance corrections to leg motors; upper body retains its fixed native servo targets')
    p.add_argument('--posture-weight',type=float,default=1.,help='Velocity tracking priority for all joint posture tasks')
    p.add_argument('--command-rate-rad-s',type=float,help='Limit each motor-command step and validate it with a native rollout')
    a=p.parse_args()
    if a.out.exists() or not 2<=a.seconds<=10:p.error('New output and 2–10 second duration required')
    if not 0<a.posture_weight<=100:p.error('Posture weight must be in (0,100]')
    if a.command_rate_rad_s is not None and not 0<a.command_rate_rad_s<=5:p.error('Command rate must be in (0,5]')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only server execution required')
    import mujoco
    import numpy as np
    from scipy.optimize import lsq_linear
    a.out.mkdir(parents=True);(a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    (a.out/'launch.json').write_text(json.dumps({'pid':os.getpid(),'configuration':{k:str(v) for k,v in vars(a).items()}},indent=2))
    report={'scope':__doc__,'policy_evaluation':False,'full_transition_verified':False,'tasks':{}}
    for name,row in json.loads(a.attribution.read_text()).items():
        if a.task and name!=a.task:continue
        folder=a.out/name;folder.mkdir();source=a.targets/name/'terminal-candidates.npz';(folder/'terminal-candidates.npz').write_bytes(source.read_bytes())
        q=np.load(source)['candidate'];m=mujoco.MjModel.from_binary_path(row['model']);initial=mujoco.MjData(m)
        initial.qpos[:]=q;mujoco.mj_inverse(m,initial)
        joint=m.actuator_trnid[:,0].astype(int);qc=m.jnt_qposadr[joint];vc=m.jnt_dofadr[joint]
        kp=m.actuator_gainprm[:,0];kd=-m.actuator_biasprm[:,2];cap=abs(m.actuator_forcerange).max(1)
        active=np.array([i for i,j in enumerate(joint) if not a.leg_controls_only or any(k in m.joint(j).name for k in ['hip_','knee_','ankle_'])],dtype=int)
        nominal=q[qc]+initial.qfrc_inverse[vc]/kp
        actual=mujoco.MjData(m);actual.qpos[:]=q;actual.ctrl[:]=nominal;mujoco.mj_forward(m,actual)
        linear=mujoco.MjData(m);prediction=mujoco.MjData(m)
        B=np.zeros((2*m.nv,m.nu));delta=np.zeros(m.nv);states=[];controls=[];rows=[];status='completed'
        root=int(m.jnt_bodyid[0]);weights=np.r_[np.full(6,30.),np.full(m.nv-6,a.posture_weight)]
        for step in range(round(a.seconds/m.opt.timestep)):
            base_control=actual.ctrl.copy() if a.command_rate_rad_s is not None else nominal.copy()
            mujoco.mj_copyData(linear,m,actual);linear.ctrl[:]=base_control;mujoco.mj_forward(m,linear)
            mujoco.mjd_transitionFD(m,linear,1e-4,True,None,B,None,None)
            mujoco.mj_copyData(prediction,m,actual);prediction.ctrl[:]=base_control;mujoco.mj_step(m,prediction)
            mujoco.mj_differentiatePos(m,delta,1.,q,actual.qpos)
            accel=-np.r_[np.full(6,40.),np.full(m.nv-6,80.)]*delta-np.r_[np.full(6,12.),np.full(m.nv-6,18.)]*actual.qvel
            wanted=actual.qvel+m.opt.timestep*accel
            matrix=np.vstack([B[m.nv:,active]*weights[:,None],np.eye(len(active))*.01])
            rhs=np.r_[(wanted-prediction.qvel)*weights,np.zeros(len(active))]
            lo=np.maximum(nominal-.05,actual.qpos[qc]+(-cap+kd*actual.qvel[vc])/kp)-base_control
            hi=np.minimum(nominal+.05,actual.qpos[qc]+(cap+kd*actual.qvel[vc])/kp)-base_control
            if a.command_rate_rad_s is not None:
                limit=a.command_rate_rad_s*m.opt.timestep;lo=np.maximum(lo,-limit);hi=np.minimum(hi,limit)
            if (lo[active]>=hi[active]).any():status='control_bounds_rejected';break
            fit=lsq_linear(matrix,rhs,bounds=(lo[active],hi[active]),tol=1e-8,max_iter=60)
            correction=np.zeros(m.nu);correction[active]=fit.x
            if a.command_rate_rad_s is not None:
                base_cost=float(np.square((wanted-prediction.qvel)*weights).sum())
                for contraction in range(12):
                    mujoco.mj_copyData(prediction,m,actual);prediction.ctrl[:]=base_control+correction
                    mujoco.mj_step(m,prediction)
                    if float(np.square((wanted-prediction.qvel)*weights).sum())<=base_cost+1e-12:break
                    correction*=.5
                else:correction[:]=0.
            actual.ctrl[:]=base_control+correction;controls.append(actual.ctrl.copy())
            mujoco.mj_step(m,actual);mujoco.mj_forward(m,actual)
            states.append(np.r_[actual.qpos.copy(),actual.qvel.copy()])
            hard=float(np.maximum(np.maximum(m.jnt_range[1:,0]-actual.qpos[7:],actual.qpos[7:]-m.jnt_range[1:,1]),0).max())
            up=float(actual.xmat[root].reshape(3,3)[2,2]);drift=float(np.linalg.norm(actual.qpos[:3]-q[:3]))
            rows.append({'time_s':(step+1)*m.opt.timestep,'hard_violation_rad':hard,'root_up':up,'root_drift_m':drift,
                         'control_correction_max_rad':float(abs(fit.x).max()),'least_squares_success':bool(fit.success)})
            if step%100==0:print(json.dumps({'task':name,**rows[-1]}),flush=True)
            if hard>.001 or up<.9 or drift>.15 or not np.isfinite(states[-1]).all():status='safety_stop';break
        np.savez_compressed(folder/'candidate-hold.npz',state=np.array(states),controls=np.array(controls),dt=m.opt.timestep)
        item={'status':status,'seconds':len(states)*m.opt.timestep,'feedforward_mode':'inverse','control_mode':'native-one-step-feedback',
              'leg_controls_only':a.leg_controls_only,
              'maximum_hard_violation_rad':max((x['hard_violation_rad'] for x in rows),default=0.),
              'maximum_root_drift_m':max((x['root_drift_m'] for x in rows),default=0.),'minimum_root_up':min((x['root_up'] for x in rows),default=0.),
              'state_resets_after_initialization':0,'approved_for_training':False}
        report['tasks'][name]=item;(a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        (folder/'steps.json').write_text(json.dumps(rows,indent=2));print(json.dumps({name:item}),flush=True)


if __name__=='__main__':main()
