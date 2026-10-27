"""Bounded native-physics Webster control probe; never a training approval."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import mujoco
import numpy as np
import torch
from scipy.interpolate import BPoly
from scipy.spatial.transform import Rotation


def curve(start,end,duration,velocity=None):
    return BPoly.from_derivatives([0.,duration],[[start,np.zeros_like(start),np.zeros_like(start)],
        [end,np.zeros_like(end) if velocity is None else velocity,np.zeros_like(end)]])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--equilibrium',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--seconds',type=float,default=2.5)
    p.add_argument('--push-seconds',type=float,default=.5);p.add_argument('--rise',type=float,default=.16)
    p.add_argument('--launch-vz',type=float,default=2.);p.add_argument('--advance',type=float,default=.25)
    p.add_argument('--launch-vx',type=float,default=1.);p.add_argument('--momentum',type=float,default=24.)
    p.add_argument('--toe-angle',type=float,default=1.1);p.add_argument('--swing-angle',type=float,default=.5)
    p.add_argument('--crouch',type=float,default=.03)
    p.add_argument('--entry-vx',type=float,default=0.);p.add_argument('--delay',type=float,default=.15)
    p.add_argument('--momentum-delay-fraction',type=float,default=0.)
    p.add_argument('--direct-rise',action='store_true',help='Use a single monotone rise rather than a preparatory dip')
    p.add_argument('--ground-root-weight',type=float,default=0.)
    p.add_argument('--ground-root-pitch',type=float,default=.4)
    p.add_argument('--extend-support',action='store_true')
    p.add_argument('--joint-task-weight',type=float,default=.01)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    program_source=Path(__file__).read_text()
    (args.out/'probe-source.py').write_text(program_source)
    from supermimic.envs import SuperMimicEnvCfgArgs,make_env
    from supermimic.envs.state_io import StateIO
    from supermimic.policy.native_inverse import NativeContactController
    from supermimic.data.design_metrics import BatchedDesignMetrics
    from supermimic.robots.dimensions import PROFILE
    snapshots={}
    for relative in ['src/supermimic/policy/native_inverse.py','src/supermimic/policy/whole_body.py',
                     'src/supermimic/policy/joint_prediction.py','src/supermimic/data/entry_state.py']:
        data=Path(relative).read_bytes();(args.out/Path(relative).name).write_bytes(data)
        snapshots[relative]=hashlib.sha256(data).hexdigest()
    env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(args.source)],num_envs=1,action_noise_std=0.,
        obs_noise=False,reset_noise=False,domain_randomization=False,sampling_mode='start',
        record_transitions=False,episode_length_s=1e5),'cpu')
    traces=[];metrics_trace=[];solutions=[];controls=[];phases=[];accelerations=[]
    status='failed';error=None;started=time.monotonic();controller=None
    try:
        env.command_manager.get_term('motion').forced_frame[:]=0;env.reset()
        m=env.sim.mj_model;io=StateIO(env);state=io.read();names=list(PROFILE['joint_names'])
        with np.load(args.equilibrium) as eq:
            assert eq['joint_names'].tolist()==names
            state[0,:m.nq]=torch.tensor(eq['qpos'],dtype=state.dtype);state[0,m.nq:]=0
        entry=None
        if args.entry_vx:
            from supermimic.data.entry_state import entry_velocity_state
            speed_limit=np.array([PROFILE['velocity_limit'][n] for n in names])
            incoming,entry=entry_velocity_state(m,state[0,:m.nq].numpy(),np.zeros(m.nv),args.entry_vx,speed_limit)
            state[0,m.nq:]=torch.tensor(incoming,dtype=state.dtype)
        io.write(torch.tensor([0]),state)
        np.savez(args.out/'initial-state.npz',state=state.numpy());mujoco.mj_saveModel(m,str(args.out/'native-model.mjb'))
        kp=np.array([PROFILE['stiffness'][n] for n in names]);kd=np.array([PROFILE['damping'][n] for n in names])
        effort=np.array([PROFILE['effort_limit'][n] for n in names]);velocity=np.array([PROFILE['velocity_limit'][n] for n in names])
        q0=state[0,:m.nq].numpy().copy();foot=m.body('robot/left_ankle_roll_link').id
        controller=NativeContactController(m,q0,effort,velocity,foot)
        metrics=BatchedDesignMetrics(m,torch.device('cpu'));lo,hi=m.jnt_range[1:].T
        com0=controller.com_ref.copy();push=args.push_seconds;delay=args.delay;deadline=delay+push
        launch=com0+np.array([args.advance,0,args.rise]);launch_velocity=np.array([args.launch_vx,0,args.launch_vz])
        midpoint=com0+np.array([args.advance*(.55 if args.entry_vx else .2),0,-args.crouch])
        mid_v=np.array([args.entry_vx*.65,0,0])
        com_curve=BPoly.from_derivatives([0.,push*.5,push],[[com0,np.array([args.entry_vx,0,0]),np.zeros(3)],
            [midpoint,mid_v,np.zeros(3)],[launch,launch_velocity,np.zeros(3)]])
        if args.direct_rise:
            com_curve=BPoly.from_derivatives([0.,push],[[com0,np.array([args.entry_vx,0,0]),np.zeros(3)],
                [launch,launch_velocity,np.zeros(3)]])
        momentum_start=push*args.momentum_delay_fraction;momentum_duration=push-momentum_start
        hcurve=curve(np.zeros(3),np.array([0,args.momentum,0]),momentum_duration,np.array([0,args.momentum/momentum_duration,0]))
        swing=q0.copy();swing[7+names.index('right_hip_pitch_joint')]=args.swing_angle
        swing[7+names.index('right_knee_joint')]=2.2
        if args.extend_support:
            swing[7+names.index('left_hip_pitch_joint')]=-.1
            swing[7+names.index('left_knee_joint')]=.08
        for side in ['left','right']:
            swing[7+names.index(side+'_shoulder_pitch_joint')]=-.7
            swing[7+names.index(side+'_elbow_joint')]=-1.4
        swing[7:]=np.clip(swing[7:],lo+.025,hi-.025)
        swing_curve=curve(q0[7:],swing[7:],push)
        tuck=q0.copy()
        for side in ['left','right']:
            for suffix,value in [('hip_pitch_joint',-2.4),('knee_joint',2.65),('ankle_pitch_joint',-.1),
                                 ('shoulder_pitch_joint',-1.),('elbow_joint',-1.4)]:
                tuck[7+names.index(side+'_'+suffix)]=value
        tuck[7+names.index('waist_pitch_joint')]=.25;tuck[7:]=np.clip(tuck[7:],lo+.025,hi-.025)
        land=q0.copy()
        for side in ['left','right']:
            for suffix,value in [('hip_pitch_joint',-.6),('knee_joint',1.2),('ankle_pitch_joint',-.6)]:
                land[7+names.index(side+'_'+suffix)]=value
        land[7:]=np.clip(land[7:],lo+.025,hi-.025)
        foot0=controller.foot_position.copy();rot0=controller.foot_rotation
        pivot=foot0+rot0.apply([.12,0,-.075]);pivot[2]=0
        original_step=env.sim.step;dt=float(m.opt.timestep);rotation_sum=0.
        last_rotation=Rotation.from_quat(q0[3:7][[1,2,3,0]])
        root_rotation0=last_rotation
        root_curve=curve(np.zeros(3),np.array([0,args.ground_root_pitch,0]),push,np.array([0,2.,0]))
        airborne=None;unfold=None;landed=None;air_count=0;flight_curve=None;unfold_curve=None
        def sample(poly,t):return poly(t),poly.derivative()(t),poly.derivative(2)(t)
        def checked_step(*a,**kw):
            nonlocal airborne,air_count,rotation_sum,last_rotation,flight_curve,unfold,unfold_curve,landed
            t=len(traces)*dt;q=env.sim.data.qpos[0].numpy().copy();v=env.sim.data.qvel[0].numpy().copy()
            measured=metrics(env.sim.data)[0].numpy();d=controller.data;d.qpos[:]=q;d.qvel[:]=v;mujoco.mj_forward(m,d)
            _,cj,hj=controller.jacobians(d);com=d.subtree_com[controller.root].copy();cv=cj@v;h=hj@v
            air_count=air_count+1 if min(measured[7:9])>.005 and t>delay else 0
            if airborne is None and air_count>=3:
                airborne={'time':t,'com':com.copy(),'velocity':cv.copy(),'momentum':h.copy(),'rotation':rotation_sum}
                flight_curve=curve(q[7:],tuck[7:],.15)
            if airborne is not None and landed is None and t-airborne['time']>.12 and min(measured[7:9])<=.003 and cv[2]<0:
                landed={'time':t,'com':com.copy()}
            qref=q0.copy();jv=np.zeros(m.nv-6);ja=jv.copy();hr=np.zeros(3)
            if airborne is None:
                phase='support' if t<delay else 'push';u=float(np.clip(t-delay,0,push))
                ct,cvt,cat=sample(com_curve,u);ht,hv,_=sample(hcurve,max(0.,u-momentum_start));hr=hv
                if t<delay:ct=com0.copy();cvt=np.zeros(3);cat=np.zeros(3);ht=np.zeros(3);hr=np.zeros(3)
                qref[7:],jv,ja=sample(swing_curve,u)
                toe_duration=min(.25,push);toe_u=float(np.clip((t-(deadline-toe_duration))/toe_duration,0,1))
                smooth=toe_u**3*(10-15*toe_u+6*toe_u**2)
                angle=args.toe_angle*smooth;rate=args.toe_angle*30*toe_u**2*(1-toe_u)**2/toe_duration
                acceleration=args.toe_angle*60*toe_u*(1-3*toe_u+2*toe_u**2)/toe_duration**2
                turn=Rotation.from_rotvec([0,angle,0]);arm=turn.apply(foot0-pivot);axis=np.array([0,1,0])
                controller.foot_position=pivot+arm;controller.foot_rotation=turn*rot0
                controller.foot_velocity_ref=np.r_[rate*np.cross(axis,arm),axis*rate]
                controller.foot_acceleration_ref=np.r_[acceleration*np.cross(axis,arm)+rate**2*np.cross(axis,np.cross(axis,arm)),axis*acceleration]
                if args.ground_root_weight:
                    root_angle,root_v,root_a=sample(root_curve,u)
                    current=Rotation.from_quat(q[3:7][[1,2,3,0]])
                    desired=Rotation.from_rotvec(root_angle)*root_rotation0
                    error_body=current.inv().apply((desired*current.inv()).as_rotvec())
                    controller.root_acceleration_ref=current.inv().apply(root_a)+60*error_body+15*(current.inv().apply(root_v)-v[3:6])
                    controller.root_acceleration_weight=args.ground_root_weight
            else:
                if hasattr(controller,'root_acceleration_ref'):del controller.root_acceleration_ref
                phase='flight';u=t-airborne['time'];ct=airborne['com']+airborne['velocity']*u+[0,0,-.5*9.81*u*u]
                cvt=airborne['velocity']+np.array([0,0,-9.81*u]);cat=np.array([0,0,-9.81]);ht=airborne['momentum']
                qref[7:],jv,ja=sample(flight_curve,min(u,.15))
                if u>=.15:jv[:]=0;ja[:]=0
                if unfold is None and (rotation_sum>4.2 or u>.5):
                    unfold=t;unfold_curve=curve(q[7:],land[7:],.2)
                if unfold is not None:
                    phase='unfold';u_land=min(t-unfold,.2);qref[7:],jv,ja=sample(unfold_curve,u_land)
                    if t-unfold>=.2:jv[:]=0;ja[:]=0
                if landed is not None:
                    phase='landing';ct=landed['com'].copy();ct[2]=com0[2];cvt=np.zeros(3);cat=np.zeros(3);ht=np.zeros(3)
            controller.joint_task_weight=args.joint_task_weight
            tau,solution=controller.solve(q,v,com_target=ct,com_velocity=cvt,com_acceleration=cat,
                momentum_target=ht,momentum_rate=hr,qref=qref,joint_velocity=jv,joint_acceleration=ja)
            solutions.append({'time':t,'phase':phase,**solution})
            target=q[7:]+(tau+kd*v[6:])/kp;env.sim.data.ctrl[0]=torch.tensor(target,dtype=env.sim.data.ctrl.dtype)
            result=original_step(*a,**kw)
            qnext=env.sim.data.qpos[0].numpy().copy();vnext=env.sim.data.qvel[0].numpy().copy()
            rotation=Rotation.from_quat(qnext[3:7][[1,2,3,0]]);rotation_sum+=(rotation*last_rotation.inv()).as_rotvec()[1];last_rotation=rotation
            values=metrics(env.sim.data)[0].numpy();actual_tau=env.sim.data.actuator_force[0].numpy()
            traces.append(np.r_[qnext,vnext]);controls.append(target);phases.append(phase)
            accelerations.append([controller.previous*20,env.sim.data.qacc[0].numpy().copy()])
            metrics_trace.append(np.r_[values,np.maximum(np.maximum(lo-qnext[7:],qnext[7:]-hi),0).max(),
                (abs(actual_tau)/effort).max(),(abs(vnext[6:])/velocity).max(),rotation_sum])
            return result
        env.sim.step=checked_step;env.termination_manager.compute=lambda:torch.zeros(1,dtype=torch.bool)
        for _ in range(round(args.seconds/env.step_dt)):env.step(torch.zeros((1,len(names))))
        status='completed'
    except Exception as exc:error=str(exc)
    finally:
        if traces:np.savez(args.out/'trajectory.npz',state=np.asarray(traces),metrics=np.asarray(metrics_trace),
            servo_target=np.asarray(controls),phase=np.asarray(phases),predicted_and_actual_acceleration=np.asarray(accelerations),fps=np.array([200.]))
        report={'status':status,'error':error,'elapsed_s':time.monotonic()-started,'physical_steps':len(traces),
            'approved_for_training':False,'configuration':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            'pid':os.getpid(),'source_sha256':hashlib.sha256(program_source.encode()).hexdigest(),'module_sha256':snapshots,
            'state_edits_after_start':False,'solutions':solutions}
        report['entry_initial_condition']=locals().get('entry')
        if traces:
            x=np.asarray(metrics_trace);report['maxima']={'hard_violation':float(x[:,12].max()),'effort_ratio':float(x[:,13].max()),
                'speed_ratio':float(x[:,14].max()),'minimum_nonfoot_height':float(x[:,9].min()),'minimum_left_height':float(x[:,7].min()),
                'minimum_right_height':float(x[:,8].min()),'rotation':float(x[-1,15]),'max_vz':float(x[:,1].max()),'max_Hy':float(x[:,3].max())}
            report['tail']={'up':float(x[-40:,11].min()),'speed':float(x[-40:,5].max()),'com_height':float(x[-40:,0].min())}
        for name in ['airborne','landed']:
            value=locals().get(name)
            report[name]=None if value is None else {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in value.items()}
        if controller is not None and hasattr(controller,'failure'):
            report['failed_constraint_audit']=controller.failure;np.savez(args.out/'failed-qp.npz',**controller.failure_arrays)
        (args.out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='solutions'}),flush=True)
        env.close()
    if status!='completed':raise SystemExit(1)


if __name__=='__main__':main()
