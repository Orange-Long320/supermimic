"""Bounded native-physics Webster control probe; never a training approval."""
import argparse
import hashlib
import json
import os
import time
from types import SimpleNamespace
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
    p.add_argument('--device',default='cpu')
    p.add_argument('--hold-toe-pitch',action='store_true')
    p.add_argument('--require-backswing',action='store_true')
    p.add_argument('--backswing-min-speed',type=float,default=2.)
    p.add_argument('--forward-seed',action='store_true',help='Retry rejected inverse solves from the previous command native forward acceleration')
    p.add_argument('--maximum-push-coast',action='store_true',help='Continue the incoming forward COM speed during the push')
    p.add_argument('--integration-aware',action='store_true')
    p.add_argument('--source',type=Path,required=True);p.add_argument('--equilibrium',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--seconds',type=float,default=2.5)
    p.add_argument('--push-seconds',type=float,default=.5);p.add_argument('--rise',type=float,default=.16)
    p.add_argument('--launch-vz',type=float,default=2.);p.add_argument('--advance',type=float,default=.25)
    p.add_argument('--launch-vx',type=float,default=1.);p.add_argument('--momentum',type=float,default=24.)
    p.add_argument('--toe-angle',type=float,default=1.1);p.add_argument('--swing-angle',type=float,default=.5)
    p.add_argument('--crouch',type=float,default=.03)
    p.add_argument('--entry-hy',type=float,default=0.);p.add_argument('--entry-vz',type=float,default=0.)
    p.add_argument('--predictive-filter-device',help='Independent Warp planning worlds for motor-command feasibility')
    p.add_argument('--filter-steps',type=int,default=4)
    p.add_argument('--maximum-push',type=float,help='Desired vertical acceleration during support; original motor caps remain active')
    p.add_argument('--side-momentum-weight',type=float,default=.1)
    p.add_argument('--foot-floor',type=float)
    p.add_argument('--entry-vx',type=float,default=0.);p.add_argument('--delay',type=float,default=.15)
    p.add_argument('--momentum-delay-fraction',type=float,default=0.)
    p.add_argument('--direct-rise',action='store_true',help='Use a single monotone rise rather than a preparatory dip')
    p.add_argument('--ground-root-weight',type=float,default=0.)
    p.add_argument('--ground-root-pitch',type=float,default=.4)
    p.add_argument('--extend-support',action='store_true')
    p.add_argument('--joint-task-weight',type=float,default=.01)
    p.add_argument('--support-reference',type=Path)
    p.add_argument('--foot-task-weight',type=float,default=.05)
    p.add_argument('--scheduled-release',action='store_true',help='Start commanded leg retraction at the reference boundary; record actual flight independently')
    p.add_argument('--vertical-task-weight',type=float,default=2.)
    p.add_argument('--pitch-momentum-weight',type=float,default=.1)
    args=p.parse_args()
    if args.maximum_push is not None and args.support_reference:p.error('Maximum push and a prescribed push reference are separate modes')
    args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    program_source=Path(__file__).read_text()
    (args.out/'probe-source.py').write_text(program_source)
    from supermimic.envs import SuperMimicEnvCfgArgs,make_env
    from supermimic.envs.state_io import StateIO
    from supermimic.policy.native_inverse import NativeContactController
    from supermimic.data.design_metrics import BatchedDesignMetrics
    from supermimic.robots.dimensions import PROFILE
    snapshots={}
    for relative in ['src/supermimic/policy/native_inverse.py','src/supermimic/policy/whole_body.py',
                     'src/supermimic/policy/joint_prediction.py','src/supermimic/data/entry_state.py',
                     'src/supermimic/kinematics/support_reference.py']:
        data=Path(relative).read_bytes();(args.out/Path(relative).name).write_bytes(data)
        snapshots[relative]=hashlib.sha256(data).hexdigest()
    env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(args.source)],num_envs=1,action_noise_std=0.,
        obs_noise=False,reset_noise=False,domain_randomization=False,sampling_mode='start',
        record_transitions=False,episode_length_s=1e5),args.device)
    traces=[];metrics_trace=[];solutions=[];controls=[];phases=[];accelerations=[];velocity_accelerations=[]
    status='failed';error=None;started=time.monotonic();controller=None;command_filter=None;filter_reports=[]
    try:
        env.command_manager.get_term('motion').forced_frame[:]=0;env.reset()
        m=env.sim.mj_model;io=StateIO(env);state=io.read();names=list(PROFILE['joint_names'])
        with np.load(args.equilibrium) as eq:
            assert eq['joint_names'].tolist()==names
            state[0,:m.nq]=torch.tensor(eq['qpos'],dtype=state.dtype);state[0,m.nq:]=0
        entry=None
        reference=None
        if args.support_reference:
            from supermimic.kinematics.support_reference import SupportReference
            reference=SupportReference(args.support_reference)
            np.testing.assert_allclose(reference.arrays['qpos'][0],state[0,:m.nq].cpu().numpy(),atol=1e-6)
            initial=reference.sample(0)
            speed_limits=np.array([PROFILE['velocity_limit'][n] for n in names])
            speed_ratio=float((abs(initial['qvel'][6:])/speed_limits).max())
            if speed_ratio>1:raise ValueError('Reference initial joint velocity exceeds limits')
            if args.delay and np.linalg.norm(initial['qvel'])>.01:raise ValueError('Moving reference requires zero delay')
            state[0,m.nq:]=torch.tensor(initial['qvel'],dtype=state.dtype)
            entry={'scope':'Initial state only; coordinated reference velocity','joint_speed_ratio':speed_ratio,
                'com_velocity':initial['com_velocity'].tolist(),'momentum':initial['momentum'].tolist(),
                'reference_sha256':hashlib.sha256(args.support_reference.read_bytes()).hexdigest()}
        elif any([args.entry_vx,args.entry_vz,args.entry_hy]):
            from supermimic.data.entry_state import entry_velocity_state
            speed_limit=np.array([PROFILE['velocity_limit'][n] for n in names])
            incoming,entry=entry_velocity_state(m,state[0,:m.nq].cpu().numpy(),np.zeros(m.nv),args.entry_vx,speed_limit,
                com_velocity=[args.entry_vx,0.,args.entry_vz],angular_momentum=[0.,args.entry_hy,0.])
            state[0,m.nq:]=torch.tensor(incoming,dtype=state.dtype)
        io.write(torch.tensor([0],device=args.device),state)
        np.savez(args.out/'initial-state.npz',state=state.cpu().numpy());mujoco.mj_saveModel(m,str(args.out/'native-model.mjb'))
        kp=np.array([PROFILE['stiffness'][n] for n in names]);kd=np.array([PROFILE['damping'][n] for n in names])
        effort=np.array([PROFILE['effort_limit'][n] for n in names]);velocity=np.array([PROFILE['velocity_limit'][n] for n in names])
        q0=state[0,:m.nq].cpu().numpy().copy();foot=m.body('robot/left_ankle_roll_link').id
        controller=NativeContactController(m,q0,effort,velocity,foot)
        controller.integration_aware=args.integration_aware
        controller.foot_task_weight=args.foot_task_weight
        controller.com_task_weight=np.array([2.,2.,args.vertical_task_weight])
        controller.momentum_task_weight=np.array([args.side_momentum_weight,args.pitch_momentum_weight,args.side_momentum_weight])
        controller.foot_geometry_floor=args.foot_floor
        if args.predictive_filter_device:
            from supermimic.policy.warp_command_filter import WarpCommandFilter
            command_filter=WarpCommandFilter(args.source,m,args.predictive_filter_device,args.filter_steps)
            data=Path('src/supermimic/policy/warp_command_filter.py').read_bytes()
            (args.out/'warp_command_filter.py').write_bytes(data);snapshots['src/supermimic/policy/warp_command_filter.py']=hashlib.sha256(data).hexdigest()
        with np.load(args.equilibrium) as eq:initial_torque=eq['required_motor_torque'].copy()
        previous_control=q0[7:]+(initial_torque+kd*state[0,m.nq+6:].cpu().numpy())/kp
        metrics=BatchedDesignMetrics(m,torch.device('cpu'));lo,hi=m.jnt_range[1:].T
        observation=mujoco.MjData(m)
        def observe(q,v):
            observation.qpos[:]=q;observation.qvel[:]=v;mujoco.mj_forward(m,observation)
            arrays={key:torch.as_tensor(np.array(getattr(observation,key)),dtype=torch.float32)[None]
                    for key in ['xpos','xquat','cvel','geom_xpos','geom_xmat','qvel']}
            return metrics(SimpleNamespace(**arrays))[0].cpu().numpy()
        com0=controller.com_ref.copy();push=reference.duration if reference else args.push_seconds;delay=args.delay;deadline=delay+push
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
        if args.maximum_push is not None:
            swing_curve=BPoly.from_derivatives([0.,push],[[q0[7:],state[0,m.nq+6:].cpu().numpy(),np.zeros(m.nv-6)],
                [swing[7:],np.zeros(m.nv-6),np.zeros(m.nv-6)]])
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
        initial_foot_pitch=float(np.arctan2(rot0.as_matrix()[0,2],rot0.as_matrix()[2,2]))
        swing_dof=int(m.jnt_dofadr[m.joint('robot/right_hip_pitch_joint').id])
        swing_qpos=int(m.jnt_qposadr[m.joint('robot/right_hip_pitch_joint').id])
        pivot=foot0+rot0.apply([.12,0,-.075]);pivot[2]=0
        original_step=env.sim.step;dt=float(m.opt.timestep);rotation_sum=0.
        last_rotation=Rotation.from_quat(q0[3:7][[1,2,3,0]])
        root_rotation0=last_rotation
        root_curve=curve(np.zeros(3),np.array([0,args.ground_root_pitch,0]),push,np.array([0,2.,0]))
        airborne=None;release=None;unfold=None;landed=None;air_count=0;flight_curve=None;unfold_curve=None;fold_duration=.15
        def sample(poly,t):return poly(t),poly.derivative()(t),poly.derivative(2)(t)
        def checked_step(*a,**kw):
            nonlocal airborne,release,air_count,rotation_sum,last_rotation,flight_curve,unfold,unfold_curve,landed,fold_duration,previous_control
            t=len(traces)*dt;q=env.sim.data.qpos[0].cpu().numpy().copy();v=env.sim.data.qvel[0].cpu().numpy().copy()
            measured=observe(q,v);d=controller.data;d.qpos[:]=q;d.qvel[:]=v;mujoco.mj_forward(m,d)
            _,cj,hj=controller.jacobians(d);com=d.subtree_com[controller.root].copy();cv=cj@v;h=hj@v
            air_count=air_count+1 if min(measured[7:9])>.005 and t>delay else 0
            if airborne is None and air_count>=3:
                airborne={'time':t,'com':com.copy(),'velocity':cv.copy(),'momentum':h.copy(),'rotation':rotation_sum}
                if release is None:
                    fold_duration=max(.15,1.97*float((abs(tuck[7:]-q[7:])/velocity).max()))
                    flight_curve=curve(q[7:],tuck[7:],fold_duration)
            if args.scheduled_release and (reference is not None or args.maximum_push is not None) and release is None and t>=deadline and airborne is None:
                release={'time':t,'com':com.copy(),'velocity':cv.copy(),'momentum':h.copy(),'rotation':rotation_sum}
                fold_duration=max(.15,1.97*float((abs(tuck[7:]-q[7:])/velocity).max()))
                flight_curve=curve(q[7:],tuck[7:],fold_duration)
            if airborne is not None and landed is None and t-airborne['time']>.12 and min(measured[7:9])<=.003 and cv[2]<0:
                landed={'time':t,'com':com.copy()}
            qref=q0.copy();jv=np.zeros(m.nv-6);ja=jv.copy();hr=np.zeros(3)
            if airborne is None and release is None:
                phase='support' if t<delay else 'push';u=float(np.clip(t-delay,0,push))
                ct,cvt,cat=sample(com_curve,u);ht,hv,_=sample(hcurve,max(0.,u-momentum_start));hr=hv
                if t<delay:ct=com0.copy();cvt=np.zeros(3);cat=np.zeros(3);ht=np.zeros(3);hr=np.zeros(3)
                if args.maximum_push is not None and t>=delay:
                    ct=com0.copy();forward=args.entry_vx if args.maximum_push_coast else 0.
                    ct[0]+=forward*(t-delay);cvt=np.array([forward,0.,cv[2]]);cat=np.array([0.,0.,args.maximum_push])
                    ht=np.array([0.,args.momentum,0.]);hr=np.zeros(3)
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
                if reference is not None:
                    if t>deadline+.15:raise RuntimeError('Coordinated takeoff missed its deadline')
                    r=reference.sample(min(t-delay,reference.duration))
                    ct=r['com'];cvt=r['com_velocity'];cat=r['com_acceleration']
                    ht=r['momentum'];hr=r['momentum_rate'];qref=r['qpos'];jv=r['qvel'][6:];ja=r['qacc'][6:]
                    controller.foot_position=r['foot_position'];controller.foot_rotation=Rotation.from_quat(r['foot_quat'])
                    controller.foot_velocity_ref=r['foot_velocity'];controller.foot_acceleration_ref=r['foot_acceleration']
                    if t>deadline:
                        over=t-deadline;ct=ct+cvt*over+np.array([0,0,-.5*9.81*over*over])
                        cvt=cvt+np.array([0,0,-9.81*over]);cat=np.array([0,0,-9.81]);hr=np.zeros(3);jv[:]=0;ja[:]=0
                    if args.ground_root_weight:
                        current=Rotation.from_quat(q[3:7][[1,2,3,0]]);desired=Rotation.from_quat(r['qpos'][3:7][[1,2,3,0]])
                        error=current.inv().apply((desired*current.inv()).as_rotvec())
                        target_v=current.inv().apply(desired.apply(r['qvel'][3:6]));target_a=current.inv().apply(desired.apply(r['qacc'][3:6]))
                        controller.root_acceleration_ref=target_a+80*error+20*(target_v-v[3:6])
            else:
                if hasattr(controller,'root_acceleration_ref'):del controller.root_acceleration_ref
                controller.stabilize_support=False;controller.foot_geometry_floor=None
                anchor=release if release is not None else airborne
                if airborne is None and t-anchor['time']>.2:raise RuntimeError('Commanded release did not produce actual flight')
                phase='release' if airborne is None else 'flight';u=t-anchor['time'];ct=anchor['com']+anchor['velocity']*u+[0,0,-.5*9.81*u*u]
                cvt=anchor['velocity']+np.array([0,0,-9.81*u]);cat=np.array([0,0,-9.81]);ht=anchor['momentum']
                qref[7:],jv,ja=sample(flight_curve,min(u,fold_duration))
                if u>=fold_duration:jv[:]=0;ja[:]=0
                if unfold is None and (rotation_sum>4.2 or u>.5):
                    unfold=t;unfold_curve=curve(q[7:],land[7:],.2)
                if unfold is not None:
                    phase='unfold';u_land=min(t-unfold,.2);qref[7:],jv,ja=sample(unfold_curve,u_land)
                    if t-unfold>=.2:jv[:]=0;ja[:]=0
                if landed is not None:
                    controller.stabilize_support=True
                    phase='landing';ct=landed['com'].copy();ct[2]=com0[2];cvt=np.zeros(3);cat=np.zeros(3);ht=np.zeros(3)
            controller.joint_task_weight=args.joint_task_weight
            if args.maximum_push is not None:
                factor=float(np.clip((t-delay-momentum_start)/.05,0,1)) if args.momentum_delay_fraction else 1.
                controller.momentum_task_weight=np.array([args.side_momentum_weight,args.pitch_momentum_weight*(factor if phase=='push' else 1.),args.side_momentum_weight])
            support_guard=phase=='push'
            controller.minimum_foot_pitch_rate=0. if args.hold_toe_pitch and support_guard else None
            controller.minimum_joint_velocities={swing_dof:min(args.backswing_min_speed,max(0.,float(jv[swing_dof-6])))} if args.require_backswing and support_guard else {}
            if args.forward_seed:controller.servo_seed=previous_control.copy()
            tau,solution=controller.solve(q,v,com_target=ct,com_velocity=cvt,com_acceleration=cat,
                momentum_target=ht,momentum_rate=hr,qref=qref,joint_velocity=jv,joint_acceleration=ja)
            solutions.append({'time':t,'phase':phase,**solution})
            target=q[7:]+(tau+kd*v[6:])/kp;prediction=None;filter_report=None
            if command_filter is not None:
                target,prediction,filter_report=command_filter.choose(q,v,target,previous_control,env.sim.data.qacc_warmstart[0].cpu().numpy().copy(),
                    initial_foot_pitch-.015 if args.hold_toe_pitch and support_guard else None,
                    float(q0[swing_qpos])-.015 if args.require_backswing and support_guard else None)
            previous_control=target.copy();env.sim.data.ctrl[0]=torch.tensor(target,dtype=env.sim.data.ctrl.dtype,device=args.device)
            result=original_step(*a,**kw)
            qnext=env.sim.data.qpos[0].cpu().numpy().copy();vnext=env.sim.data.qvel[0].cpu().numpy().copy()
            if filter_report is not None:
                filter_report.update(time_s=t,next_qpos_prediction_error=float(abs(qnext-prediction[:m.nq]).max()),
                    next_qvel_prediction_error=float(abs(vnext-prediction[m.nq:]).max()))
                filter_reports.append(filter_report)
            rotation=Rotation.from_quat(qnext[3:7][[1,2,3,0]]);rotation_sum+=(rotation*last_rotation.inv()).as_rotvec()[1];last_rotation=rotation
            values=observe(qnext,vnext);actual_tau=env.sim.data.actuator_force[0].cpu().numpy()
            traces.append(np.r_[qnext,vnext]);controls.append(target);phases.append(phase)
            accelerations.append([getattr(controller,'last_explicit_acceleration',controller.previous*20),env.sim.data.qacc[0].cpu().numpy().copy()])
            velocity_accelerations.append([controller.previous*20,(vnext-v)/dt])
            metrics_trace.append(np.r_[values,np.maximum(np.maximum(lo-qnext[7:],qnext[7:]-hi),0).max(),
                (abs(actual_tau)/effort).max(),(abs(vnext[6:])/velocity).max(),rotation_sum])
            return result
        env.sim.step=checked_step;env.termination_manager.compute=lambda:torch.zeros(1,dtype=torch.bool,device=args.device)
        for _ in range(round(args.seconds/env.step_dt)):env.step(torch.zeros((1,len(names)),device=args.device))
        status='completed'
    except Exception as exc:error=str(exc)
    finally:
        if traces:np.savez(args.out/'trajectory.npz',state=np.asarray(traces),metrics=np.asarray(metrics_trace),
            servo_target=np.asarray(controls),phase=np.asarray(phases),predicted_and_actual_acceleration=np.asarray(accelerations),
            predicted_and_observed_velocity_acceleration=np.asarray(velocity_accelerations),fps=np.array([200.]))
        report={'status':status,'error':error,'elapsed_s':time.monotonic()-started,'physical_steps':len(traces),
            'approved_for_training':False,'configuration':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            'pid':os.getpid(),'source_sha256':hashlib.sha256(program_source.encode()).hexdigest(),'module_sha256':snapshots,
            'state_edits_after_start':False,'solutions':solutions}
        report['command_filter_reports']=filter_reports
        report['entry_initial_condition']=locals().get('entry')
        if traces:
            x=np.asarray(metrics_trace);report['maxima']={'hard_violation':float(x[:,12].max()),'effort_ratio':float(x[:,13].max()),
                'speed_ratio':float(x[:,14].max()),'minimum_nonfoot_height':float(x[:,9].min()),'minimum_left_height':float(x[:,7].min()),
                'minimum_right_height':float(x[:,8].min()),'rotation':float(x[-1,15]),'max_vz':float(x[:,1].max()),'max_Hy':float(x[:,3].max())}
            report['tail']={'up':float(x[-40:,11].min()),'speed':float(x[-40:,5].max()),'com_height':float(x[-40:,0].min())}
        for name in ['airborne','release','landed']:
            value=locals().get(name)
            report[name]=None if value is None else {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in value.items()}
        if controller is not None and hasattr(controller,'failure'):
            report['failed_constraint_audit']=controller.failure;np.savez(args.out/'failed-qp.npz',**controller.failure_arrays)
        (args.out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='solutions'}),flush=True)
        if command_filter is not None:
            if hasattr(command_filter,'failure'):(args.out/'command-filter-failure.json').write_text(json.dumps(command_filter.failure,indent=2))
            command_filter.close()
        env.close()
    if status!='completed':raise SystemExit(1)


if __name__=='__main__':main()
