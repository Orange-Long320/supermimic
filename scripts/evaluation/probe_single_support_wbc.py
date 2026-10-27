"""Closed-loop single-support hold using bounded whole-body inverse dynamics."""
import argparse
import hashlib
import inspect
import json
import os
import sys
import time
from types import SimpleNamespace
from pathlib import Path
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--initial-foot-height',type=float,help='Initial support-sole height only; no state edits after the first step')
    p.add_argument('--equilibrium',type=Path,help='Use an independently calibrated initial pose, before physics starts')
    p.add_argument('--integration-aware',action='store_true')
    p.add_argument('--native-inverse',action='store_true',help='Use the native compliant-contact inverse instead of a rigid support approximation')
    p.add_argument('--com-z-offset',type=float,default=0.,help='Smooth COM lift over 0.2 to 0.8 seconds')
    p.add_argument('--physics-feedback',action='store_true',help='Recompute commands at each native physics step')
    p.add_argument('--waist-posture-weight',type=float,default=.01)
    p.add_argument('--support-reference',type=Path,help='Coordinated native-model kinematic reference')
    p.add_argument('--reference-joint-weight',type=float,default=.1)
    p.add_argument('--foot-task-weight',type=float,default=.05)
    p.add_argument('--reference-root-weight',type=float,default=0.)
    p.add_argument('--vertical-demand',type=float,help='Diagnostic desired upward acceleration; unchanged native motors')
    p.add_argument('--lock-root-orientation',action='store_true')
    p.add_argument('--foot-floor',type=float)
    p.add_argument('--pitch-momentum',type=float,default=0.)
    p.add_argument('--pitch-weight',type=float,default=0.)
    p.add_argument('--free-joint-weight',type=float,default=.02)
    p.add_argument('--root-weight',type=float,default=1.)
    p.add_argument('--swing-hip',type=float)
    p.add_argument('--bounded-release',action='store_true')
    p.add_argument('--entry-vx',type=float,default=0.)
    p.add_argument('--entry-vz',type=float,default=0.)
    p.add_argument('--entry-hy',type=float,default=0.)
    p.add_argument('--free-knee-weight',type=float)
    p.add_argument('--release-on-knee',type=float,default=.12)
    p.add_argument('--release-after',type=float,default=.4)
    p.add_argument('--seconds',type=float,default=1.);args=p.parse_args()
    if args.integration_aware and not args.native_inverse:p.error('Integration-aware control requires native inverse dynamics')
    args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    program_source=Path(__file__).read_bytes();(args.out/'probe-source.py').write_bytes(program_source)
    configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    configuration.update(pid=os.getpid(),interpreter=sys.executable,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    for relative in ['src/supermimic/policy/joint_prediction.py','src/supermimic/policy/whole_body.py','src/supermimic/data/design_metrics.py','src/supermimic/data/entry_state.py']:
        (args.out/Path(relative).name).write_bytes(Path(relative).read_bytes())
    (args.out/'configuration.json').write_text(json.dumps(configuration,indent=2))
    from supermimic.envs import SuperMimicEnvCfgArgs,make_env
    from supermimic.envs.state_io import StateIO
    from supermimic.policy.whole_body import SingleSupportController
    from supermimic.data.design_metrics import BatchedDesignMetrics
    from supermimic.robots.dimensions import PROFILE
    env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(args.source)],num_envs=1,action_noise_std=0.,
        obs_noise=False,reset_noise=False,domain_randomization=False,sampling_mode='start',
        record_transitions=False,episode_length_s=1e5),'cpu')
    states=[];measurements=[];reactions=[];controls=[];solutions=[];accelerations=[];velocity_accelerations=[];controller=None
    status='failed';error=None;started=time.monotonic()
    try:
        cmd=env.command_manager.get_term('motion');cmd.forced_frame[:]=0;env.reset()
        io=StateIO(env);state=io.read();m=env.sim.mj_model;state[:,m.nq:]=0
        if args.equilibrium:
            with np.load(args.equilibrium) as equilibrium:
                if equilibrium['joint_names'].tolist()!=list(PROFILE['joint_names']):raise ValueError('Equilibrium joint order differs')
                state[0,:m.nq]=torch.as_tensor(equilibrium['qpos'],dtype=state.dtype)
        entry_audit=None
        if any([args.entry_vx,args.entry_vz,args.entry_hy]):
            from supermimic.data.entry_state import entry_velocity_state
            limits=np.array([PROFILE['velocity_limit'][n] for n in PROFILE['joint_names']])
            incoming,entry_audit=entry_velocity_state(m,state[0,:m.nq].numpy(),np.zeros(m.nv),args.entry_vx,limits,
                com_velocity=[args.entry_vx,0.,args.entry_vz],angular_momentum=[0.,args.entry_hy,0.])
            state[0,m.nq:]=torch.tensor(incoming,dtype=state.dtype)
        io.write(torch.tensor([0]),state)
        metrics=BatchedDesignMetrics(m,torch.device('cpu'))
        observation=mujoco.MjData(m)
        def observed_metrics(q,v):
            # mj_step caches body/geometry fields before the final integration.
            # Independent FK aligns every metric with the recorded post-step state.
            observation.qpos[:]=q;observation.qvel[:]=v;mujoco.mj_forward(m,observation)
            arrays={key:torch.as_tensor(np.array(getattr(observation,key)),dtype=torch.float32)[None]
                    for key in ['xpos','xquat','cvel','geom_xpos','geom_xmat','qvel']}
            return metrics(SimpleNamespace(**arrays))[0].numpy()

        original_foot_height=float(metrics(env.sim.data)[0,7])
        if args.initial_foot_height is not None:
            state[:,2]+=args.initial_foot_height-original_foot_height
            io.write(torch.tensor([0]),state)
        mujoco.mj_saveModel(m,str(args.out/'native-model.mjb'))
        np.savez(args.out/'initial-state.npz',state=state.numpy())
        names=list(PROFILE['joint_names']);kp=np.array([PROFILE['stiffness'][n] for n in names])
        kd=np.array([PROFILE['damping'][n] for n in names]);effort=np.array([PROFILE['effort_limit'][n] for n in names])
        velocity=np.array([PROFILE['velocity_limit'][n] for n in names])
        default=np.array([PROFILE['default_joint_pos'][n] for n in names]);scale=PROFILE['action_scale']
        scale=np.array([scale[n] for n in names]) if isinstance(scale,dict) else scale
        foot=next(i for i in range(m.nbody) if m.body(i).name.endswith('/left_ankle_roll_link'))
        controller_class=SingleSupportController
        if args.native_inverse:
            from supermimic.policy.native_inverse import NativeContactController
            controller_class=NativeContactController
        (args.out/'controller-source.py').write_text(inspect.getsource(sys.modules[controller_class.__module__]))
        controller=controller_class(m,state[0,:m.nq].numpy(),effort,velocity,foot)
        controller.integration_aware=args.integration_aware
        controller.foot_task_weight=args.foot_task_weight
        if args.native_inverse:
            weights=np.full(len(names),.01)
            for i,name in enumerate(names):
                if name.startswith('waist_'):weights[i]=args.waist_posture_weight
            controller.joint_task_weight=weights
        if args.vertical_demand is not None:
            if not args.native_inverse or args.support_reference:raise ValueError('Vertical diagnostic requires native inverse without a guide')
            controller.com_task_weight=np.array([2.,2.,2.])
            controller.momentum_task_weight=np.array([.1,args.pitch_weight,.1])
            controller.joint_task_weight=np.array([.02 if n.startswith('left_') else args.free_joint_weight for n in names])
            if args.free_knee_weight is not None:controller.joint_task_weight[names.index('right_knee_joint')]=args.free_knee_weight
            controller.root_acceleration_weight=args.root_weight
            controller.stabilize_support=False
            controller.foot_geometry_floor=args.foot_floor
            controller.root_acceleration_hard=args.lock_root_orientation
        initial_rotation=Rotation.from_quat(state[0,3:7].numpy()[[1,2,3,0]])
        release_time=None
        reference=None
        if args.support_reference:
            if not args.native_inverse:raise ValueError('Coordinated reference requires native inverse dynamics')
            from supermimic.kinematics.support_reference import SupportReference
            reference=SupportReference(args.support_reference)
            np.testing.assert_allclose(reference.arrays['qpos'][0],state[0,:m.nq].numpy(),atol=1e-6)
            controller.joint_task_weight=np.full(len(names),args.reference_joint_weight)
        if args.equilibrium:
            with np.load(args.equilibrium) as equilibrium:tau0=equilibrium['required_motor_torque']
            target=state[0,7:m.nq].numpy()+tau0/kp
            action=torch.tensor((target-default)/scale,dtype=torch.float32)[None]
            env.action_manager.process_action(action);env.action_manager.apply_action();env.scene.write_data_to_sim();env.sim.forward()
            initial_forward={'maximum_joint_acceleration':float(abs(env.sim.data.qacc[0,6:]).max()),
                'root_acceleration':env.sim.data.qacc[0,:6].numpy().tolist(),
                'ground_reaction_world':env.sim.data.qfrc_constraint[0,:3].numpy().tolist()}
        low,high=m.jnt_range[1:].T
        env.termination_manager.compute=lambda:torch.zeros(1,dtype=torch.bool)
        original_step=env.sim.step
        def command(time_s):
            nonlocal release_time
            q=env.sim.data.qpos[0].numpy().copy();v=env.sim.data.qvel[0].numpy().copy()
            if args.vertical_demand is not None:
                knee=names.index('left_knee_joint') if 'left_knee_joint' in names else next(i for i,n in enumerate(names) if n.startswith('left_knee'))
                # Only commands change after the initial state; release is a motor retraction command.
                if release_time is None and (q[7+knee]<args.release_on_knee or time_s>=args.release_after):release_time=time_s
                controller.data.qpos[:]=q;controller.data.qvel[:]=v;mujoco.mj_forward(m,controller.data)
                _,com_jac,h_jac=controller.jacobians(controller.data)
                if release_time is not None:
                    target=controller.qref[7:].copy()
                    for i,n in enumerate(names):
                        if n.startswith('left_hip_pitch'):target[i]=-.8
                        elif n.startswith('left_knee'):target[i]=1.8
                        elif n.startswith('right_knee'):target[i]=2.2
                        elif 'elbow' in n:target[i]=-1.4
                    if args.bounded_release:
                        controller.root_acceleration_hard=False;controller.foot_geometry_floor=None
                        if hasattr(controller,'root_acceleration_ref'):del controller.root_acceleration_ref
                        qref=q.copy();qref[7:]=target
                        tau,report=controller.solve(q,v,com_target=controller.data.subtree_com[controller.root].copy(),
                            com_velocity=com_jac@v,com_acceleration=np.array([0.,0.,-9.81]),
                            momentum_target=h_jac@v,qref=qref)
                        solutions.append({'time':time_s,'phase':'bounded_retraction',**report})
                        return q[7:]+(tau+kd*v[6:])/kp
                    solutions.append({'time':time_s,'phase':'retraction'})
                    return target
                current=Rotation.from_quat(q[3:7][[1,2,3,0]])
                controller.root_acceleration_ref=60*(current.inv()*initial_rotation).as_rotvec()-15*v[3:6]
                target_v=np.array([0.,0.,(com_jac@v)[2]])
                posture=controller.qref.copy()
                if args.swing_hip is not None:
                    u=np.clip(time_s/.15,0,1);u=u*u*(3-2*u)
                    i=7+names.index('right_hip_pitch_joint');posture[i]=(1-u)*posture[i]+u*args.swing_hip
                tau,report=controller.solve(q,v,com_target=controller.com_ref,com_velocity=target_v,
                    com_acceleration=np.array([0.,0.,args.vertical_demand]),momentum_target=np.array([0.,args.pitch_momentum,0.]),qref=posture)
                solutions.append({'time':time_s,'phase':'vertical_push',**report})
                return q[7:]+(tau+kd*v[6:])/kp
            if reference is not None:
                r=reference.sample(time_s-.2)
                controller.foot_position=r['foot_position'];controller.foot_rotation=Rotation.from_quat(r['foot_quat'])
                controller.foot_velocity_ref=r['foot_velocity'];controller.foot_acceleration_ref=r['foot_acceleration']
                if args.reference_root_weight:
                    current=Rotation.from_quat(q[3:7][[1,2,3,0]])
                    desired=Rotation.from_quat(r['qpos'][3:7][[1,2,3,0]])
                    error=current.inv().apply((desired*current.inv()).as_rotvec())
                    target_v=current.inv().apply(desired.apply(r['qvel'][3:6]))
                    target_a=current.inv().apply(desired.apply(r['qacc'][3:6]))
                    controller.root_acceleration_ref=target_a+80*error+20*(target_v-v[3:6])
                    controller.root_acceleration_weight=args.reference_root_weight
                tau,report=controller.solve(q,v,com_target=r['com'],com_velocity=r['com_velocity'],
                    com_acceleration=r['com_acceleration'],momentum_target=r['momentum'],momentum_rate=r['momentum_rate'],
                    qref=r['qpos'],joint_velocity=r['qvel'][6:],joint_acceleration=r['qacc'][6:])
                solutions.append({'time':time_s,**report})
                return q[7:]+(tau+kd*v[6:])/kp
            u=float(np.clip((time_s-.2)/.6,0,1));offset=args.com_z_offset
            position=offset*u**3*(10-15*u+6*u*u)
            speed=offset*30*u*u*(1-u)**2/.6
            acceleration=offset*60*u*(1-3*u+2*u*u)/.6**2
            tau,report=controller.solve(q,v,com_target=controller.com_ref+[0,0,position],
                com_velocity=np.array([0,0,speed]),com_acceleration=np.array([0,0,acceleration]))
            solutions.append({'time':time_s,**report})
            return q[7:]+(tau+kd*v[6:])/kp
        def capture(*a,**kw):
            velocity_before=env.sim.data.qvel[0].numpy().copy()
            if args.physics_feedback:
                target=command(len(states)*float(m.opt.timestep))
                env.sim.data.ctrl[0]=torch.tensor(target,dtype=env.sim.data.ctrl.dtype)
            result=original_step(*a,**kw)
            predicted=np.full(m.nv,np.nan) if release_time is not None and not args.bounded_release else controller.previous[:m.nv]*20.
            explicit=getattr(controller,'last_explicit_acceleration',predicted)
            accelerations.append(np.stack([explicit,env.sim.data.qacc[0].numpy().copy()]))
            velocity_accelerations.append(np.stack([predicted,(env.sim.data.qvel[0].numpy()-velocity_before)/float(m.opt.timestep)]))
            q=env.sim.data.qpos[0].numpy().copy();v=env.sim.data.qvel[0].numpy().copy()
            tau=env.sim.data.actuator_force[0].numpy().copy()
            states.append(np.r_[q,v]);measurements.append(np.r_[observed_metrics(q,v),
                np.maximum(np.maximum(low-q[7:],q[7:]-high),0).max(),(abs(tau)/effort).max()])
            reactions.append(env.sim.data.qfrc_constraint[0].numpy().copy())
            controls.append(env.sim.data.ctrl[0].numpy().copy())
            return result
        env.sim.step=capture
        for step in range(round(args.seconds/env.step_dt)):
            target=state[0,7:m.nq].numpy() if args.physics_feedback else command(step*env.step_dt)
            action=torch.tensor((target-default)/scale,dtype=torch.float32)[None]
            env.step(action)
        status='completed'
    except Exception as exc:error=str(exc)
    finally:
        if states:np.savez(args.out/'trajectory.npz',state=np.array(states),metrics=np.array(measurements),
            generalized_constraint_force=np.array(reactions),servo_target=np.array(controls),
            predicted_and_actual_acceleration=np.array(accelerations),
            predicted_and_observed_velocity_acceleration=np.array(velocity_accelerations),fps=np.array([200.]))
        a=np.array(measurements)
        report={'status':status,'error':error,'elapsed_s':time.monotonic()-started,'physical_steps':len(states),
            'source':str(args.source),'approved_for_training':False,'purpose':__doc__,'solutions':solutions}
        report['entry_audit']=entry_audit if 'entry_audit' in locals() else None
        report['release_time_s']=release_time if 'release_time' in locals() else None
        report['configuration']=configuration;report['source_sha256']=hashlib.sha256(program_source).hexdigest()
        report['integration_aware']=args.integration_aware
        report['metric_state_alignment']='Independent native FK from each recorded post-step qpos/qvel'
        report['controller']='native_contact_inverse' if args.native_inverse else 'rigid_contact_inverse'
        report['com_z_offset_m']=args.com_z_offset
        report['feedback_interval_s']=float(m.opt.timestep) if args.physics_feedback else env.step_dt
        report['waist_posture_weight']=args.waist_posture_weight
        report['support_reference']=str(args.support_reference) if args.support_reference else None
        report['reference_joint_weight']=args.reference_joint_weight if args.support_reference else None
        report['foot_task_weight']=args.foot_task_weight
        report['reference_root_weight']=args.reference_root_weight
        if 'original_foot_height' in locals():report['initial_condition']={'original_support_sole_height_m':original_foot_height,
            'requested_support_sole_height_m':args.initial_foot_height,'equilibrium':str(args.equilibrium) if args.equilibrium else None,'state_edits_after_start':False}
        if 'initial_forward' in locals():report['initial_native_forward']=initial_forward
        if controller is not None and hasattr(controller,'failure'):
            report['failed_constraint_audit']=controller.failure
            np.savez(args.out/'failed-qp.npz',**controller.failure_arrays)
        if len(a):report['maxima']={'hard_violation_rad':float(a[:,12].max()),'effort_ratio':float(a[:,13].max()),
            'minimum_nonfoot_height_m':float(a[:,9].min()),'minimum_left_foot_height_m':float(a[:,7].min()),
            'minimum_right_foot_height_m':float(a[:,8].min()),'tail_min_root_up_z':float(a[-40:,11].min()),
            'tail_max_com_speed':float(a[-40:,5].max()),'tail_min_com_height':float(a[-40:,0].min())}
        (args.out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='solutions'}),flush=True)
        env.close()
    if status!='completed':raise SystemExit(1)


if __name__=='__main__':main()
