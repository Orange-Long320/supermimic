"""Closed-loop single-support hold using bounded whole-body inverse dynamics."""
import argparse
import json
import time
from pathlib import Path
import mujoco
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--initial-foot-height',type=float,help='Initial support-sole height only; no state edits after the first step')
    p.add_argument('--equilibrium',type=Path,help='Use an independently calibrated initial pose, before physics starts')
    p.add_argument('--native-inverse',action='store_true',help='Use the native compliant-contact inverse instead of a rigid support approximation')
    p.add_argument('--com-z-offset',type=float,default=0.,help='Smooth COM lift over 0.2 to 0.8 seconds')
    p.add_argument('--physics-feedback',action='store_true',help='Recompute commands at each native physics step')
    p.add_argument('--waist-posture-weight',type=float,default=.01)
    p.add_argument('--seconds',type=float,default=1.);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    from supermimic.envs import SuperMimicEnvCfgArgs,make_env
    from supermimic.envs.state_io import StateIO
    from supermimic.policy.whole_body import SingleSupportController
    from supermimic.data.design_metrics import BatchedDesignMetrics
    from supermimic.robots.dimensions import PROFILE
    env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(args.source)],num_envs=1,action_noise_std=0.,
        obs_noise=False,reset_noise=False,domain_randomization=False,sampling_mode='start',
        record_transitions=False,episode_length_s=1e5),'cpu')
    states=[];measurements=[];reactions=[];controls=[];solutions=[];accelerations=[];controller=None
    status='failed';error=None;started=time.monotonic()
    try:
        cmd=env.command_manager.get_term('motion');cmd.forced_frame[:]=0;env.reset()
        io=StateIO(env);state=io.read();m=env.sim.mj_model;state[:,m.nq:]=0
        if args.equilibrium:
            with np.load(args.equilibrium) as equilibrium:
                if equilibrium['joint_names'].tolist()!=list(PROFILE['joint_names']):raise ValueError('Equilibrium joint order differs')
                state[0,:m.nq]=torch.as_tensor(equilibrium['qpos'],dtype=state.dtype)
        io.write(torch.tensor([0]),state)
        metrics=BatchedDesignMetrics(m,torch.device('cpu'))
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
        controller=controller_class(m,state[0,:m.nq].numpy(),effort,velocity,foot)
        if args.native_inverse:
            weights=np.full(len(names),.01)
            for i,name in enumerate(names):
                if name.startswith('waist_'):weights[i]=args.waist_posture_weight
            controller.joint_task_weight=weights
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
            q=env.sim.data.qpos[0].numpy().copy();v=env.sim.data.qvel[0].numpy().copy()
            u=float(np.clip((time_s-.2)/.6,0,1));offset=args.com_z_offset
            position=offset*u**3*(10-15*u+6*u*u)
            speed=offset*30*u*u*(1-u)**2/.6
            acceleration=offset*60*u*(1-3*u+2*u*u)/.6**2
            tau,report=controller.solve(q,v,com_target=controller.com_ref+[0,0,position],
                com_velocity=np.array([0,0,speed]),com_acceleration=np.array([0,0,acceleration]))
            solutions.append({'time':time_s,**report})
            return q[7:]+(tau+kd*v[6:])/kp
        def capture(*a,**kw):
            if args.physics_feedback:
                target=command(len(states)*float(m.opt.timestep))
                env.sim.data.ctrl[0]=torch.tensor(target,dtype=env.sim.data.ctrl.dtype)
            result=original_step(*a,**kw)
            accelerations.append(np.stack([controller.previous[:m.nv]*20.,env.sim.data.qacc[0].numpy().copy()]))
            q=env.sim.data.qpos[0].numpy().copy();v=env.sim.data.qvel[0].numpy().copy()
            tau=env.sim.data.actuator_force[0].numpy().copy()
            states.append(np.r_[q,v]);measurements.append(np.r_[metrics(env.sim.data)[0].numpy(),
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
            predicted_and_actual_acceleration=np.array(accelerations),fps=np.array([200.]))
        a=np.array(measurements)
        report={'status':status,'error':error,'elapsed_s':time.monotonic()-started,'physical_steps':len(states),
            'source':str(args.source),'approved_for_training':False,'purpose':__doc__,'solutions':solutions}
        report['controller']='native_contact_inverse' if args.native_inverse else 'rigid_contact_inverse'
        report['com_z_offset_m']=args.com_z_offset
        report['feedback_interval_s']=float(m.opt.timestep) if args.physics_feedback else env.step_dt
        report['waist_posture_weight']=args.waist_posture_weight
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
