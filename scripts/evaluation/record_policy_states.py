"""Capture a frozen policy's real physics from a selected reference frame.

Stops at the first original termination; never includes an automatic reset.
No reference retiming, residual actions, or physics changes are applied.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--start-frame',type=int,default=0)
    p.add_argument('--seconds',type=float,default=4.)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    if args.seconds<=0 or args.start_frame<0:p.error('Invalid duration or start frame')
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    from supermimic.envs import SuperMimicEnvCfgArgs,make_env
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.data.design_metrics import BatchedDesignMetrics
    from supermimic.robots.dimensions import PROFILE
    checkpoint_stat=args.ckpt.stat()
    checkpoint_sha=hashlib.sha256(args.ckpt.read_bytes()).hexdigest()
    policy,cfg=load_policy_from_checkpoint(args.ckpt,args.device)
    loaded_stat=args.ckpt.stat()
    assert all(getattr(loaded_stat,k)==getattr(checkpoint_stat,k)
               for k in ('st_dev','st_ino','st_size','st_mtime_ns')),'Checkpoint changed while loading'
    env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(args.source)],num_envs=1,
        action_noise_std=0.,obs_noise=False,reset_noise=False,domain_randomization=False,
        sampling_mode='start',record_transitions=False,episode_length_s=1e5,
        anchor_obs=cfg['env'].get('anchor_obs',False)),args.device)
    try:
        cmd=env.command_manager.get_term('motion')
        assert args.start_frame<int(cmd.library.clip_last[0]-cmd.library.clip_start[0])
        cmd.forced_frame[:]=cmd.library.clip_start[0]+args.start_frame
        obs,_=env.reset()
        original_compute=env.termination_manager.compute
        failures=[]
        def stop_without_reset():
            done=original_compute()
            if bool(done.any()):failures.append(len(physics))
            return torch.zeros_like(done)
        env.termination_manager.compute=stop_without_reset
        model=env.sim.mj_model
        metrics=BatchedDesignMetrics(model,torch.device(args.device))
        centre=model.jnt_range[1:].mean(axis=1)
        half=.45*np.diff(model.jnt_range[1:],axis=1).ravel()
        lower,upper=centre-half,centre+half
        hard_lower,hard_upper=model.jnt_range[1:].T
        assert [model.joint(i).name.rsplit('/',1)[-1] for i in range(1,model.njnt)]==list(PROFILE['joint_names'])
        effort=np.array([PROFILE['effort_limit'][n] for n in PROFILE['joint_names']])
        speed=np.array([PROFILE['velocity_limit'][n] for n in PROFILE['joint_names']])
        physics=[];positions=[];velocities=[];actions=[]
        origin=env.scene.env_origins[0].cpu().numpy()
        original_step=env.sim.step
        def capture(*a,**kw):
            result=original_step(*a,**kw)
            q=env.sim.data.qpos[0].cpu().numpy().copy();q[:3]-=origin
            v=env.sim.data.qvel[0].cpu().numpy().copy()
            f=env.sim.data.actuator_force[0].cpu().numpy().copy()
            positions.append(q);velocities.append(v)
            physics.append(np.r_[metrics(env.sim.data)[0].cpu().numpy(),
                np.maximum(np.maximum(lower-q[7:],q[7:]-upper),0).max(),
                (np.abs(v[6:])/speed).max(),(np.abs(f)/effort).max(),
                np.maximum(np.maximum(hard_lower-q[7:],q[7:]-hard_upper),0).max()])
            return result
        env.sim.step=capture
        with torch.no_grad():
            for _ in range(round(args.seconds/env.step_dt)):
                action=policy(obs['policy'],obs['goal'])
                actions.append(action[0].cpu().numpy().copy())
                obs,*_=env.step(action)
                if failures:break
        np.savez(args.out/'trajectory.npz',qpos=np.array(positions),qvel=np.array(velocities),
            actions=np.array(actions),metrics=np.array(physics),physics_dt=np.array([model.opt.timestep]),
            action_dt=np.array([env.step_dt]))
        a=np.array(physics)
        report={'checkpoint_sha256':checkpoint_sha,'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
            'start_reference_frame':args.start_frame,'physics_steps':len(physics),'actions':len(actions),
            'failure_physics_steps':failures,'maximum_com_vz':float(a[:,1].max()),
            'maximum_forward_momentum':float(a[:,3].max()),'minimum_nonfoot_height':float(a[:,9].min()),
            'maximum_soft_violation':float(a[:,12].max()),'maximum_speed_ratio':float(a[:,13].max()),
            'maximum_effort_ratio':float(a[:,14].max()),'approved_for_training':False,
            'maximum_hard_violation':float(a[:,15].max()),'hard_joint_ranges_rad':model.jnt_range[1:].tolist(),
            'metric_columns':['com_z','com_vz','H_x','H_y','H_z','com_speed','root_angular_speed',
                'left_foot_min','right_foot_min','nonfoot_min','root_pitch','root_up_z',
                'soft_violation','speed_ratio','effort_ratio','hard_violation'],
            'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
        (args.out/'result.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
    finally:env.close()


if __name__=='__main__':main()
