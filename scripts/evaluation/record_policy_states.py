"""Capture a frozen policy's real physics from a selected reference frame.

Stops at the first original termination; never includes an automatic reset.
No reference retiming or physics changes are applied. Optional recorded command
prefixes are explicit counterfactuals, never labelled as the policy alone.
Optional final-reference holding ignores only clip-end timeout, retaining all
physical failure checks. Geometry is measured in a separate native data object.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
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
    p.add_argument('--hold-last-reference',action='store_true',help='Ignore clip-end timeout and hold its last reference state; original physical failures still stop capture')
    p.add_argument('--record-observations',action='store_true')
    p.add_argument('--record-contact-forces',action='store_true',help='Record actual engine contact wrenches with the installed MuJoCo-Warp extraction API')
    p.add_argument('--clip-servo-targets',action='store_true',help='Diagnostic only: clamp proposed motor position targets to original hard joint ranges; not policy-alone execution')
    p.add_argument('--residual-commands',type=Path,help='Explicit 50 Hz additive action proposal; marks capture as hybrid control')
    p.add_argument('--prefix-trace',type=Path,help='Directory containing recorded 50 Hz actions and initial native state')
    p.add_argument('--prefix-until',type=float,default=0.,help='Keep recorded commands until this time, then blend back to the live policy')
    p.add_argument('--prefix-blend',type=float,default=.08)
    p.add_argument('--prefix-joints',nargs='+',help='Only replace these named joints; default is all joints')
    args=p.parse_args()
    if args.seconds<=0 or args.start_frame<0:p.error('Invalid duration or start frame')
    if args.prefix_trace and (args.prefix_until<=0 or args.prefix_blend<0):p.error('Prefix needs positive duration and nonnegative blending')
    if not args.prefix_trace and (args.prefix_until or args.prefix_joints):p.error('Prefix options require a recorded trace')
    if args.residual_commands and (args.prefix_trace or args.start_frame):p.error('Residual proposals require frame zero and no recorded prefix')
    if args.clip_servo_targets and (args.residual_commands or args.prefix_trace):p.error('Servo clipping diagnostic must be isolated from other command interventions')
    args.out.mkdir(parents=True,exist_ok=False)
    as_run_source=Path(__file__).read_bytes()
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
        anchor_obs=cfg['env'].get('anchor_obs',False),
        terrain_manifest=Path(cfg['env']['terrain_manifest']) if cfg['env'].get('terrain_manifest') else None),args.device)
    try:
        cmd=env.command_manager.get_term('motion')
        assert args.start_frame<int(cmd.library.clip_last[0]-cmd.library.clip_start[0])
        cmd.forced_frame[:]=cmd.library.clip_start[0]+args.start_frame
        obs,_=env.reset()
        original_compute=env.termination_manager.compute
        failures=[]
        def stop_without_reset():
            if args.hold_last_reference:cmd.clip_end.zero_()
            done=original_compute()
            if bool(done.any()):failures.append(len(physics))
            return torch.zeros_like(done)
        env.termination_manager.compute=stop_without_reset
        model=env.sim.mj_model
        mujoco.mj_saveModel(model,str(args.out/'native-model.mjb'))
        measured_data=mujoco.MjData(model)
        # Native reconstruction and its tensors live on CPU, independently of
        # the device executing the policy and physics.
        metrics=BatchedDesignMetrics(model,torch.device('cpu'))
        centre=model.jnt_range[1:].mean(axis=1)
        half=.45*np.diff(model.jnt_range[1:],axis=1).ravel()
        lower,upper=centre-half,centre+half
        hard_lower,hard_upper=model.jnt_range[1:].T
        target_scale=PROFILE['action_scale']
        target_scale=np.array([target_scale[n] for n in PROFILE['joint_names']]) if isinstance(target_scale,dict) else np.full(len(hard_lower),target_scale)
        if not np.isfinite(target_scale).all() or not (target_scale>0).all():raise ValueError('Expected positive finite action scales')
        target_default=np.array([PROFILE['default_joint_pos'][n] for n in PROFILE['joint_names']])
        clip_lower=torch.as_tensor((hard_lower-target_default)/target_scale,device=args.device)
        clip_upper=torch.as_tensor((hard_upper-target_default)/target_scale,device=args.device)
        clipped_commands=0
        assert [model.joint(i).name.rsplit('/',1)[-1] for i in range(1,model.njnt)]==list(PROFILE['joint_names'])
        effort=np.array([PROFILE['effort_limit'][n] for n in PROFILE['joint_names']])
        speed=np.array([PROFILE['velocity_limit'][n] for n in PROFILE['joint_names']])
        physics=[];positions=[];velocities=[];actions=[];forces=[];constraint_forces=[];external_force_max=[0.]
        origin=env.scene.env_origins[0].cpu().numpy()
        initial_q=env.sim.data.qpos[0].cpu().numpy().copy();initial_q[:3]-=origin
        initial_v=env.sim.data.qvel[0].cpu().numpy().copy()
        np.savez(args.out/'initial-state.npz',state=np.r_[initial_q,initial_v][None],qpos=initial_q,qvel=initial_v)
        prefix_actions=None;prefix_details=None;prefix_errors=[];observation_trace=[]
        residual_actions=None;residual_details=None
        contact_offsets=[0];contact_geoms=[];contact_wrenches=[];contact_points=[];contact_normals=[];contact_errors=[];contact_relative_errors=[]
        if args.record_contact_forces:
            import warp as wp
            import mujoco_warp as mjwarp
            raw_data=env.sim.wp_data
            contact_ids=wp.array(np.arange(raw_data.contact.geom.shape[0],dtype=np.int32),device=raw_data.qpos.device)
            contact_output=wp.zeros(contact_ids.size,dtype=wp.spatial_vector,device=raw_data.qpos.device)
            robot_root=int(model.jnt_bodyid[0])
            robot_geometry=np.array([model.body_rootid[model.geom_bodyid[g]]==robot_root for g in range(model.ngeom)])
            mjwarp.contact_force(env.sim.wp_model,raw_data,contact_ids,True,contact_output)
        if args.residual_commands:
            with np.load(args.residual_commands) as proposal:
                if str(proposal['source_checkpoint_sha256'])!=checkpoint_sha or str(proposal['reference_sha256'])!=hashlib.sha256(args.source.read_bytes()).hexdigest():
                    raise ValueError('Residual proposal uses a different policy/reference')
                residual_actions=proposal['actions'].copy()
                if list(proposal['joint_names'])!=list(PROFILE['joint_names']) or abs(float(proposal['fps'][0])-50)>1e-6:
                    raise ValueError('Residual joint order or control clock differs')
                if residual_actions.shape!=(round(args.seconds/env.step_dt),len(PROFILE['joint_names'])) or not np.isfinite(residual_actions).all():
                    raise ValueError('Residual must cover the complete capture with finite actions')
            residual_details={'path':str(args.residual_commands),'sha256':hashlib.sha256(args.residual_commands.read_bytes()).hexdigest(),
                'kind':'ordinary additive position action with live frozen-policy feedback; no state rewrite'}
        if args.prefix_trace:
            prefix_path=args.prefix_trace/'baseline-trajectory.npz'
            prefix=np.load(prefix_path);prefix_physics=np.load(args.prefix_trace/'baseline-physics.npz')
            if abs(float(prefix['fps'][0])-1/env.step_dt)>1e-6:raise ValueError('Prefix control frequency differs')
            prefix_actions=prefix['actions'].copy()
            if len(prefix_actions)*env.step_dt<args.prefix_until+args.prefix_blend:raise ValueError('Recorded prefix is too short')
            selected=args.prefix_joints or list(PROFILE['joint_names'])
            if len(set(selected))!=len(selected) or any(n not in PROFILE['joint_names'] for n in selected):raise ValueError('Invalid prefix joint selection')
            prefix_indices=[PROFILE['joint_names'].index(n) for n in selected]
            initial_errors={'qpos':float(abs(prefix_physics['initial_qpos']-initial_q).max()),
                'qvel':float(abs(prefix_physics['initial_qvel']-initial_v).max())}
            if max(initial_errors.values())>1e-5:raise ValueError('Recorded and live initial states differ; no state rewrite is permitted')
            if hashlib.sha256((args.prefix_trace/'native-model.mjb').read_bytes()).hexdigest()!=hashlib.sha256((args.out/'native-model.mjb').read_bytes()).hexdigest():
                raise ValueError('Recorded prefix uses a different native model')
            prefix_details={'trace':str(args.prefix_trace),'actions_sha256':hashlib.sha256(prefix_path.read_bytes()).hexdigest(),
                'joint_names':selected,'until_s':args.prefix_until,'blend_s':args.prefix_blend,'initial_errors':initial_errors}
        original_step=env.sim.step
        def capture(*a,**kw):
            result=original_step(*a,**kw)
            q=env.sim.data.qpos[0].cpu().numpy().copy();q[:3]-=origin
            v=env.sim.data.qvel[0].cpu().numpy().copy()
            f=env.sim.data.actuator_force[0].cpu().numpy().copy()
            constraint_forces.append(env.sim.data.qfrc_constraint[0].cpu().numpy().copy())
            external_force_max[0]=max(external_force_max[0],float(env.sim.data.qfrc_applied[0].abs().max()),
                float(env.sim.data.xfrc_applied[0].abs().max()))
            positions.append(q);velocities.append(v);forces.append(f)
            if args.record_contact_forces:
                count=int(raw_data.nacon.numpy()[0])
                if not 0<=count<=contact_ids.size:raise RuntimeError('Invalid actual contact count')
                mjwarp.contact_force(env.sim.wp_model,raw_data,contact_ids,True,contact_output)
                wrench=contact_output.numpy()[:count].copy()
                geoms=raw_data.contact.geom.numpy()[:count].copy()
                points=raw_data.contact.pos.numpy()[:count].copy()-origin
                normals=raw_data.contact.frame.numpy()[:count,0,:].copy()
                if np.any(geoms<0) or np.any(geoms>=model.ngeom):raise RuntimeError('Unsupported flexible contact geometry')
                sign=robot_geometry[geoms[:,1]].astype(float)-robot_geometry[geoms[:,0]].astype(float)
                total=(wrench[:,:3]*sign[:,None]).sum(axis=0)
                error=float(abs(total-constraint_forces[-1][:3]).max())
                contact_errors.append(error)
                contact_relative_errors.append(error/max(1.,float(abs(total).max()),float(abs(constraint_forces[-1][:3]).max())))
                contact_offsets.append(contact_offsets[-1]+count)
                contact_geoms.append(geoms);contact_wrenches.append(wrench);contact_points.append(points);contact_normals.append(normals)
            # Use a separate native data object: Warp geometry caches precede
            # integration, and measurement must not refresh the executing sim.
            measured_data.qpos[:]=q;measured_data.qvel[:]=v;mujoco.mj_forward(model,measured_data)
            tensors={key:torch.as_tensor(np.array(getattr(measured_data,key)),dtype=torch.float32)[None]
                for key in ['xpos','xquat','cvel','geom_xpos','geom_xmat','qvel']}
            physics.append(np.r_[metrics(SimpleNamespace(**tensors))[0].cpu().numpy(),
                np.maximum(np.maximum(lower-q[7:],q[7:]-upper),0).max(),
                (np.abs(v[6:])/speed).max(),(np.abs(f)/effort).max(),
                np.maximum(np.maximum(hard_lower-q[7:],q[7:]-hard_upper),0).max()])
            return result
        env.sim.step=capture
        with torch.no_grad():
            for step in range(round(args.seconds/env.step_dt)):
                if args.record_observations:
                    observation_trace.append((obs['policy'][0].cpu().numpy().copy(),obs['goal'][0].cpu().numpy().copy(),int(cmd.frame[0])))
                action=policy(obs['policy'],obs['goal'])
                if residual_actions is not None:
                    action=action+torch.as_tensor(residual_actions[step],dtype=action.dtype,device=action.device)[None]
                t=step*env.step_dt
                if prefix_actions is not None and t<args.prefix_until+args.prefix_blend:
                    fraction=min(max((t-args.prefix_until)/args.prefix_blend,0),1) if args.prefix_blend else 0.
                    recorded=torch.as_tensor(prefix_actions[step,prefix_indices],dtype=action.dtype,device=action.device)
                    action[:,prefix_indices]=torch.lerp(recorded,action[:,prefix_indices],fraction)
                if args.clip_servo_targets:
                    proposed=action
                    action=torch.maximum(torch.minimum(action,clip_upper),clip_lower).to(proposed.dtype)
                    clipped_commands+=int(bool((action!=proposed).any()))
                actions.append(action[0].cpu().numpy().copy())
                obs,*_=env.step(action)
                if prefix_actions is not None and t<args.prefix_until+args.prefix_blend and t<=args.prefix_until and len(prefix_indices)==len(PROFILE['joint_names']):
                    prefix_errors.append(float(abs(positions[-1]-prefix['qpos'][step]).max()))
                if failures:break
        if observation_trace:
            np.savez(args.out/'baseline-observations.npz',policy=np.stack([r[0] for r in observation_trace]),
                goal=np.stack([r[1] for r in observation_trace]),reference_frame=[r[2] for r in observation_trace],
                initial_qpos=initial_q,initial_qvel=initial_v,fps=np.array([1/env.step_dt]))
        scale=PROFILE['action_scale']
        scale=np.array([scale[n] for n in PROFILE['joint_names']]) if isinstance(scale,dict) else scale
        defaults=np.array([PROFILE['default_joint_pos'][n] for n in PROFILE['joint_names']])
        ratio=round(env.step_dt/model.opt.timestep)
        np.savez(args.out/'trajectory.npz',qpos=np.array(positions),qvel=np.array(velocities),
            actions=np.array(actions),metrics=np.array(physics),physics_dt=np.array([model.opt.timestep]),
            action_dt=np.array([env.step_dt]),state=np.c_[positions,velocities],
            servo_target=np.repeat(defaults+np.array(actions)*scale,ratio,axis=0),fps=np.array([1/model.opt.timestep]))
        np.savez(args.out/'baseline-physics.npz',qpos=positions,qvel=velocities,actuator_force=forces,
            constraint_generalized_force=constraint_forces,
            initial_qpos=initial_q,initial_qvel=initial_v,fps=np.array([1/model.opt.timestep]))
        np.savez(args.out/'baseline-trajectory.npz',qpos=np.array(positions)[ratio-1::ratio],
            qvel=np.array(velocities)[ratio-1::ratio],actions=actions,fps=np.array([1/env.step_dt]))
        if args.record_contact_forces:
            np.savez_compressed(args.out/'contacts-actual.npz',offsets=np.array(contact_offsets,dtype=np.int64),
                geom=np.concatenate(contact_geoms),force_torque_on_geom2_world=np.concatenate(contact_wrenches),
                position_world=np.concatenate(contact_points),normal_geom1_to_geom2_world=np.concatenate(contact_normals),
                fps=np.array([1/model.opt.timestep]))
        a=np.array(physics)
        report={'checkpoint_sha256':checkpoint_sha,'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
            'start_reference_frame':args.start_frame,'physics_steps':len(physics),'actions':len(actions),
            'failure_physics_steps':failures,'maximum_com_vz':float(a[:,1].max()),
            'maximum_forward_momentum':float(a[:,3].max()),'minimum_nonfoot_height':float(a[:,9].min()),
            'maximum_soft_violation':float(a[:,12].max()),'maximum_speed_ratio':float(a[:,13].max()),
            'maximum_effort_ratio':float(a[:,14].max()),'approved_for_training':False,
            'maximum_hard_violation':float(a[:,15].max()),'hard_joint_ranges_rad':model.jnt_range[1:].tolist(),
            'policy_alone':prefix_details is None and residual_details is None and not args.clip_servo_targets,'recorded_prefix':prefix_details,
            'diagnostic_servo_target_clipping':args.clip_servo_targets,'clipped_command_steps':clipped_commands,
            'recorded_residual':residual_details,
            'prefix_reproduction_maximum_qpos_error':max(prefix_errors) if prefix_errors else None,
            'maximum_foot_compression_m':float(max(-a[:,7:9].min(),0)),
            'metric_state_alignment':'Independent native mj_forward of each post-integration recorded state; execution sim untouched',
            'external_applied_force_max':external_force_max[0],'state_resets_after_entry':0,
            'last_reference_frame':int(cmd.frame[0]-cmd.library.clip_start[0]),
            'reference_frame_count':int(cmd.library.clip_len[0]),
            'actual_contact_forces_recorded':args.record_contact_forces,
            'contact_force_root_translation_maximum_error_n':max(contact_errors) if contact_errors else None,
            'contact_force_root_translation_maximum_relative_error':max(contact_relative_errors) if contact_relative_errors else None,
            'contact_force_balance_verified':bool(contact_errors and all(e<=1e-3 or r<=1e-5 for e,r in zip(contact_errors,contact_relative_errors))),
            'contact_measurement_roundoff_tolerance':{'absolute_n':1e-3,'relative':1e-5,'scope':'Readout consistency only; physical force limits remain unchanged'},
            'contact_trace_sha256':hashlib.sha256((args.out/'contacts-actual.npz').read_bytes()).hexdigest() if args.record_contact_forces else None,
            'contact_force_state_alignment':'Engine preintegration contact solution; pose trajectory stores the following postintegration state',
            'metric_columns':['com_z','com_vz','H_x','H_y','H_z','com_speed','root_angular_speed',
                'left_foot_min','right_foot_min','nonfoot_min','root_pitch','root_up_z',
                'soft_violation','speed_ratio','effort_ratio','hard_violation'],
            'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
        (args.out/'result.json').write_text(json.dumps(report,indent=2))
        (args.out/'as-run.py').write_bytes(as_run_source)
        print(json.dumps(report),flush=True)
    finally:env.close()


if __name__=='__main__':main()
