"""Execute an immutable existing X2 policy in native CPU MuJoCo terrain.

Reuses the project's analytic observation/history/goal interfaces. No policy
updates, external wrench or state resets occur after initialization. This is
a controller diagnostic and possible trajectory source, not training approval.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--motion',type=Path,required=True,help='Explicit 50 Hz FK motion, with original joint/body names')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--seconds',type=float,required=True)
    p.add_argument('--start-frame',type=int,default=0,help='Explicit 50 Hz source entry; a nonzero start is not automatically a complete climb')
    p.add_argument('--entry-velocity-mode',choices=['reference','zero'],default='reference',help='Initial state only; no subsequent state rewrites')
    p.add_argument('--joint-barrier-margin',type=float,help='Optional conservative position-command barrier inside unchanged mechanical ranges')
    p.add_argument('--joint-barrier-frequency',type=float,default=35.)
    p.add_argument('--joint-projection-source',type=Path,help='Explicit private project_joint_commands module for the optional servo guard')
    p.add_argument('--coupled-projection-dir',type=Path,help='Private coupled projection, native inverse and prediction snapshots; alternative to approximate guard')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists() or a.seconds<=0:p.error('New output and positive duration required')
    if a.start_frame<0:p.error('Source entry must be nonnegative')
    if a.coupled_projection_dir and a.joint_barrier_margin is not None:p.error('Select only one command projection')
    if a.joint_barrier_margin is not None and (a.joint_barrier_margin<=0 or a.joint_barrier_frequency<=0 or not a.joint_projection_source):p.error('Barrier requires positive bounds and an explicit private projection source')
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra',check_assets=True)
    import mujoco
    import numpy as np
    import torch
    from supermimic.robots.dimensions import PROFILE,XI_DIM
    from supermimic.envs.mdp.motion_library import MotionLibrary
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.world_model.heads import analytic_observation,flatten_history_term_major,anchor_relative_obs
    torch.set_num_threads(1)
    m=mujoco.MjModel.from_binary_path(str(a.model));d=mujoco.MjData(m)
    if (m.nq,m.nv,m.nu)!=(38,37,31) or abs(float(m.opt.timestep)-.005)>1e-12:raise ValueError('Expected verified native X2 at 200 Hz')
    names=[m.joint(i).name.removeprefix('robot/') for i in range(1,m.njnt)]
    if names!=PROFILE['joint_names']:raise ValueError('Unexpected native joint order')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Explicit CPU-only execution required')
    with np.load(a.motion,allow_pickle=False) as z:
        if float(z['fps'][0])!=50.:raise ValueError('Frozen controller requires an explicit 50 Hz reference')
        if list(z['joint_names'])!=names:raise ValueError('Motion joint names differ')
    checkpoint_before=a.checkpoint.stat();checkpoint_sha=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    policy,cfg=load_policy_from_checkpoint(a.checkpoint,'cpu');policy.requires_grad_(False)
    checkpoint_after=a.checkpoint.stat()
    if (checkpoint_before.st_ino,checkpoint_before.st_size,checkpoint_before.st_mtime_ns)!=(checkpoint_after.st_ino,checkpoint_after.st_size,checkpoint_after.st_mtime_ns):raise ValueError('Controller changed while loading')
    if cfg.get('robot')!='x2_ultra' or cfg['env'].get('box'):raise ValueError('Expected the existing non-box X2 motion controller')
    defaults=torch.tensor([PROFILE['default_joint_pos'][n] for n in names],dtype=torch.float32)
    scale=torch.tensor([PROFILE['action_scale'][n] for n in names],dtype=torch.float32)
    effort=np.array([PROFILE['effort_limit'][n] for n in names],dtype=float)
    lib=MotionLibrary([a.motion],tuple(PROFILE['tracked_bodies']),defaults,device='cpu')
    if a.start_frame>=lib.num_frames:raise ValueError('Source entry beyond the clip')
    anchor_name=PROFILE['anchor_body'];anchor=m.body('robot/'+anchor_name).id;anchor_index=lib.body_names.index(anchor_name)
    anchor_obs=bool(cfg['env'].get('anchor_obs',False));future_frames=int(cfg['env'].get('future_frames',10))
    if policy.cfg.goal_dim!=future_frames*72+(9 if anchor_obs else 0):raise ValueError('Frozen policy goal interface differs')
    q,v=lib.ref_qpos_qvel(torch.tensor([a.start_frame]));d.qpos[:]=q[0].numpy();d.qvel[:]=v[0].numpy()
    if a.entry_velocity_mode=='zero':d.qvel[:]=0.
    columns=m.jnt_qposadr[m.actuator_trnid[:,0]];vcols=m.jnt_dofadr[m.actuator_trnid[:,0]]
    if not np.array_equal(columns,np.arange(7,38)):raise ValueError('Actuator joint order differs')
    d.ctrl[:]=d.qpos[columns]-m.actuator_biasprm[:,2]*d.qvel[vcols]/m.actuator_gainprm[:,0]
    mujoco.mj_forward(m,d)
    a.out.mkdir(parents=True);(a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    (a.out/'launch.json').write_text(json.dumps({'pid':os.getpid(),'cpu_threads':1,'gpu':None,'configuration':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}},indent=2))
    states=[np.r_[d.qpos.copy(),d.qvel.copy()]];controls=[];actions=[];rows=[]
    projector=None;projection_counts=np.zeros(31,dtype=int);infeasible_counts=np.zeros(31,dtype=int)
    if a.joint_barrier_margin is not None:
        import importlib.util
        module_spec=importlib.util.spec_from_file_location('omni_joint_projection_snapshot',a.joint_projection_source)
        module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module);projector=module.project_joint_commands
        (a.out/'joint-projection-as-run.py').write_bytes(a.joint_projection_source.read_bytes())
        if np.any(np.diff(m.jnt_range[1:],axis=1).ravel()<=2*a.joint_barrier_margin):raise ValueError('Barrier margin leaves no joint interior')
        tensor=lambda x:torch.as_tensor(np.asarray(x).copy(),dtype=torch.float64)
        kp,kd,cap,lo,hi=[tensor(x) for x in [m.actuator_gainprm[:,0],-m.actuator_biasprm[:,2],effort,m.jnt_range[1:,0],m.jnt_range[1:,1]]]
        mass=np.zeros((m.nv,m.nv))
    coupled=None;coupled_rows=[];coupled_sources={}
    if a.coupled_projection_dir:
        import importlib.util
        private_path=a.coupled_projection_dir/'omni_coupled_projection.py'
        spec=importlib.util.spec_from_file_location('omni_private_coupled_projection',private_path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        speed=np.array([PROFILE['velocity_limit'][n] for n in names])
        coupled=module.CoupledProjection(m,d,effort,speed,a.coupled_projection_dir)
        for filename in ['omni_coupled_projection.py','native_inverse.py','joint_prediction.py']:
            source=(a.coupled_projection_dir/filename).read_bytes()
            (a.out/filename).write_bytes(source);coupled_sources[filename]=hashlib.sha256(source).hexdigest()
    previous=torch.zeros((1,31));xi=torch.zeros((1,XI_DIM));history=[];obs_trace=[];goal_trace=[]
    def observe():
        return analytic_observation(torch.tensor(np.r_[d.qpos,d.qvel][None],dtype=torch.float32),previous,xi,defaults)
    initial=observe();history=[initial.clone() for _ in range(policy.cfg.history)]
    status='completed';failure=None
    with torch.no_grad():
        for tick in range(round(a.seconds*50)):
            frame=torch.tensor([min(a.start_frame+tick+1,lib.num_frames-1)])
            goal=lib.goal_window(frame,future_frames)
            if anchor_obs:
                rel=anchor_relative_obs(torch.tensor(d.xpos[anchor][None],dtype=torch.float32),
                    torch.tensor(d.xquat[anchor][None],dtype=torch.float32),lib.body_pos_w[frame,anchor_index],lib.body_quat_w[frame,anchor_index])
                goal=torch.cat([goal,rel],dim=-1)
            observation=flatten_history_term_major(torch.stack(history,dim=0))
            obs_trace.append(observation[0].numpy().copy());goal_trace.append(goal[0].numpy().copy())
            action=policy(observation,goal);target=(defaults+action[0]*scale).numpy().astype(float)
            if not np.isfinite(target).all():raise ValueError('Nonfinite policy command')
            actions.append(action[0].numpy().copy())
            for substep in range(4):
                mujoco.mj_forward(m,d);executed=target
                if coupled is not None:
                    try:
                        executed,projection_report=coupled.project(d,target)
                        coupled_rows.append({'time_s':len(controls)*.005,**projection_report})
                    except (RuntimeError,ValueError) as error:
                        status='controller_rejected';failure={'time_s':len(controls)*.005,'reason':str(error),
                            'inverse':getattr(coupled.controller,'failure',None)}
                        break
                if projector is not None:
                    mujoco.mj_fullM(m,d,mass);effective=1/np.diag(np.linalg.inv(mass))[6:]
                    projected,changed,infeasible=projector(tensor(target),tensor(d.qpos[7:]),tensor(d.qvel[6:]),tensor(d.qacc[6:]),
                        tensor(d.actuator_force),kp,kd,cap,tensor(effective),lo,hi,a.joint_barrier_margin,a.joint_barrier_frequency)
                    executed=projected.numpy();projection_counts+=changed.numpy();infeasible_counts+=infeasible.numpy()
                d.ctrl[:]=executed;mujoco.mj_forward(m,d)
                controls.append(executed.copy());mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
                states.append(np.r_[d.qpos.copy(),d.qvel.copy()])
                hard=float(np.maximum(np.maximum(m.jnt_range[1:,0]-d.qpos[7:],d.qpos[7:]-m.jnt_range[1:,1]),0.).max())
                up=float(d.xmat[int(m.jnt_bodyid[0])].reshape(3,3)[2,2])
                height_error=float(abs(d.qpos[2]-lib.root_pos_w[frame,2].item()))
                rows.append({'time_s':len(controls)*.005,'hard_violation_rad':hard,'motor_ratio':float((abs(d.actuator_force)/effort).max()),
                    'root_height_m':float(d.qpos[2]),'root_up_z':up,'reference_height_error_m':height_error})
                if hard>.001 or up<.3 or height_error>.25:
                    status='safety_stop';failure=rows[-1];break
            previous=action;history=history[1:]+[observe()]
            if tick%50==0:print(json.dumps(rows[-1] if rows else failure),flush=True)
            if failure:break
    np.savez_compressed(a.out/'trajectory.npz',state=np.array(states),controls=np.asarray(controls).reshape(-1,31),dt=np.array(.005))
    np.savez_compressed(a.out/'policy-interface.npz',observation=np.array(obs_trace),goal=np.array(goal_trace),actions=np.array(actions),fps=np.array([50.]))
    report={'status':status,'failure':failure,'seconds_executed':len(controls)*.005,'state_resets_after_entry':0,
        'configuration':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),'checkpoint_sha256':checkpoint_sha,
        'reference_sha256':hashlib.sha256(a.motion.read_bytes()).hexdigest(),'maximum_hard_violation_rad':max((x['hard_violation_rad'] for x in rows),default=0.),
        'maximum_motor_ratio':max((x['motor_ratio'] for x in rows),default=0.),'last':rows[-1] if rows else None,'policy_evaluation':True,'policy_weight_updates':0,
        'terrain_observation':'Existing motion/anchor interface only; no new box features added to the frozen policy',
        'policy_alone':projector is None and coupled is None,'coupled_projection':{'sources':coupled_sources,'rows':coupled_rows,
            'scope':'Coupled native inverse command projection with predictive joint bounds; independent actual geometry still required.'},
        'joint_projection':{'margin':a.joint_barrier_margin,'frequency':a.joint_barrier_frequency,
            'interventions_per_joint':projection_counts.tolist(),'infeasible_predictions_per_joint':infeasible_counts.tolist(),
            'source_sha256':hashlib.sha256(a.joint_projection_source.read_bytes()).hexdigest() if projector else None,
            'scope':'Approximate acceleration barrier; only position commands change. Actual trajectory audit remains authoritative.'},
        'training_approved':False,'complete_climb_verified':False,'rows':rows,'scope':__doc__}
    (a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
