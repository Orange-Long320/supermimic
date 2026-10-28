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
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists() or a.seconds<=0:p.error('New output and positive duration required')
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
    scale=np.array([PROFILE['action_scale'][n] for n in names],dtype=float)
    effort=np.array([PROFILE['effort_limit'][n] for n in names],dtype=float)
    lib=MotionLibrary([a.motion],tuple(PROFILE['tracked_bodies']),defaults,device='cpu')
    anchor_name=PROFILE['anchor_body'];anchor=m.body('robot/'+anchor_name).id;anchor_index=lib.body_names.index(anchor_name)
    anchor_obs=bool(cfg['env'].get('anchor_obs',False));future_frames=int(cfg['env'].get('future_frames',10))
    if policy.cfg.goal_dim!=future_frames*72+(9 if anchor_obs else 0):raise ValueError('Frozen policy goal interface differs')
    q,v=lib.ref_qpos_qvel(torch.tensor([0]));d.qpos[:]=q[0].numpy();d.qvel[:]=v[0].numpy()
    columns=m.jnt_qposadr[m.actuator_trnid[:,0]];vcols=m.jnt_dofadr[m.actuator_trnid[:,0]]
    if not np.array_equal(columns,np.arange(7,38)):raise ValueError('Actuator joint order differs')
    d.ctrl[:]=d.qpos[columns]-m.actuator_biasprm[:,2]*d.qvel[vcols]/m.actuator_gainprm[:,0]
    mujoco.mj_forward(m,d)
    a.out.mkdir(parents=True);(a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    (a.out/'launch.json').write_text(json.dumps({'pid':os.getpid(),'cpu_threads':1,'gpu':None,'configuration':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}},indent=2))
    states=[np.r_[d.qpos.copy(),d.qvel.copy()]];controls=[];actions=[];rows=[]
    previous=torch.zeros((1,31));xi=torch.zeros((1,XI_DIM));history=[];obs_trace=[];goal_trace=[]
    def observe():
        return analytic_observation(torch.tensor(np.r_[d.qpos,d.qvel][None],dtype=torch.float32),previous,xi,defaults)
    initial=observe();history=[initial.clone() for _ in range(policy.cfg.history)]
    status='completed';failure=None
    with torch.no_grad():
        for tick in range(round(a.seconds*50)):
            frame=torch.tensor([min(tick+1,lib.num_frames-1)])
            goal=lib.goal_window(frame,future_frames)
            if anchor_obs:
                rel=anchor_relative_obs(torch.tensor(d.xpos[anchor][None],dtype=torch.float32),
                    torch.tensor(d.xquat[anchor][None],dtype=torch.float32),lib.body_pos_w[frame,anchor_index],lib.body_quat_w[frame,anchor_index])
                goal=torch.cat([goal,rel],dim=-1)
            observation=flatten_history_term_major(torch.stack(history,dim=0))
            obs_trace.append(observation[0].numpy().copy());goal_trace.append(goal[0].numpy().copy())
            action=policy(observation,goal);target=defaults.numpy()+action[0].numpy()*scale
            if not np.isfinite(target).all():raise ValueError('Nonfinite policy command')
            actions.append(action[0].numpy().copy())
            for substep in range(4):
                mujoco.mj_forward(m,d);d.ctrl[:]=target;mujoco.mj_forward(m,d)
                controls.append(target.copy());mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
                states.append(np.r_[d.qpos.copy(),d.qvel.copy()])
                hard=float(np.maximum(np.maximum(m.jnt_range[1:,0]-d.qpos[7:],d.qpos[7:]-m.jnt_range[1:,1]),0.).max())
                up=float(d.xmat[int(m.jnt_bodyid[0])].reshape(3,3)[2,2])
                height_error=float(abs(d.qpos[2]-lib.root_pos_w[frame,2].item()))
                rows.append({'time_s':len(controls)*.005,'hard_violation_rad':hard,'motor_ratio':float((abs(d.actuator_force)/effort).max()),
                    'root_height_m':float(d.qpos[2]),'root_up_z':up,'reference_height_error_m':height_error})
                if hard>.001 or up<.3 or height_error>.25:
                    status='safety_stop';failure=rows[-1];break
            previous=action;history=history[1:]+[observe()]
            if tick%50==0:print(json.dumps(rows[-1]),flush=True)
            if failure:break
    np.savez_compressed(a.out/'trajectory.npz',state=np.array(states),controls=np.array(controls),dt=np.array(.005))
    np.savez_compressed(a.out/'policy-interface.npz',observation=np.array(obs_trace),goal=np.array(goal_trace),actions=np.array(actions),fps=np.array([50.]))
    report={'status':status,'failure':failure,'seconds_executed':len(controls)*.005,'state_resets_after_entry':0,
        'configuration':{'model':str(a.model),'motion':str(a.motion),'checkpoint':str(a.checkpoint)},
        'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),'checkpoint_sha256':checkpoint_sha,
        'reference_sha256':hashlib.sha256(a.motion.read_bytes()).hexdigest(),'maximum_hard_violation_rad':max(x['hard_violation_rad'] for x in rows),
        'maximum_motor_ratio':max(x['motor_ratio'] for x in rows),'last':rows[-1],'policy_evaluation':True,'policy_weight_updates':0,
        'terrain_observation':'Existing motion/anchor interface only; no new box features added to the frozen policy',
        'training_approved':False,'complete_climb_verified':False,'rows':rows,'scope':__doc__}
    (a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
