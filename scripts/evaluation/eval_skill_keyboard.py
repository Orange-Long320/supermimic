"""Run scripted keyboard skills continuously in physics, without resets."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
    a=p.parse_args()
    import os
    if a.out.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(a.gpu):
        raise ValueError('New output directory and explicit GPU required')
    import mujoco,torch
    from scipy.spatial.transform import Rotation
    from mjlab.utils.lab_api.math import subtract_frame_transforms,matrix_from_quat
    from supermimic.control.skill_phase import SkillPhase
    from supermimic.envs import SuperMimicEnvCfgArgs,make_env
    from supermimic.envs.mdp.motion_library import MotionLibrary
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.robots.dimensions import NJ
    torch.set_num_threads(2)
    policy,cfg=load_policy_from_checkpoint(a.checkpoint,'cuda:0')
    names=['walk','backward','turn_left_inplace','turn_right_inplace','jog']
    files=[a.dataset/(n+'.npz') for n in names]+[a.dataset/(n+'_stop.npz') for n in names]
    meta=json.loads((a.dataset/'manifest.json').read_text())
    brake={c['name'].removesuffix('_stop'):c['release_frame']+10 for c in meta['clips'] if c['type']=='exploratory_stop'}
    if set(brake)!=set(names):raise ValueError('Five stop phases required')
    stand_file=a.dataset/'walk_stop.npz'
    with np.load(stand_file) as z:initial=len(z['joint_pos'])-1
    env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(stand_file)],num_envs=1,seed=0,
        future_frames=cfg['env']['future_frames'],episode_length_s=1000.,action_noise_std=0.,obs_noise=False,
        reset_noise=False,domain_randomization=False,record_transitions=False,sampling_mode='start',anchor_obs=True),device='cuda:0')
    env.cfg.auto_reset=False
    cmd=env.command_manager.get_term('motion');cmd.forced_frame[:]=initial
    lib=MotionLibrary(files,cmd.cfg.body_names,cmd.library.default_joint_pos,device='cuda:0')
    clips={c.name:lib.goal_feat[c.start:c.end].cpu().numpy() for c in lib.clips}
    scheduler=SkillPhase({n:clips[n] for n in names},{n:clips[n+'_stop'] for n in names},brake,NJ,horizon=cfg['env']['future_frames'])
    model=env.sim.mj_model;reference_data=mujoco.MjData(model);anchor_name=cmd.cfg.anchor_body_name
    anchor_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'robot/'+anchor_name)
    if anchor_id<0:raise ValueError('Anchor body not found')
    joint_default=cmd.library.default_joint_pos.cpu().numpy()
    schedule=[(0.,'stand'),(3.,'walk'),(7.,'stand'),(11.,'backward'),(15.,'stand'),(19.,'turn_left_inplace'),(23.,'stand'),(27.,'turn_right_inplace'),(31.,'stand'),(35.,'jog'),(39.,'stand')]
    a.out.mkdir(parents=True)
    try:
        obs,_=env.reset(); q0=env.sim.data.qpos[0].cpu().numpy(); origin=env.scene.env_origins[0].cpu().numpy()
        pos=q0[:3].copy()-origin; yaw=Rotation.from_quat(q0[[4,5,6,3]]).as_euler('xyz')[2]
        states=[]; goals=[]; modes=[];failures=[];overlap=[]; previous=None
        with torch.no_grad():
            for step in range(50*45):
                time=step*.02
                mode=next(v for t,v in reversed(schedule) if time>=t)
                scheduler.select(mode);window=scheduler.window(); g=window[0];grav=g[2*NJ:2*NJ+3]
                if previous is not None:overlap.append(float(np.max(np.abs(previous[1:]-window[:-1]))))
                previous=window.copy()
                roll=np.arctan2(-grav[1],-grav[2]);pitch=np.arcsin(np.clip(grav[0],-1.,1.))
                rotation=Rotation.from_euler('xyz',[roll,pitch,yaw]);quat=rotation.as_quat()[[3,0,1,2]]
                reference_data.qpos[:]=np.r_[pos[:2],float(g[-1]),quat,g[:NJ]+joint_default]
                mujoco.mj_kinematics(model,reference_data)
                target_pos=torch.as_tensor(reference_data.xpos[anchor_id]+origin,dtype=torch.float32,device='cuda:0')[None]
                target_quat=torch.as_tensor(reference_data.xquat[anchor_id],dtype=torch.float32,device='cuda:0')[None]
                relpos,relquat=subtract_frame_transforms(cmd.robot_anchor_pos_w,cmd.robot_anchor_quat_w,target_pos,target_quat)
                anchor=torch.cat([relpos,matrix_from_quat(relquat)[...,:2].reshape(1,-1)],dim=-1)
                goal=torch.cat([torch.as_tensor(window.reshape(1,-1),device='cuda:0'),anchor],dim=-1)
                action=policy(obs['policy'],goal)
                q=env.sim.data.qpos[0].cpu().numpy().copy();q[:3]-=origin
                states.append(np.r_[q,env.sim.data.qvel[0].cpu().numpy()].copy());goals.append(goal[0].cpu().numpy().copy());modes.append(mode)
                obs,_,terminated,timeout,_=env.step(action)
                # The fixed backing reference ends immediately: its clip_end is
                # not a physical failure. Record the actual height/tilt terms.
                physical={n:bool(env.termination_manager.get_term(n)[0]) for n in ('root_height','root_tilt') if n in env.termination_manager.active_terms}
                if len(physical)!=2:raise RuntimeError('Expected physical termination terms')
                if any(physical.values()):failures.append({'step':step,'time_s':time,'terms':physical});break
                lin=g[2*NJ+6:2*NJ+9];pos[:2]+=Rotation.from_euler('z',yaw).apply(lin)[:2]*.02
                yaw+=float(g[2*NJ+5])*.02;scheduler.advance()
        states=np.asarray(states);np.savez_compressed(a.out/'trajectory.npz',states=states,goals=np.asarray(goals),modes=np.asarray(modes))
        report={'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'steps':len(states),'seconds':len(states)*.02,
            'failures':failures,'resets_after_initialization':0,'max_future_overlap_error':max(overlap,default=0.),'schedule':schedule,
            'events':scheduler.events,'min_root_height_m':float(states[:,2].min()),'scope':'Exploratory scripted deterministic test, not keyboard or physical acceptance'}
        (a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    finally:env.close()


if __name__=='__main__':main()
