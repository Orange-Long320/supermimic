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
    p.add_argument('--mode',choices=['all','walk','jog'],default='all');p.add_argument('--hold-seconds',type=float,default=20.)
    a=p.parse_args()
    import os
    if a.out.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(a.gpu):
        raise ValueError('New output directory and explicit GPU required')
    import torch
    from supermimic.control.skill_tracker import SkillTracker
    from mjlab.envs import ManagerBasedRlEnv
    from supermimic.envs import SuperMimicEnvCfgArgs,make_env_cfg
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    torch.set_num_threads(2)
    policy,cfg=load_policy_from_checkpoint(a.checkpoint,'cuda:0')
    stand_file=a.dataset/'walk_stop.npz'
    with np.load(stand_file) as z:initial=len(z['joint_pos'])-1
    env_cfg=make_env_cfg(SuperMimicEnvCfgArgs(npz_files=[str(stand_file)],num_envs=1,seed=0,
        future_frames=cfg['env']['future_frames'],episode_length_s=1000.,action_noise_std=0.,obs_noise=False,
        reset_noise=False,domain_randomization=False,record_transitions=False,sampling_mode='start',anchor_obs=True))
    env_cfg.auto_reset=False
    # The backing stand clip is deliberately frozen; its clip-end timeout is
    # irrelevant to the scheduled goals. Physical termination terms remain.
    del env_cfg.terminations['clip_end']
    env=ManagerBasedRlEnv(env_cfg,device='cuda:0')
    cmd=env.command_manager.get_term('motion');cmd.forced_frame[:]=initial
    schedule=[(0.,'stand'),(3.,'walk'),(7.,'stand'),(11.,'backward'),(15.,'stand'),(19.,'turn_left_inplace'),(23.,'stand'),(27.,'turn_right_inplace'),(31.,'stand'),(35.,'jog'),(39.,'stand')]
    duration=45.
    if a.mode!='all':
        if a.hold_seconds<5:raise ValueError('Hold must be at least 5 seconds')
        schedule=[(0.,'stand'),(3.,a.mode),(3.+a.hold_seconds,'stand')]
        duration=3.+a.hold_seconds+5.
    a.out.mkdir(parents=True)
    try:
        obs,_=env.reset();origin=env.scene.env_origins[0].cpu().numpy();tracker=SkillTracker(env,a.dataset,cfg)
        states=[]; goals=[]; modes=[];failures=[];overlap=[]; previous=None
        with torch.no_grad():
            for step in range(round(50*duration)):
                time=step*.02
                mode=next(v for t,v in reversed(schedule) if time>=t)
                goal=tracker.goal(mode);window=tracker.scheduler.window()
                if previous is not None:overlap.append(float(np.max(np.abs(previous[1:]-window[:-1]))))
                previous=window.copy()
                action=policy(obs['policy'],goal)
                q=env.sim.data.qpos[0].cpu().numpy().copy();q[:3]-=origin
                states.append(np.r_[q,env.sim.data.qvel[0].cpu().numpy()].copy());goals.append(goal[0].cpu().numpy().copy());modes.append(mode)
                obs,_,terminated,timeout,_=env.step(action)
                physical={n:bool(env.termination_manager.get_term(n)[0]) for n in ('root_height','root_tilt') if n in env.termination_manager.active_terms}
                if len(physical)!=2:raise RuntimeError('Expected physical termination terms')
                if any(physical.values()):failures.append({'step':step,'time_s':time,'terms':physical});break
                tracker.advance()
        states=np.asarray(states);np.savez_compressed(a.out/'trajectory.npz',states=states,goals=np.asarray(goals),modes=np.asarray(modes))
        report={'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'steps':len(states),'seconds':len(states)*.02,
            'failures':failures,'resets_after_initialization':0,'max_future_overlap_error':max(overlap,default=0.),'schedule':schedule,
            'events':tracker.scheduler.events,'min_root_height_m':float(states[:,2].min()),'scope':'Exploratory scripted deterministic test, not keyboard or physical acceptance'}
        (a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    finally:env.close()


if __name__=='__main__':main()
