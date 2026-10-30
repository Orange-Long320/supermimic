"""Evaluate one staged reference with a frozen tracker and no simulator resets.

Before release the future window follows the continue-walking reference, so
it cannot reveal the later release; after release it follows the stop reference.
Exploratory reference diagnostics only, not a controller acceptance test.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists() or len(os.environ.get('CUDA_VISIBLE_DEVICES','').split(','))!=1 or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        p.error('Require a new output directory and an explicitly selected single GPU')
    import numpy as np
    import torch
    from supermimic.envs import SuperMimicEnvCfgArgs,make_env
    from supermimic.envs.mdp.motion_library import MotionLibrary
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    manifest=json.loads((a.candidate/'manifest.json').read_text())
    candidate=a.candidate/'candidate.npz'; continuation=a.candidate/'continue.npz'
    if sha(candidate)!=manifest['candidate_sha256']:raise ValueError('Candidate hash mismatch')
    torch.set_num_threads(2)
    policy,cfg=load_policy_from_checkpoint(a.checkpoint,'cuda:0')
    k=cfg['env']['future_frames']; release=int(manifest['release_frame'])
    env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(candidate)],num_envs=1,seed=0,
        future_frames=k,episode_length_s=1000.,action_noise_std=0.,obs_noise=False,reset_noise=False,
        domain_randomization=False,record_transitions=False,sampling_mode='start',anchor_obs=cfg['env'].get('anchor_obs',False)),device='cuda:0')
    env.cfg.auto_reset=False
    a.out.mkdir(parents=True)
    cmd=env.command_manager.get_term('motion');cmd.forced_frame[:]=0
    alternative=MotionLibrary([continuation],cmd.cfg.body_names,cmd.library.default_joint_pos,device='cuda:0')
    if alternative.num_frames<release+k:raise ValueError('Continuation does not cover the release horizon')
    # Verify the prefix is exactly shared; the branch only affects future goals.
    ix=torch.arange(release-1,device='cuda:0')
    if not torch.allclose(cmd.library.ref71(ix),alternative.ref71(ix),atol=1e-5,rtol=0):
        raise ValueError('Candidate and continuation prefixes differ')
    original_compute=env.termination_manager.compute;failures=[];step_index=[0]
    def capture_termination():
        done=original_compute()
        if bool(done.any()):
            failures.append({'step':step_index[0],'frame':int(cmd.frame[0]),'done':done.cpu().tolist(),
                'terms':{name:bool(env.termination_manager.get_term(name)[0]) for name in env.termination_manager.active_terms}})
        return torch.zeros_like(done)
    env.termination_manager.compute=capture_termination
    try:
        obs,_=env.reset();states=[];goals=[];frames=[];actions=[];phases=[]
        with torch.no_grad():
            for step in range(len(manifest['phases'])-2):
                step_index[0]=step
                frame=int(cmd.frame[0]); source=alternative if frame<release else cmd.library
                future=source.goal_window(cmd.frame,k)
                goal=torch.cat([future,obs['goal'][:,future.shape[1]:]],dim=-1)
                qpos=env.sim.data.qpos[0].cpu().numpy().copy();qpos[:3]-=env.scene.env_origins[0].cpu().numpy()
                states.append(np.r_[qpos,env.sim.data.qvel[0].cpu().numpy()].copy());goals.append(goal[0].cpu().numpy().copy())
                frames.append(frame);phases.append(manifest['phases'][frame]);action=policy(obs['policy'],goal)
                actions.append(action[0].cpu().numpy().copy());obs,_,_,_,_=env.step(action)
                if failures:break
        np.savez_compressed(a.out/'trajectory.npz',states=np.asarray(states),goals=np.asarray(goals),
                            actions=np.asarray(actions),frames=np.asarray(frames),phases=np.asarray(phases))
        states=np.asarray(states);report={'candidate_sha256':sha(candidate),'continuation_sha256':sha(continuation),
            'checkpoint':str(a.checkpoint),'checkpoint_sha256':sha(a.checkpoint),'interpreter':sys.executable,
            'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'gpu':os.environ['CUDA_VISIBLE_DEVICES'],
            'release_frame':release,'future_frames':k,'first_frame':frames[0],'last_frame':frames[-1],
            'steps':len(frames),'reached_release':frames[-1]>=release,'reached_hold':bool('stand' in phases),
            'failure':failures,'state_resets_after_initialization':0,'goal_before_release':'continuation without release preview',
            'finite':bool(np.isfinite(states).all()),'final_root_height_m':float(states[-1,2]),
            'status':'failed' if failures else 'completed_unaccepted',
            'scope':'Exploratory one-phase release; no acceptance without physical reference validity and long-horizon regression.'}
        (a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    finally:env.close()

if __name__=='__main__':main()
