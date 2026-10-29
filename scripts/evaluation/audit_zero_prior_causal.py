"""Causal diagnostic: restore only old zero-command actions toward the old prior."""
from pathlib import Path
import argparse,json

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['candidate','prior','dataset','out']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();assert not a.out.exists()
 import torch,numpy as np
 from supermimic.eval.benchmark import load_policy_from_checkpoint
 from supermimic.envs import SuperMimicEnvCfgArgs,make_env
 from supermimic.locomotion.config import VelocityCfg
 from supermimic.locomotion.task import measured_velocity
 from supermimic.robots.dimensions import PROFILE,NQ
 torch.set_num_threads(2);policy,cfg=load_policy_from_checkpoint(a.candidate,'cuda:0');prior,_=load_policy_from_checkpoint(a.prior,'cuda:0');env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(a.dataset/'walk.npz')],num_envs=24,seed=0,future_frames=cfg['env']['future_frames'],episode_length_s=5,action_noise_std=0.,obs_noise=False,reset_noise=False,domain_randomization=False,record_transitions=False,anchor_obs=cfg['env']['anchor_obs'],velocity=VelocityCfg(**cfg['velocity'])),device='cuda:0');env.cfg.auto_reset=False
 cmd=env.command_manager.get_term('motion');cmd.is_velocity[:]=True;cmd.force_standing_reset=True;cmd.set_manual_command((0,0,0));obs,_=env.reset();arms=[i for i,n in enumerate(PROFILE['joint_names']) if any(t in n for t in ('shoulder','elbow','wrist'))];alpha=torch.tensor([0.,.5,1.],device='cuda').repeat_interleave(8)[:,None];trace=[];failed=torch.zeros(24,dtype=torch.bool,device='cuda')
 try:
  with torch.no_grad():
   for step in range(150):
    x=obs['policy'];c=cmd.velocity_command.clone();old=prior.velocity_action(x,c);new=policy.velocity_action(x,c);action=(1-alpha)*old+alpha*new;penalty=10*(new-old).square().mean(-1)
    obs,_,term,tout,_=env.step(action);failed|=term|tout;s=cmd.state_io.read()
    if step%5==0:trace.append(dict(time_s=(step+1)*env.step_dt,velocity=measured_velocity(s).cpu().tolist(),arm_square=s[:,NQ+6:][:,arms].square().mean(-1).cpu().tolist(),prior_action_cost=penalty.cpu().tolist()))
    if (term|tout).any():obs,_=env.reset(env_ids=(term|tout).nonzero().squeeze(-1))
  rows=[r for r in trace if r['time_s']>=1];v=np.array([r['velocity'] for r in rows]);arm=np.array([r['arm_square'] for r in rows]);cost=np.array([r['prior_action_cost'] for r in rows]);groups=[]
  for i,al in enumerate([0.,.5,1.]):
   sl=slice(i*8,(i+1)*8);groups.append(dict(new_standing_fraction=al,failed=failed[sl].cpu().tolist(),planar_speed=np.linalg.norm(v[:,sl,:2],axis=-1).mean(0).tolist(),absolute_yaw=np.abs(v[:,sl,2]).mean(0).tolist(),arm_rms=np.sqrt(arm[:,sl].mean(0)).tolist(),mean_prior_action_cost=cost[:,sl].mean(0).tolist()))
  report=dict(candidate=str(a.candidate),prior=str(a.prior),groups=groups,trace=trace,scope='Diagnostic action blending only, no trained or accepted policy claim');a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(groups,indent=2))
 finally:env.close()
if __name__=='__main__':main()
