"""Measure startup pose mismatch under the unchanged zero-command policy."""
from pathlib import Path
import argparse,json

def main():
 p=argparse.ArgumentParser()
 for k in ['checkpoint','dataset','out']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args()
 import torch
 from supermimic.eval.benchmark import load_policy_from_checkpoint
 from supermimic.envs import SuperMimicEnvCfgArgs,make_env
 from supermimic.locomotion.config import VelocityCfg
 from supermimic.locomotion.task import measured_velocity
 from supermimic.robots.dimensions import PROFILE,NQ
 torch.set_num_threads(2);policy,cfg=load_policy_from_checkpoint(a.checkpoint,'cuda:0')
 env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(a.dataset/'walk.npz')],num_envs=8,seed=0,future_frames=cfg['env']['future_frames'],episode_length_s=12.,action_noise_std=0.,obs_noise=False,reset_noise=False,domain_randomization=False,record_transitions=False,anchor_obs=cfg['env']['anchor_obs'],velocity=VelocityCfg(**cfg['velocity'])),device='cuda:0')
 cmd=env.command_manager.get_term('motion');cmd.is_velocity[:]=True;cmd.force_standing_reset=True;cmd.set_manual_command((0.,0.,0.))
 names=PROFILE['joint_names'];arms=[i for i,n in enumerate(names) if any(k in n for k in ['shoulder','elbow','wrist'])];rows=[]
 try:
  obs,_=env.reset();initial=cmd.state_io.read().cpu()
  with torch.no_grad():
   for step in range(400):
    action=policy.velocity_action(obs['policy'],cmd.velocity_command);obs,_,term,tout,_=env.step(action)
    if step%10==0:
     state=cmd.state_io.read();rows.append(dict(time=(step+1)*.02,q=state[:,7:NQ].cpu().tolist(),dq=state[:,NQ+6:].cpu().tolist(),speed=measured_velocity(state).cpu().tolist(),action=action.cpu().tolist()))
    if (term|tout).any():raise RuntimeError('Diagnostic failed physically')
  result=dict(initial_joints=initial[:,7:NQ].tolist(),joint_names=names,arm_indices=arms,records=rows)
  a.out.write_text(json.dumps(result)+'\n')
  held=torch.tensor([r['q'] for r in rows if r['time']>5]).mean((0,1));print('arms',[(names[i],round(float(initial[0,7+i]),3),round(float(held[i]),3)) for i in arms])
 finally:env.close()
if __name__=='__main__':main()
