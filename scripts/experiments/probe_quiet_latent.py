"""Server-only reachability diagnostic for quiet-arm latents; does not save policy."""
import argparse,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['checkpoint','dataset','out']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();a.out.mkdir(exist_ok=False)
 import torch
 from supermimic.eval.benchmark import load_policy_from_checkpoint
 from supermimic.envs import SuperMimicEnvCfgArgs,make_env
 from supermimic.locomotion.config import VelocityCfg
 from supermimic.robots.dimensions import PROFILE
 torch.set_num_threads(2);policy,cfg=load_policy_from_checkpoint(a.checkpoint,'cuda:0');policy.requires_grad_(False)
 env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(a.dataset/'walk.npz')],num_envs=4,seed=0,future_frames=cfg['env']['future_frames'],episode_length_s=20.,action_noise_std=0.,obs_noise=False,reset_noise=False,domain_randomization=False,record_transitions=False,anchor_obs=cfg['env']['anchor_obs'],velocity=VelocityCfg(**cfg['velocity'])),device='cuda:0')
 cmd=env.command_manager.get_term('motion');cmd.is_velocity[:]=True;cmd.force_standing_reset=True;cmd.set_manual_command((0.,0.,0.))
 arm=[i for i,n in enumerate(PROFILE['joint_names']) if any(k in n for k in ('shoulder','elbow','wrist'))];other=[i for i in range(len(PROFILE['joint_names'])) if i not in arm]
 xs=[]
 try:
  obs,_=env.reset()
  with torch.no_grad():
   for step in range(200):
    if step>=50 and step%2==0:xs.append(obs['policy'].clone())
    obs,*_=env.step(policy.velocity_action(obs['policy'],cmd.velocity_command))
 finally:env.close()
 x=torch.cat(xs);c=torch.zeros(len(x),3,device='cuda');z0=policy.encoders['velocity'](x,c).detach();target=policy.decoder(z0,x).detach()
 target[:,arm]=target[:,arm].mean(0)
 z=z0.clone().requires_grad_();opt=torch.optim.Adam([z],lr=.03)
 def metrics():
  y=policy.decoder(z,x);return dict(arm_action_rmse=float((y[:,arm]-target[:,arm]).square().mean().sqrt()),other_action_rmse=float((y[:,other]-target[:,other]).square().mean().sqrt()))
 before=metrics()
 for step in range(400):
  y=policy.decoder(z,x);loss=(y[:,arm]-target[:,arm]).square().mean()+5*(y[:,other]-target[:,other]).square().mean()
  opt.zero_grad();loss.backward();opt.step()
  with torch.no_grad():z.clamp_(-.999,.999)
 after=metrics();r=dict(before=before,after=after,scope='Decoder reachability diagnostic only, not physical acceptance')
 (a.out/'projection.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
 torch.save(dict(observations=x.cpu(),command=c.cpu(),target_latent=z.detach().cpu()),a.out/'diagnostic.pt')
if __name__=='__main__':main()
