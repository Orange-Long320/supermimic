"""Server-only matched backward teacher/student direction probe."""
from pathlib import Path
import argparse,json

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['checkpoint','dataset','out']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();a.out.mkdir(exist_ok=False)
 import torch
 from supermimic.eval.benchmark import load_policy_from_checkpoint
 from supermimic.envs import SuperMimicEnvCfgArgs,make_env
 from supermimic.locomotion.config import VelocityCfg
 from supermimic.locomotion.task import measured_velocity
 torch.set_num_threads(2)
 policy,saved=load_policy_from_checkpoint(a.checkpoint,'cuda:0')
 args=SuperMimicEnvCfgArgs(npz_files=[str(a.dataset/'backward.npz')],num_envs=8,seed=0,
  future_frames=saved['env']['future_frames'],episode_length_s=20.,action_noise_std=0.,obs_noise=False,
  reset_noise=False,domain_randomization=False,record_transitions=False,anchor_obs=saved['env']['anchor_obs'],velocity=VelocityCfg(**saved['velocity']))
 env=make_env(args,device='cuda:0');env.cfg.auto_reset=False
 cmd=env.command_manager.get_term('motion');cmd.is_velocity[:]=True
 result=[]
 try:
  with torch.no_grad():
   for standing in [False,True]:
    for teacher in [True,False]:
     cmd.force_standing_reset=standing;cmd.forced_frame[:]=0;cmd.set_manual_command((-.2,0.,0.))
     obs,_=env.reset();rows=[]
     # Stay within the finite teacher clip; extend only student beyond it.
     steps=min(int(cmd.library.clips[0].length)-2,300) if teacher else 400
     for step in range(steps):
      x=obs['policy'];g=obs['goal'];c=cmd.velocity_command
      z=policy.encoders['velocity'](x,c)
      plus=c.clone();plus[:,0]=.2
      action=policy(x,g) if teacher else policy.decoder(z,x)
      obs,_,term,tout,_=env.step(action)
      if step%5==0:
       rows.append(dict(t=(step+1)*env.step_dt,velocity=measured_velocity(cmd.state_io.read()).tolist(),
        latent_command_delta=float((z-policy.encoders['velocity'](x,plus)).square().mean()),
        action_command_delta=float((policy.decoder(z,x)-policy.velocity_action(x,plus)).square().mean())))
      if (term|tout).any():break
     result.append(dict(standing=standing,teacher=teacher,steps=step+1,failed=bool(term.any()),samples=rows))
     print(json.dumps({k:v for k,v in result[-1].items() if k!='samples'}),flush=True)
 finally:env.close()
 (a.out/'probe.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
