"""Causal diagnostic only: test a fixed nominal/bent PD target in real physics."""
from pathlib import Path
import argparse,json

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['checkpoint','dataset','pose','out']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();assert not a.out.exists()
 import torch,numpy as np
 from supermimic.envs import SuperMimicEnvCfgArgs,make_env
 from supermimic.locomotion.config import VelocityCfg
 from supermimic.locomotion.task import measured_velocity
 from supermimic.robots.dimensions import PROFILE,NQ
 cfg=torch.load(a.checkpoint,map_location='cpu',weights_only=False)['config'];torch.set_num_threads(2)
 env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(a.dataset/'walk.npz')],num_envs=16,seed=0,future_frames=cfg['env']['future_frames'],episode_length_s=25,action_noise_std=0.,obs_noise=False,reset_noise=False,domain_randomization=False,record_transitions=False,anchor_obs=cfg['env']['anchor_obs'],velocity=VelocityCfg(**cfg['velocity'])),device='cuda:0');env.cfg.auto_reset=False
 cmd=env.command_manager.get_term('motion');cmd.is_velocity[:]=True;cmd.force_standing_reset=True;cmd.set_manual_command((0,0,0));obs,_=env.reset();act=env.action_manager.get_term('joint_pos');arms=[i for i,n in enumerate(PROFILE['joint_names']) if any(k in n for k in ('shoulder','elbow','wrist'))];pose=json.loads(a.pose.read_text())['joint_positions_rad']
 desired=cmd.library.default_joint_pos.expand(16,-1).clone()
 for i in arms:desired[8:,i]=pose[PROFILE['joint_names'][i]]
 action=(desired-act._offset)/act._scale
 failed=torch.zeros(16,dtype=torch.bool,device='cuda');trace=[];observations=[]
 try:
  with torch.no_grad():
   for step in range(1000):
    if step%2==0:observations.append(obs['policy'].clone())
    obs,_,term,tout,_=env.step(action);failed|=term|tout;s=cmd.state_io.read()
    if step%10==0:trace.append(dict(time_s=(step+1)*env.step_dt,velocity=measured_velocity(s).cpu().tolist(),arm_rms=s[:,NQ+6:][:,arms].square().mean(-1).sqrt().cpu().tolist(),position=s[:,:3].cpu().tolist(),joints=s[:,7:NQ].cpu().tolist(),failed=failed.cpu().tolist()))
    if (term|tout).any():obs,_=env.reset(env_ids=(term|tout).nonzero().squeeze(-1))
  a.out.mkdir();torch.save(dict(observations=torch.stack(observations).cpu(),actions=action.cpu(),default_joint_positions=cmd.library.default_joint_pos.cpu(),selected_target=desired.cpu(),scope='Diagnostic fixed PD controller, not a trained policy'),a.out/'states.pt');(a.out/'probe.json').write_text(json.dumps(dict(failed=failed.cpu().tolist(),trace=trace,scope='Diagnostic action override, not deployable learned policy'),indent=2))
  for name,sl in [('default',slice(0,8)),('bent',slice(8,16))]:
   rows=[r for r in trace if r['time_s']>=1];v=np.array([r['velocity'] for r in rows])[:,sl];arm=np.array([r['arm_rms'] for r in rows])[:,sl];pos=np.array([r['position'] for r in rows])[:,sl];print(json.dumps(dict(name=name,failed=failed[sl].cpu().tolist(),planar=np.linalg.norm(v[:,:,:2],axis=-1).mean(0).tolist(),yaw=np.abs(v[:,:,2]).mean(0).tolist(),arm_rms=np.sqrt((arm**2).mean(0)).tolist(),drift=np.linalg.norm(pos[-1,:,:2]-pos[0,:,:2],axis=-1).tolist())))
 finally:env.close()
if __name__=='__main__':main()
