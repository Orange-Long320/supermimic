"""Audit actual standing data, old control-prior labels and physical baseline."""
from pathlib import Path
import argparse,json,hashlib

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['checkpoint','dataset','out','standing-pose']:p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();a.out.mkdir(exist_ok=False)
 import numpy as np,torch
 from supermimic.eval.benchmark import load_policy_from_checkpoint
 from supermimic.envs import SuperMimicEnvCfgArgs,make_env
 from supermimic.locomotion.config import VelocityCfg
 from supermimic.locomotion.task import measured_velocity
 from supermimic.robots.dimensions import PROFILE,NQ
 torch.set_num_threads(2)
 policy,saved=load_policy_from_checkpoint(a.checkpoint,'cuda:0');policy.requires_grad_(False)
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);prior_meta=ck['velocity_training']['control_prior']
 prior,_=load_policy_from_checkpoint(Path(prior_meta['checkpoint']),'cuda:0');prior.requires_grad_(False)
 clips=[]
 for path in sorted(a.dataset.glob('*.npz')):
  d=np.load(path);arms=[i for i,n in enumerate(d['joint_names']) if any(t in str(n) for t in ['shoulder','elbow','wrist'])]
  planar=np.linalg.norm(d['body_lin_vel_w'][:,0,:2],axis=-1);yaw=np.abs(d['body_ang_vel_w'][:,0,2]);arm=np.sqrt(np.mean(d['joint_vel'][:,arms]**2,axis=-1))
  quiet=(planar<=.06)&(yaw<=.08)&(arm<=.12);longest=cur=0
  for ok in quiet:cur=cur+1 if ok else 0;longest=max(longest,cur)
  fps=float(d['fps'][0]);clips.append(dict(name=path.stem,frames=len(quiet),quiet_frames=int(quiet.sum()),longest_quiet_seconds=longest/fps,mean_arm_velocity_rms=float(arm.mean())))
 env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(a.dataset/'walk.npz')],num_envs=8,seed=0,future_frames=saved['env']['future_frames'],episode_length_s=12,action_noise_std=0.,obs_noise=False,reset_noise=False,domain_randomization=False,record_transitions=False,anchor_obs=saved['env']['anchor_obs'],velocity=VelocityCfg(**saved['velocity'])),device='cuda:0')
 cmd=env.command_manager.get_term('motion');cmd.is_velocity[:]=True;cmd.force_standing_reset=True;cmd.set_manual_command((0.,0.,0.))
 arms=[i for i,n in enumerate(PROFILE['joint_names']) if any(t in n for t in ['shoulder','elbow','wrist'])]
 target=json.loads(a.standing_pose.read_text())['joint_positions_rad'];pose=torch.tensor([target[PROFILE['joint_names'][i]] for i in arms],device='cuda');rows=[]
 try:
  obs,_=env.reset()
  with torch.no_grad():
   for step in range(500):
    x=obs['policy'];c=cmd.velocity_command.clone();z=policy.encoders['velocity'](x,c);zp=prior.encoders['velocity'](x,c);action=policy.decoder(z,x)
    prior_loss=prior_meta['weight']*((z-zp).square().mean()+(action-prior.decoder(zp,x)).square().mean())
    obs,_,term,tout,_=env.step(action);s=cmd.state_io.read()
    if step%10==0:rows.append(dict(time_s=(step+1)*env.step_dt,velocity=measured_velocity(s).cpu().tolist(),arms_rms=s[:,NQ+6:][:,arms].square().mean(-1).sqrt().cpu().tolist(),old_zero_prior_weighted_loss=float(prior_loss),old_default_posture_error=(s[:,7:NQ][:,arms]-cmd.library.default_joint_pos[arms]).square().mean(-1).cpu().tolist(),bent_posture_error=(s[:,7:NQ][:,arms]-pose).square().mean(-1).cpu().tolist(),failed=(term|tout).cpu().tolist()))
  defaults=cmd.library.default_joint_pos[arms]
  report=dict(checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),data=clips,control_prior=prior_meta,old_prior_zero_query_fraction=.5,default_to_selected_arm_pose_rmse_rad=float((defaults-pose).square().mean().sqrt()),samples=rows,scope='Evidence audit only; no candidate is accepted or trained here.')
  (a.out/'audit.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
  print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2,default=str))
 finally:env.close()
if __name__=='__main__':main()
