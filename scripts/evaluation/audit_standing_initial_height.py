"""Paired physical audit: change only root height to the actual sole surface."""
from pathlib import Path
import argparse,json

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['checkpoint','dataset','out']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();a.out.mkdir(exist_ok=False)
 import torch,numpy as np,mujoco
 from supermimic.eval.benchmark import load_policy_from_checkpoint
 from supermimic.envs import SuperMimicEnvCfgArgs,make_env
 from supermimic.locomotion.config import VelocityCfg
 from supermimic.locomotion.task import measured_velocity
 from supermimic.robots.dimensions import NQ,PROFILE
 torch.set_num_threads(2);policy,cfg=load_policy_from_checkpoint(a.checkpoint,'cuda:0');env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(a.dataset/'walk.npz')],num_envs=8,seed=0,future_frames=cfg['env']['future_frames'],episode_length_s=8,action_noise_std=0.,obs_noise=False,reset_noise=False,domain_randomization=False,record_transitions=False,anchor_obs=cfg['env']['anchor_obs'],velocity=VelocityCfg(**cfg['velocity'])),device='cuda:0');env.cfg.auto_reset=False;cmd=env.command_manager.get_term('motion');cmd.is_velocity[:]=True;cmd.force_standing_reset=True;cmd.set_manual_command((0,0,0));obs,_=env.reset();initial=cmd.state_io.read().clone();initial_obs=obs['policy'].clone();model=env.sim.mj_model;data=mujoco.MjData(model);clearance=[]
 for s in initial.cpu().numpy():
  data.qpos[:]=s[:NQ];data.qvel[:]=0;mujoco.mj_forward(model,data);minimum=float('inf')
  for g in range(model.ngeom):
   if '/foot_' not in model.geom(g).name:continue
   mid=model.geom_dataid[g];v=model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]];world=v@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g];minimum=min(minimum,float(world[:,2].min()))
  clearance.append(minimum)
 arms=[i for i,n in enumerate(PROFILE['joint_names']) if any(k in n for k in ['shoulder','elbow','wrist'])];result=[]
 try:
  for lowered in [False,True]:
   env.reset();s=initial.clone()
   if lowered:s[:,2]-=torch.tensor(clearance,device='cuda')+.0005
   cmd.state_io.write(torch.arange(8,device='cuda'),s);x=initial_obs.clone();failed=torch.zeros(8,dtype=torch.bool,device='cuda');rows=[]
   with torch.no_grad():
    for step in range(150):
     action=policy.velocity_action(x,cmd.velocity_command);nxt,_,term,tout,_=env.step(action);x=nxt['policy'];failed|=term|tout;state=cmd.state_io.read()
     rows.append(dict(time_s=(step+1)*env.step_dt,velocity=measured_velocity(state).cpu().numpy(),arm_square=state[:,NQ+6:][:,arms].square().mean(-1).cpu().numpy(),position=state[:,:3].cpu().numpy()))
     if (term|tout).any():x=env.reset(env_ids=(term|tout).nonzero().squeeze(-1))[0]['policy']
   steady=[r for r in rows if r['time_s']>=1];v=np.array([r['velocity'] for r in steady]);arm=np.array([r['arm_square'] for r in steady]);pos=np.array([r['position'] for r in steady]);result.append(dict(only_change='root_height_to_sole_contact' if lowered else 'original',failed=failed.cpu().tolist(),planar_speed=np.linalg.norm(v[:,:,:2],axis=-1).mean(0).tolist(),absolute_yaw=np.abs(v[:,:,2]).mean(0).tolist(),arm_rms=np.sqrt(arm.mean(0)).tolist(),net_xy_displacement=np.linalg.norm(pos[-1,:,:2]-pos[0,:,:2],axis=-1).tolist()))
  report=dict(checkpoint=str(a.checkpoint),original_root_height=initial[:,2].cpu().tolist(),original_sole_clearance_m=clearance,results=result,scope='Paired causal height diagnostic; no weight/config mutation or accepted policy claim');(a.out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
 finally:env.close()
if __name__=='__main__':main()
