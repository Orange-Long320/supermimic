"""Construct a physically checked bent-arm standing sample using unchanged assets.

Static inverse dynamics supplies a data-generation controller only. A policy
still has to learn the reference and pass the independent keyboard evaluation.
"""
from pathlib import Path
import argparse,json,hashlib

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['checkpoint','dataset','pose','out']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();a.out.mkdir(exist_ok=False)
 import numpy as np,torch,mujoco
 from scipy.optimize import least_squares
 from supermimic.envs import SuperMimicEnvCfgArgs,make_env
 from supermimic.locomotion.config import VelocityCfg
 from supermimic.locomotion.task import measured_velocity
 from supermimic.robots.dimensions import PROFILE,NQ
 from supermimic.kinematics.rotations import projected_gravity
 torch.set_num_threads(2)
 cfg=torch.load(a.checkpoint,map_location='cpu',weights_only=False)['config']
 env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(a.dataset/'walk.npz')],num_envs=2,seed=29,future_frames=cfg['env']['future_frames'],episode_length_s=30.,action_noise_std=0.,obs_noise=False,reset_noise=False,domain_randomization=False,record_transitions=False,anchor_obs=cfg['env']['anchor_obs'],velocity=VelocityCfg(**cfg['velocity'])),device='cuda:0');env.cfg.auto_reset=False
 cmd=env.command_manager.get_term('motion');cmd.is_velocity[:]=True;cmd.force_standing_reset=True;cmd.set_manual_command((0,0,0));env.reset();model=env.sim.mj_model;data=mujoco.MjData(model);names=PROFILE['joint_names'];arms=[i for i,n in enumerate(names) if any(t in n for t in ('shoulder','elbow','wrist'))];pose=json.loads(a.pose.read_text())['joint_positions_rad']
 base=cmd.state_io.read()[0,:NQ].cpu().numpy().astype(float);base[:2]=0;base[3:7]=[1,0,0,0]
 for n,v in pose.items():base[7+names.index(n)]=v
 def set_pose(v):
  q=base.copy();q[2]=v[0]
  for side in ['left','right']:
   q[7+names.index(side+'_hip_pitch_joint')]+=v[1]
   q[7+names.index(side+'_hip_roll_joint')]+=v[2]
   q[7+names.index(side+'_knee_joint')]+=v[3]
   q[7+names.index(side+'_ankle_pitch_joint')]-=v[1]+v[3]
   q[7+names.index(side+'_ankle_roll_joint')]-=v[2]
  data.qpos[:]=q;data.qvel[:]=0;data.qacc[:]=0;mujoco.mj_inverse(model,data);return q
 scans=[]
 for z in np.linspace(base[2]-.12,base[2]+.04,161):
  set_pose([z,0,0,0]);scans.append((np.linalg.norm(data.qfrc_inverse[:6]),z))
 start=min(scans)[1]
 def residual(v):
  set_pose(v);return np.r_[data.qfrc_inverse[:3],data.qfrc_inverse[3:6]*5,np.array(v[1:])*.05]
 fit=least_squares(residual,[start,0,0,0],bounds=([base[2]-.15,-.3,-.15,-.25],[base[2]+.06,.3,.15,.25]),max_nfev=500,diff_step=1e-4,ftol=1e-11,xtol=1e-11,gtol=1e-9)
 qeq=set_pose(fit.x);wrench=data.qfrc_inverse[:6].copy();tau=data.qfrc_inverse[6:].copy();kp=np.array([PROFILE['stiffness'][n] for n in names]);effort=np.array([PROFILE['effort_limit'][n] for n in names]);target=qeq[7:]+tau/kp
 report=dict(source_checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),selected_pose=str(a.pose),solver_success=bool(fit.success),solver_variables=fit.x.tolist(),base_inverse_wrench=wrench.tolist(),max_effort_fraction=float(np.max(np.abs(tau)/effort)),joint_names=names,equilibrium_qpos=qeq.tolist(),pd_position_target=target.tolist(),scope='Candidate generated standing reference; no deployment controller change or policy success claim.')
 (a.out/'equilibrium.json').write_text(json.dumps(report,indent=2)+'\n')
 if np.linalg.norm(wrench[:3])>1 or np.linalg.norm(wrench[3:])>.2 or np.max(np.abs(tau)/effort)>.8:
  report.update(passed=False,reason='static_force_or_torque_residual');(a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');env.close();print(json.dumps(report));return
 # Reference validation deliberately starts at the solved pose. Cold-start policy
 # acceptance remains separate and continues to use the original initial state.
 s=cmd.state_io.read();s[:,:NQ]=torch.tensor(qeq,device='cuda',dtype=s.dtype);s[:,NQ:]=0;cmd.state_io.write(torch.arange(2,device='cuda'),s)
 act=env.action_manager.get_term('joint_pos');actions=(torch.tensor(target,device='cuda',dtype=s.dtype)-act._offset)/act._scale
 frames=[];trace=[];failed=torch.zeros(2,dtype=torch.bool,device='cuda')
 try:
  with torch.no_grad():
   for step in range(1000):
    _,_,term,tout,_=env.step(actions);failed|=term|tout;state=cmd.state_io.read();robot=cmd.robot
    trace.append(dict(time_s=(step+1)*env.step_dt,velocity=measured_velocity(state).cpu().numpy(),arm_square=state[:,NQ+6:][:,arms].square().mean(-1).cpu().numpy(),position=state[:,:3].cpu().numpy(),tilt=projected_gravity(state[:,3:7])[:,:2].norm(dim=-1).cpu().numpy()))
    if step>=100:frames.append(dict(joint_pos=robot.data.joint_pos[0].cpu().numpy().copy(),joint_vel=robot.data.joint_vel[0].cpu().numpy().copy(),body_pos_w=(robot.data.body_pos_w[0]-env.scene.env_origins[0]).cpu().numpy().copy(),body_quat_w=robot.data.body_quat_w[0].cpu().numpy().copy(),body_lin_vel_w=robot.data.body_lin_vel_w[0].cpu().numpy().copy(),body_ang_vel_w=robot.data.body_ang_vel_w[0].cpu().numpy().copy()))
    if failed.any():break
  rows=trace[100:];passed=bool(rows) and not failed.any();v=np.array([r['velocity'] for r in rows]);pos=np.array([r['position'] for r in rows]);arm=np.array([r['arm_square'] for r in rows]);metrics={} if not rows else dict(max_mean_planar=float(np.linalg.norm(v[:,:,:2],axis=-1).mean(0).max()),max_mean_abs_yaw=float(np.abs(v[:,:,2]).mean(0).max()),max_arm_rms=float(np.sqrt(arm.mean(0)).max()),max_xy_excursion=float(np.linalg.norm(pos[:,:,:2]-pos[:1,:,:2],axis=-1).max()))
  if rows:passed=passed and metrics['max_mean_planar']<.01 and metrics['max_mean_abs_yaw']<.02 and metrics['max_arm_rms']<.05 and metrics['max_xy_excursion']<.02
  report.update(passed=bool(passed),failed=failed.cpu().tolist(),physics_seconds=len(trace)*env.step_dt,metrics=metrics)
  if passed:
   body_names=np.load(a.dataset/'walk.npz')['body_names'];assert list(body_names)==robot.body_names
   payload={k:np.stack([f[k] for f in frames[:500]]) for k in frames[0]};payload.update(fps=np.array([1/env.step_dt]),joint_names=np.array(names),body_names=body_names,source=np.array('Generated static inverse dynamics + 20-second physical hold, unchanged X2 Ultra assets; reference data only.'))
   path=a.out/'stand_bent.npz';np.savez_compressed(path,**payload);report.update(reference=str(path),reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reference_frames=len(payload['joint_pos']))
  (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
 finally:env.close()
if __name__=='__main__':main()
