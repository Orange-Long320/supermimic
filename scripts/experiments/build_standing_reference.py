"""Construct a physically checked bent-arm standing sample using unchanged assets.

Static inverse dynamics supplies a data-generation controller only. A policy
still has to learn the reference and pass the independent keyboard evaluation.
"""
from pathlib import Path
import argparse,json,hashlib

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['checkpoint','dataset','pose','out']:p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--equilibrium',type=Path)
 a=p.parse_args();a.out.mkdir(exist_ok=False);(a.out/'entry-as-run.py').write_bytes(Path(__file__).read_bytes())
 import numpy as np,torch,mujoco
 from scipy.optimize import least_squares
 from scipy.spatial import ConvexHull
 from supermimic.envs import SuperMimicEnvCfgArgs,make_env
 from supermimic.locomotion.config import VelocityCfg
 from supermimic.locomotion.task import measured_velocity
 from supermimic.robots.dimensions import PROFILE,NQ
 from supermimic.kinematics.rotations import projected_gravity
 torch.set_num_threads(2)
 cfg=torch.load(a.checkpoint,map_location='cpu',weights_only=False)['config']
 env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(a.dataset/'walk.npz')],num_envs=2,seed=29,future_frames=cfg['env']['future_frames'],episode_length_s=30.,action_noise_std=0.,obs_noise=False,reset_noise=False,domain_randomization=False,record_transitions=False,anchor_obs=cfg['env']['anchor_obs'],velocity=VelocityCfg(**cfg['velocity'])),device='cuda:0');env.cfg.auto_reset=False
 cmd=env.command_manager.get_term('motion');cmd.is_velocity[:]=True;cmd.force_standing_reset=True;cmd.set_manual_command((0,0,0));env.reset();model=env.sim.mj_model;data=mujoco.MjData(model);names=PROFILE['joint_names'];arms=[i for i,n in enumerate(names) if any(t in n for t in ('shoulder','elbow','wrist'))];pose=json.loads(a.pose.read_text())['joint_positions_rad']
 if a.equilibrium:
  report=json.loads(a.equilibrium.read_text());assert report['source_checkpoint_sha256']==hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
  qeq=np.array(report['equilibrium_qpos']);target=np.array(report['pd_position_target']);data.qpos[:]=qeq;data.qvel[:]=0;data.qacc[:]=0;mujoco.mj_inverse(model,data)
  assert np.linalg.norm(data.qfrc_inverse[:3])<1 and np.linalg.norm(data.qfrc_inverse[3:6])<.2
  report={k:v for k,v in report.items() if k not in ['passed','failed','reason','physics_seconds','metrics']};report.update(reused_equilibrium=str(a.equilibrium),entry_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(a.out/'equilibrium.json').write_text(json.dumps(report,indent=2)+'\n')
 else:
  base=cmd.state_io.read()[0,:NQ].cpu().numpy().astype(float);base[:2]=0;base[3:7]=[1,0,0,0]
  for n,v in pose.items():base[7+names.index(n)]=v
  sole_pitch=sole_roll=0.
  def set_pose(v):
   q=base.copy();q[2]=v[0]
   cr,sr=np.cos(v[4]/2),np.sin(v[4]/2);cp,sp=np.cos(v[5]/2),np.sin(v[5]/2);q[3:7]=[cr*cp,sr*cp,cr*sp,-sr*sp]
   for side in ['left','right']:
    q[7+names.index(side+'_hip_pitch_joint')]+=v[1]
    q[7+names.index(side+'_hip_roll_joint')]+=v[2]
    asym=v[10]*(1 if side=='left' else -1) if len(v)>6 else 0.
    q[7+names.index(side+'_knee_joint')]+=v[3]+asym
    q[7+names.index(side+'_ankle_pitch_joint')]+=sole_pitch-v[1]-v[3]-v[5]-asym+(v[6 if side=='left' else 7] if len(v)>6 else 0.)
    q[7+names.index(side+'_ankle_roll_joint')]+=sole_roll-v[2]-v[4]+(v[8 if side=='left' else 9] if len(v)>6 else 0.)
   data.qpos[:]=q;data.qvel[:]=0;data.qacc[:]=0;mujoco.mj_inverse(model,data);return q
  scans=[]
  for z in np.linspace(base[2]-.12,base[2]+.04,161):
   set_pose([z,0,0,0,0,0]);scans.append((np.linalg.norm(data.qfrc_inverse[:6]),z))
  start=min(scans)[1]
  # Align the largest downward face of each original collision mesh with the
  # floor. Joint-angle sums of zero do not establish that a mesh sole is flat.
  set_pose([start,0,0,0,0,0]);sole_normals=[]
  for geom_name in ['robot/foot_6_collision','robot/foot_12_collision']:
   gi=model.geom(geom_name).id;mi=model.geom_dataid[gi];vertices=model.mesh_vert[model.mesh_vertadr[mi]:model.mesh_vertadr[mi]+model.mesh_vertnum[mi]];hull=ConvexHull(vertices);R=data.geom_xmat[gi].reshape(3,3);groups={}
   for face,plane in zip(hull.simplices,hull.equations):
    normal=plane[:3]
    if (R@normal)[2]>-.9:continue
    area=np.linalg.norm(np.cross(vertices[face[1]]-vertices[face[0]],vertices[face[2]]-vertices[face[0]]))/2;key=tuple(np.round(normal,4));old=groups.get(key,(0.,normal));groups[key]=(old[0]+float(area),normal)
   _,normal=max(groups.values(),key=lambda t:t[0]);sole_normals.append((gi,normal))
  def align_feet(v):
   nonlocal sole_pitch,sole_roll
   sole_pitch,sole_roll=v;set_pose([start,0,0,0,0,0]);return np.concatenate([(data.geom_xmat[g].reshape(3,3)@normal)[:2] for g,normal in sole_normals])
  sole_fit=least_squares(align_feet,[0.,0.],bounds=(-.08,.08),jac='3-point',diff_step=1e-5,gtol=1e-12,ftol=1e-12,xtol=1e-12);align_feet(sole_fit.x)
  def residual(v):
   set_pose(v)
   normals=np.concatenate([(data.geom_xmat[g].reshape(3,3)@normal)[:2] for g,normal in sole_normals])
   bottom=[]
   for g,_ in sole_normals:
    mi=model.geom_dataid[g];vertices=model.mesh_vert[model.mesh_vertadr[mi]:model.mesh_vertadr[mi]+model.mesh_vertnum[mi]];bottom.append((vertices@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g])[:,2].min())
   return np.r_[data.qfrc_inverse[:3],data.qfrc_inverse[3:6]*5,normals*1000,(bottom[0]-bottom[1])*10000,np.array(v[1:])*.01]
  fits=[]
  for pitch in [-.08,0.,.08]:
   for hip in [-.15,0.,.15]:
    initial=[start,hip,0.,0.,0.,pitch,0.,0.,0.,0.,0.]
    fit=least_squares(residual,initial,bounds=([base[2]-.15,-.3,-.15,-.25,-.1,-.15,-.05,-.05,-.05,-.05,-.1],[base[2]+.06,.3,.15,.25,.1,.15,.05,.05,.05,.05,.1]),max_nfev=500,jac='3-point',diff_step=1e-5,ftol=1e-11,xtol=1e-11,gtol=1e-9)
    fits.append(fit)
  fit=min(fits,key=lambda f:np.linalg.norm(residual(f.x)[:-10]))
  qeq=set_pose(fit.x);wrench=data.qfrc_inverse[:6].copy();tau=data.qfrc_inverse[6:].copy();kp=np.array([PROFILE['stiffness'][n] for n in names]);effort=np.array([PROFILE['effort_limit'][n] for n in names]);target=qeq[7:]+tau/kp
  jids=np.array(model.actuator_trnid[:,0],dtype=int);limits=model.jnt_range[jids];limit_violation=np.maximum(limits[:,0]-qeq[7:],0)+np.maximum(qeq[7:]-limits[:,1],0)
  contacts=[dict(geom1=model.geom(int(data.contact[i].geom1)).name,geom2=model.geom(int(data.contact[i].geom2)).name,distance=float(data.contact[i].dist),position=data.contact[i].pos.tolist()) for i in range(data.ncon)]
  from scipy.spatial import ConvexHull
  support_points=np.array([c['position'][:2] for c in contacts if 'foot_' in c['geom2'] or 'foot_' in c['geom1']]);hull=ConvexHull(support_points);com=data.subtree_com[int(model.jnt_bodyid[0]),:2];margin=float(np.min(-(hull.equations[:,:2]@com+hull.equations[:,2])/np.linalg.norm(hull.equations[:,:2],axis=1)))
  report=dict(sole_pitch_roll_correction_rad=sole_fit.x.tolist(),support_margin_m=margin,entry_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),max_joint_limit_violation_rad=float(limit_violation.max()),equilibrium_contacts=contacts,source_checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),selected_pose=str(a.pose),solver_success=bool(fit.success),solver_variables=fit.x.tolist(),base_inverse_wrench=wrench.tolist(),max_effort_fraction=float(np.max(np.abs(tau)/effort)),joint_names=names,equilibrium_qpos=qeq.tolist(),pd_position_target=target.tolist(),scope='Candidate generated standing reference; no deployment controller change or policy success claim.')
  (a.out/'equilibrium.json').write_text(json.dumps(report,indent=2)+'\n')
  if margin<.005 or limit_violation.max()>1e-6 or np.linalg.norm(wrench[:3])>1 or np.linalg.norm(wrench[3:])>.2 or np.max(np.abs(tau)/effort)>.8:
   report.update(passed=False,reason='static_force_or_torque_residual');(a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');env.close();print(json.dumps(report));return
 # Reference validation deliberately starts at the solved pose. Cold-start policy
 # acceptance remains separate and continues to use the original initial state.
 from scipy.linalg import solve_discrete_are
 data.qpos[:]=qeq;data.qvel[:]=0;data.ctrl[:]=target;mujoco.mj_forward(model,data)
 nd=2*model.nv+model.na;A=np.zeros((nd,nd));B=np.zeros((nd,model.nu));mujoco.mjd_transitionFD(model,data,1e-6,True,A,B,None,None)
 steps=round(env.step_dt/model.opt.timestep);Ad=np.eye(nd);Bd=np.zeros_like(B)
 for _ in range(steps):Bd=A@Bd+B;Ad=A@Ad
 qw=np.r_[np.full(3,100.),np.full(3,1000.),np.full(model.nv-6,5.)];vw=np.r_[np.full(6,10.),np.full(model.nv-6,.1)];Q=np.diag(np.r_[qw,vw]);R=np.eye(model.nu)*.5
 P=solve_discrete_are(Ad,Bd,Q,R);K=np.linalg.solve(R+Bd.T@P@Bd,Bd.T@P@Ad);rho=float(np.max(abs(np.linalg.eigvals(Ad-Bd@K))))
 report.update(reference_controller='Joint motors with full-state LQR feedback, data generation only',linear_closed_loop_spectral_radius=rho)
 print('closed_loop_spectral_radius',rho,flush=True)
 (a.out/'feedback-diagnostic.json').write_text(json.dumps(dict(spectral_radius=rho,eigenvalues=[[float(v.real),float(v.imag)] for v in np.linalg.eigvals(Ad-Bd@K)]),indent=2))
 if rho>1+1e-6:raise RuntimeError('Reference feedback has an unstable linear mode')
 np.savez(a.out/'reference_feedback.npz',qeq=qeq,target=target,K=K,A=Ad,B=Bd)
 s=cmd.state_io.read();s[:,:NQ]=torch.tensor(qeq,device='cuda',dtype=s.dtype);s[:,NQ:]=0;cmd.state_io.write(torch.arange(2,device='cuda'),s)
 act=env.action_manager.get_term('joint_pos');actions=(torch.tensor(target,device='cuda',dtype=s.dtype)-act._offset)/act._scale
 frames=[];trace=[];demo_observations=[];demo_actions=[];failed=torch.zeros(2,dtype=torch.bool,device='cuda')
 try:
  with torch.no_grad():
   for step in range(1000):
    obs_before=nxt['policy'].clone() if step else None
    current=cmd.state_io.read().cpu().numpy();controls=[]
    for state in current:
     tangent=np.empty(model.nv);mujoco.mj_differentiatePos(model,tangent,1.,qeq,state[:NQ]);delta=np.r_[tangent,state[NQ:]];controls.append(target-K@delta)
    actions=(torch.tensor(np.array(controls),device='cuda',dtype=s.dtype)-act._offset)/act._scale
    nxt,_,term,tout,_=env.step(actions);failed|=term|tout;state=cmd.state_io.read();robot=cmd.robot
    trace.append(dict(time_s=(step+1)*env.step_dt,velocity=measured_velocity(state).cpu().numpy(),arm_square=state[:,NQ+6:][:,arms].square().mean(-1).cpu().numpy(),position=state[:,:3].cpu().numpy(),tilt=projected_gravity(state[:,3:7])[:,:2].norm(dim=-1).cpu().numpy()))
    if step>=100:demo_observations.append(obs_before.cpu().clone());demo_actions.append(actions.cpu().clone())
    if step>=100:frames.append(dict(joint_pos=robot.data.joint_pos[0].cpu().numpy().copy(),joint_vel=robot.data.joint_vel[0].cpu().numpy().copy(),body_pos_w=(robot.data.body_link_pos_w[0]-env.scene.env_origins[0]).cpu().numpy().copy(),body_quat_w=robot.data.body_link_quat_w[0].cpu().numpy().copy(),body_lin_vel_w=robot.data.body_link_lin_vel_w[0].cpu().numpy().copy(),body_ang_vel_w=robot.data.body_link_ang_vel_w[0].cpu().numpy().copy()))
    if failed.any():break
  rows=trace[100:];passed=bool(rows) and not failed.any();v=np.array([r['velocity'] for r in rows]);pos=np.array([r['position'] for r in rows]);arm=np.array([r['arm_square'] for r in rows]);metrics={} if not rows else dict(max_mean_planar=float(np.linalg.norm(v[:,:,:2],axis=-1).mean(0).max()),max_mean_abs_yaw=float(np.abs(v[:,:,2]).mean(0).max()),max_arm_rms=float(np.sqrt(arm.mean(0)).max()),max_xy_excursion=float(np.linalg.norm(pos[:,:,:2]-pos[:1,:,:2],axis=-1).max()))
  if rows:passed=passed and metrics['max_mean_planar']<.01 and metrics['max_mean_abs_yaw']<.02 and metrics['max_arm_rms']<.05 and metrics['max_xy_excursion']<.02
  report.update(passed=bool(passed),failed=failed.cpu().tolist(),physics_seconds=len(trace)*env.step_dt,metrics=metrics)
  if passed:
   torch.save(dict(observations=torch.stack(demo_observations),actions=torch.stack(demo_actions),source='20s physically validated feedback hold; data generation only'),a.out/'standing_demonstrations.pt')
   body_names=np.load(a.dataset/'walk.npz')['body_names'];assert list(body_names)==robot.body_names
   payload={k:np.stack([f[k] for f in frames[:500]]) for k in frames[0]};payload.update(fps=np.array([1/env.step_dt]),joint_names=np.array(names),body_names=body_names,source=np.array('Generated inverse-dynamics equilibrium + joint-motor LQR, 20-second physical hold, unchanged X2 Ultra assets; reference data only.'))
   path=a.out/'stand_bent.npz';np.savez_compressed(path,**payload);report.update(reference=str(path),reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reference_frames=len(payload['joint_pos']))
  (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
 finally:env.close()
if __name__=='__main__':main()
