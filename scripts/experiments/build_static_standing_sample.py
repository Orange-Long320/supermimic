"""Add a statically feasible standing reference; active policy tracking is pending."""
from pathlib import Path
import argparse,json,hashlib

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['checkpoint','dataset','equilibrium','out']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();assert not a.out.exists()
 import torch,numpy as np,mujoco
 from scipy.spatial import ConvexHull
 from supermimic.envs import SuperMimicEnvCfgArgs,make_env
 from supermimic.locomotion.config import VelocityCfg
 from supermimic.robots.dimensions import PROFILE
 cfg=torch.load(a.checkpoint,map_location='cpu',weights_only=False)['config'];e=json.loads(a.equilibrium.read_text());env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(a.dataset/'walk.npz')],num_envs=2,obs_noise=False,reset_noise=False,domain_randomization=False,velocity=VelocityCfg(**cfg['velocity'])),device='cuda:0');env.reset()
 try:
  m=env.sim.mj_model;d=mujoco.MjData(m);q=np.array(e['equilibrium_qpos']);d.qpos[:]=q;d.qvel[:]=0;d.qacc[:]=0;mujoco.mj_inverse(m,d)
  root_wrench=d.qfrc_inverse[:6].copy();joint_names=[m.joint(int(i)).name.split('/')[-1] for i in m.actuator_trnid[:,0]];assert joint_names==PROFILE['joint_names']
  limits=m.jnt_range[m.actuator_trnid[:,0]];violation=np.maximum(limits[:,0]-q[7:],0)+np.maximum(q[7:]-limits[:,1],0);effort=np.array([PROFILE['effort_limit'][n] for n in joint_names]);effort_fraction=float(np.max(abs(d.qfrc_inverse[6:])/effort))
  points=[];feet=set();penetration=0.
  for c in d.contact[:d.ncon]:
   names=[m.geom(int(g)).name for g in [c.geom1,c.geom2]];foot=next((n for n in names if '/foot_' in n),None)
   if foot:points.append(c.pos[:2].copy());feet.add(foot);penetration=max(penetration,-float(c.dist))
  hull=ConvexHull(np.array(points));com=d.subtree_com[int(m.jnt_bodyid[0]),:2];margin=float(np.min(-(hull.equations[:,:2]@com+hull.equations[:,2])/np.linalg.norm(hull.equations[:,:2],axis=1)))
  assert len(feet)==2 and margin>=.005 and penetration<=.003 and violation.max()<1e-6 and effort_fraction<.8 and np.linalg.norm(root_wrench[:3])<1 and np.linalg.norm(root_wrench[3:])<.2
  src=np.load(a.dataset/'walk.npz');body_names=src['body_names'];ids=[m.body('robot/'+str(n)).id for n in body_names]
  n=500;repeat=lambda x:np.broadcast_to(x,(n,)+x.shape).copy().astype(np.float32)
  payload=dict(fps=np.array([50.]),joint_pos=repeat(q[7:]),joint_vel=np.zeros((n,len(joint_names)),np.float32),body_pos_w=repeat(d.xpos[ids]),body_quat_w=repeat(d.xquat[ids]),body_lin_vel_w=np.zeros((n,len(ids),3),np.float32),body_ang_vel_w=np.zeros((n,len(ids),3),np.float32),joint_names=np.array(joint_names),body_names=body_names,source=np.array('Generated X2 Ultra static equilibrium with bent arms; geometry and static forces verified; learned active tracking pending.'))
  a.out.mkdir();manifest=json.loads((a.dataset/'manifest.json').read_text());manifest['parent_manifest_sha256']=hashlib.sha256((a.dataset/'manifest.json').read_bytes()).hexdigest();manifest['standing_validation']=dict(stage='static_feasibility_passed_active_tracking_pending',source_equilibrium=str(a.equilibrium),root_inverse_wrench=root_wrench.tolist(),support_margin_m=margin,feet=sorted(feet),max_foot_penetration_m=penetration,max_joint_limit_violation_rad=float(violation.max()),max_effort_fraction=effort_fraction,reference_controller='none; static goal for learned active balance',physical_policy_accepted=False)
  for clip in manifest['clips']:(a.out/(clip['name']+'.npz')).symlink_to((a.dataset/(clip['name']+'.npz')).resolve())
  out=a.out/'stand_bent.npz';np.savez_compressed(out,**payload);manifest['clips'].append(dict(name='stand_bent',frames=n,fps=50.,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),source_kind='generated_static_reference',validation='static_feasibility_passed_active_tracking_pending'))
  (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n');np.savez(a.out/'standing_inverse_target.npz',qpos=q,position_control_target=np.array(e['pd_position_target']))
  print(json.dumps(manifest['standing_validation'],indent=2))
 finally:env.close()
if __name__=='__main__':main()
