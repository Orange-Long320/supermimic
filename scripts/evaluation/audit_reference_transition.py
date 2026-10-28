"""Independent FK audit of the repaired reference, never a physics rollout."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from supermimic.data.motion_convert import load_g1_model
p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--endpoint-polish-limit',type=float,default=0.);a=p.parse_args()
z=np.load(a.candidate/'qpos.npz');q=z['qpos'];v=z['qvel'];old=np.load(a.baseline)['qpos'];m=load_g1_model();d=mujoco.MjData(m);report=json.load(open(a.candidate/'result.json'));start=report['transition_frame'];feet={s:m.body(s+'_ankle_roll_link').id for s in ['left','right']};positions=[]
for pose in q:
 d.qpos[:]=pose;mujoco.mj_forward(m,d);positions.append([d.xpos[b].copy() for b in feet.values()])
positions=np.array(positions);horizontal=positions[start:,1,:2];direction=np.sign(horizontal[-1]-horizontal[0]);backtrack=np.maximum(-np.diff(horizontal,axis=0)*direction,0).sum(axis=0);overshoot=np.maximum(np.maximum(horizontal-np.maximum(horizontal[0],horizontal[-1]),np.minimum(horizontal[0],horizontal[-1])-horizontal),0).max(axis=0)
d.qpos[:]=q[-1];mujoco.mj_forward(m,d);headings=[Rotation.from_matrix(d.xmat[b].reshape(3,3)).as_euler('xyz')[2] for b in feet.values()];heading_error=abs((headings[0]-headings[1]+np.pi)%(2*np.pi)-np.pi);basis=Rotation.from_euler('z',Rotation.from_quat(q[-1,[4,5,6,3]]).as_euler('xyz')[2]).as_matrix();extent={}
for s,b in feet.items():
 points=[]
 for g in range(m.ngeom):
  if m.geom_bodyid[g]!=b or not(m.geom_contype[g] or m.geom_conaffinity[g]):continue
  k=m.geom_dataid[g];adr,n=m.mesh_vertadr[k],m.mesh_vertnum[k];local=m.mesh_vert[adr:adr+n];points.append((local@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])@basis)
 points=np.concatenate(points);extent[s]=dict(toe=float(points[:,0].max()),lateral=float((points[:,1].min()+points[:,1].max())/2))
result=dict(reference_only=True,physics_verified=False,source_sha256=hashlib.sha256((a.candidate/'webster.npz').read_bytes()).hexdigest(),prefix_pose_max_difference=float(abs(q[:start+1]-old[:start+1]).max()),fixed_endpoint_max_difference=float(abs(q[-1]-old[-1]).max()),last_100_pose_difference=float(abs(q[-100:]-q[-1]).max()),last_100_velocity_max=float(abs(v[-100:]).max()),right_horizontal_backtrack_xy_m=backtrack.tolist(),right_horizontal_endpoint_overshoot_xy_m=overshoot.tolist(),toe_alignment_error_m=abs(extent['left']['toe']-extent['right']['toe']),stance_width_m=extent['left']['lateral']-extent['right']['lateral'],foot_heading_difference_rad=float(heading_error),kinematic_screen_passed=report['kinematic_screen_passed'])
result['reference_repair_passed']=bool(result['kinematic_screen_passed'] and result['prefix_pose_max_difference']<1e-10 and result['fixed_endpoint_max_difference']<=max(1e-10,a.endpoint_polish_limit) and abs(q[-1,3:7]-old[-1,3:7]).max()<1e-10 and abs(q[-1,19:]-old[-1,19:]).max()<1e-10 and np.linalg.norm(q[-1,:3]-old[-1,:3])<.005 and result['last_100_pose_difference']<1e-12 and result['last_100_velocity_max']<1e-10 and backtrack.max()<.002 and overshoot.max()<.001 and result['foot_heading_difference_rad']<.001 and result['toe_alignment_error_m']<.001 and abs(result['stance_width_m']-.24)<.001)
with (a.candidate/'reference-audit.json').open('x') as f:json.dump(result,f,indent=2)
(a.candidate/'reference-audit-as-run.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(result),flush=True)
