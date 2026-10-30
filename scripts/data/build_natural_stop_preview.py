"""Generate an unapproved, contact-aware transition to a natural stance seed."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation,Slerp


def main():
    p=argparse.ArgumentParser();p.add_argument('--walking',type=Path,required=True);p.add_argument('--pose',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--stop-frames',type=int,default=100);p.add_argument('--hold-frames',type=int,default=150);p.add_argument('--match-entry-velocity',action='store_true',help='Match the preceding root, joint and foot velocities at transition entry');p.add_argument('--soft-entry',action='store_true',help='Gradually introduce contact fitting to avoid an initial IK jump');a=p.parse_args()
    if a.out.exists() or a.stop_frames<20 or a.hold_frames<2:raise ValueError('Invalid output or duration')
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.data.physics_motion import export_qpos_motion
    m=load_g1_model();d=mujoco.MjData(m);meta=json.loads((a.walking/'manifest.json').read_text());release=meta['release_frame'];lead=meta['parameters']['release_lead'];startidx=release+lead
    with np.load(a.walking/'candidate.npz') as z:
        root=list(z['body_names']).index('pelvis');fps=float(z['fps'][0]);prefix=np.c_[z['body_pos_w'][:startidx+1,root],z['body_quat_w'][:startidx+1,root],z['joint_pos'][:startidx+1]].astype(float);names=list(z['joint_names'])
    end=np.load(a.pose)['qpos'].astype(float);start=prefix[-1].copy()
    start_yaw=Rotation.from_quat(start[[4,5,6,3]]).as_euler('xyz')[2]
    end_rotation=Rotation.from_quat(end[[4,5,6,3]])
    end_yaw=end_rotation.as_euler('xyz')[2]
    # The calibrated pose can already have a world heading: rotate only the
    # difference, rather than applying the walking heading a second time.
    end[3:7]=(Rotation.from_euler('z',start_yaw-end_yaw)*end_rotation).as_quat()[[3,0,1,2]]
    final_yaw=Rotation.from_quat(end[[4,5,6,3]]).as_euler('xyz')[2]
    if abs(np.arctan2(np.sin(final_yaw-start_yaw),np.cos(final_yaw-start_yaw)))>1e-6:
        raise ValueError('Standing heading does not match walking heading')
    feet=[m.body(s+'_ankle_roll_link').id for s in ('left','right')];geoms=[]
    for foot in feet:
        gg=[]
        for g in range(m.ngeom):
            if m.geom_bodyid[g]!=foot or not(m.geom_contype[g] or m.geom_conaffinity[g]) or m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
            k=m.geom_dataid[g];gg.append((g,m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k]+m.mesh_vertnum[k]].copy()))
        if not gg:raise ValueError('Missing foot collision mesh')
        geoms.append(gg)
    def solve_pose(q):
        d.qpos[:]=q;mujoco.mj_kinematics(m,d)
        h=np.array([min(float((v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])[:,2].min()) for g,v in gg) for gg in geoms])
        return d.xpos[feet].copy(),d.xmat[feet].copy().reshape(2,3,3),h
    previous_foot=solve_pose(prefix[-2])[0]
    p0,r0,h0=solve_pose(start);support=int(np.argmin(h0));end[:2]=start[:2];p1,r1,h1=solve_pose(end);end[:2]+=p0[support,:2]-p1[support,:2];p1,r1,h1=solve_pose(end)
    leg=np.array([i for i,n in enumerate(names) if any(x in n for x in ('hip','knee','ankle'))]);jids=np.array([m.joint(names[i]).id for i in leg]);lo=m.jnt_range[jids,0]+1e-5;hi=m.jnt_range[jids,1]-1e-5
    rotations=[Slerp([0,1],Rotation.from_matrix(np.stack([r0[s],r1[s]]))) for s in range(2)];rootrot=Slerp([0,1],Rotation.from_quat(np.stack([start[[4,5,6,3]],end[[4,5,6,3]]])))
    rows=[];errs=[];prev=start[7+leg].copy()
    duration=a.stop_frames/fps
    if a.match_entry_velocity:
        v0=(start-prefix[-2])*fps
        pose_curve=CubicHermiteSpline([0.,duration],np.stack([start,end]),np.stack([v0,np.zeros_like(v0)]),axis=0)
        foot_v0=(p0-previous_foot)*fps
        foot_curve=CubicHermiteSpline([0.,duration],np.stack([p0,p1]),np.stack([foot_v0,np.zeros_like(foot_v0)]),axis=0)
    for i in range(1,a.stop_frames):
        t=i/a.stop_frames;u=3*t*t-2*t*t*t
        raw=pose_curve(t*duration) if a.match_entry_velocity else (1-u)*start+u*end
        raw[3:7]=rootrot([u]).as_quat()[0,[3,0,1,2]]
        wanted=foot_curve(t*duration) if a.match_entry_velocity else (1-u)*p0+u*p1
        wanted[1-support,2]+=.025*np.sin(np.pi*u)**2
        footrot=[rotations[s]([u]).as_matrix()[0] for s in range(2)]
        init=np.r_[raw[:3],np.clip(prev,lo,hi)];lower=np.r_[raw[:3]-.08,lo];upper=np.r_[raw[:3]+.08,hi]
        def residual(x):
            q=raw.copy();q[:3]=x[:3];q[7+leg]=x[3:];pnow,rnow,h=solve_pose(q)
            orient=np.concatenate([Rotation.from_matrix(footrot[s].T@rnow[s]).as_rotvec() for s in range(2)])
            ground=np.array([h[s] if s==support else min(h[s],0.) for s in range(2)])
            entry_weight=20*(1-u)**3 if a.soft_entry else 0.
            return np.r_[50*(pnow-wanted).ravel(),3*orient,100*ground,(.15+entry_weight)*(x[3:]-raw[7+leg]),.03*(x[3:]-prev),(.5+entry_weight)*(x[:3]-raw[:3])]
        x=least_squares(residual,init,bounds=(lower,upper),max_nfev=25,ftol=1e-6,xtol=1e-6,gtol=1e-6).x
        residual(x);raw[:3]=x[:3];raw[7+leg]=x[3:];prev=x[3:];rows.append(raw);errs.append(np.linalg.norm(d.xpos[feet]-wanted,axis=1))
    poses=np.vstack([prefix,np.asarray(rows),np.repeat(end[None],a.hold_frames+1,axis=0)])
    a.out.mkdir(parents=True);export_qpos_motion(m,poses,a.out/'natural-stop.npz',fps,source=str(a.walking/'candidate.npz'))
    export_qpos_motion(m,np.repeat(end[None],a.hold_frames+1,axis=0),a.out/'natural-stand.npz',fps,source=str(a.pose))
    delta=np.sqrt(np.mean(np.diff(poses[:,7:],axis=0)**2,axis=1));info={'release_frame':release,'brake_start_frame':startidx,'standing_frame':len(prefix)+len(rows),'frames':len(poses),'source_reference':str(a.walking/'candidate.npz'),'standing_pose':str(a.pose),'standing_pose_sha256':hashlib.sha256(a.pose.read_bytes()).hexdigest(),'support_side':['left','right'][support],'entry_velocity_matched':a.match_entry_velocity,'stop_frames':a.stop_frames,'max_ankle_target_error_m':float(np.max(errs)),'max_adjacent_joint_rms_rad':float(delta.max()),'stop_sha256':hashlib.sha256((a.out/'natural-stop.npz').read_bytes()).hexdigest(),'scope':'Kinematic preview only; physical hold and complete transition not validated','approved_for_training':False}
    (a.out/'manifest.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))
if __name__=='__main__':main()
