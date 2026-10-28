"""Polish millimetre foot alignment in an existing reference, preserving its COM.

Only root translation and leg joints change; no controller or physical model is
modified. Apply tiny endpoint corrections smoothly during the existing steps.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--repair-script',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('reference_geometry_snapshot',a.repair_script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.data.physics_motion import differentiate_qpos,export_qpos_motion
    from supermimic.robots.assets import get_robot_cfg
    from supermimic.robots.dimensions import PROFILE
    m=load_g1_model();geo=module.Geometry(m)
    z=np.load(a.source/'qpos.npz');original=z['qpos'].copy();q=original.copy();fps=float(z['fps'][0])
    report=json.load(open(a.source/'result.json'));start=report['transition_frame'];duration=report['transition_seconds'];end=start+round(duration*fps)
    names=list(z['joint_names']);np.testing.assert_array_equal(names,PROFILE['joint_names'])
    columns=np.array([m.jnt_qposadr[m.joint(s+'_'+j+'_joint').id] for s in ['left','right'] for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']])
    factor=get_robot_cfg().articulation.soft_joint_pos_limit_factor;ranges=m.jnt_range[1:];centre=ranges.mean(1);half=np.diff(ranges,axis=1)[:,0]*factor/2
    geo.update(q[-1]);heading=Rotation.from_quat(q[-1,[4,5,6,3]]).as_euler('xyz')[2];basis=Rotation.from_euler('z',heading).as_matrix()
    endpoint={};extents={};yaw_delta={}
    for side in geo.feet:
        pos,rot=geo.foot(side);yaw_delta[side]=float(heading-rot.as_euler('xyz')[2]);target_rot=Rotation.from_euler('z',yaw_delta[side])*rot
        local=rot.inv().apply(geo.vertices(side)-pos);world=target_rot.apply(local)+pos;v=world@basis
        endpoint[side]=(pos,target_rot);extents[side]=[float(v[:,0].max()),float((v[:,1].max()+v[:,1].min())/2)]
    toe=np.mean([v[0] for v in extents.values()]);middle=np.mean([v[1] for v in extents.values()]);delta={}
    for side in geo.feet:
        delta[side]=basis@np.array([toe-extents[side][0],middle+(.12 if side=='left' else -.12)-extents[side][1],0.])
    fits=[];com_error=[]
    for i in range(start+1,end+1):
        base=original[i];geo.update(base);com=geo.data.subtree_com[geo.root].copy();t=(i-start)/fps;targets={}
        for side in geo.feet:
            u=t/report['right_landing_seconds'] if side=='right' else (t-duration*.55)/(duration*.3)
            w=float(module.smoothstep(u));pos,rot=geo.foot(side)
            targets[side]=(pos+w*delta[side],Rotation.from_euler('z',w*yaw_delta[side])*rot)
        initial=np.r_[base[:3],base[columns]];ids=columns-7
        low=np.r_[base[:3]-.01,np.maximum(base[columns]-.03,centre[ids]-half[ids]+1e-8)]
        high=np.r_[base[:3]+.01,np.minimum(base[columns]+.03,centre[ids]+half[ids]-1e-8)]
        def pose(x):
            result=base.copy();result[:3]=x[:3];result[columns]=x[3:];return result
        def residual(x):
            geo.update(pose(x));r=[]
            for side in geo.feet:
                pos,rot=geo.foot(side);target_p,target_r=targets[side]
                r.extend((pos-target_p)*10000);r.extend((target_r.inv()*rot).as_rotvec()*2000)
            r.extend((geo.data.subtree_com[geo.root]-com)*10000);r.extend((x-initial)*.01)
            return np.asarray(r)
        fit=least_squares(residual,np.clip(initial,low,high),bounds=(low,high),max_nfev=100,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        q[i]=pose(fit.x);geo.update(q[i]);fits.append(float(np.linalg.norm(residual(fit.x)[:12])));com_error.append(float(np.linalg.norm(geo.data.subtree_com[geo.root]-com)))
    q[end+1:]=q[end];v=differentiate_qpos(m,q,fps)
    np.savez(a.out/'qpos.npz',qpos=q,qvel=v,fps=np.array([fps]),joint_names=np.array(names))
    export_qpos_motion(m,q,a.out/'motion-fk.npz',fps,qvel=v,source=a.source/'webster.npz')
    with np.load(a.out/'motion-fk.npz') as f:arrays={k:f[k].copy() for k in f.files}
    np.savez(a.out/'webster.npz',**arrays,diagnostic_only=np.array(True),approved_for_training=np.array(False))
    heights=[]
    for pose_i in q:geo.update(pose_i);heights.append(geo.height())
    heights=np.array(heights);speed=np.array([PROFILE['velocity_limit'][n] for n in names]);lo,hi=ranges.T
    result={'configuration':{k:str(v) for k,v in vars(a).items()},'source_sha256':hashlib.sha256((a.source/'webster.npz').read_bytes()).hexdigest(),
        'output_sha256':hashlib.sha256((a.out/'webster.npz').read_bytes()).hexdigest(),'frames':len(q),'fps':fps,'transition_frame':start,'transition_seconds':duration,
        'right_landing_seconds':report['right_landing_seconds'],'root_stopping_seconds':report['root_stopping_seconds'],
        'endpoint_alignment_delta_m':{s:v.tolist() for s,v in delta.items()},'endpoint_heading_delta_rad':yaw_delta,
        'maximum_joint_polish_rad':float(abs(q[:,7:]-original[:,7:]).max()),'maximum_root_polish_m':float(np.linalg.norm(q[:,:3]-original[:,:3],axis=1).max()),
        'maximum_com_polish_error_m':max(com_error),'maximum_ik_scaled_residual':max(fits),
        'hard_violation_rad':float(np.maximum(np.maximum(lo-q[:,7:],q[:,7:]-hi),0).max()),'soft_violation_rad':float(np.maximum(abs(q[:,7:]-centre)-half,0).max()),
        'maximum_adjacent_joint_speed_ratio':float((abs(np.diff(q[:,7:],axis=0))*fps/speed).max()),'minimum_heights_m':heights.min(0).tolist(),
        'transition_maximum_lower_foot_height_m':float(heights[start:,:2].min(axis=1).max()),'approved_for_training':False,'dynamic_tracking_verified':False,
        'limitations':'Reference alignment and kinematics only; no controller or physical model change.'}
    result['kinematic_screen_passed']=bool(result['hard_violation_rad']<1e-9 and result['soft_violation_rad']<1e-9 and result['maximum_adjacent_joint_speed_ratio']<=1 and heights[:,:2].min()>-.004 and heights[:,2].min()>.005 and result['transition_maximum_lower_foot_height_m']<.006 and max(com_error)<1e-5 and max(fits)<.05)
    (a.out/'result.json').write_text(json.dumps(result,indent=2));(a.out/'as-run.py').write_bytes(Path(__file__).read_bytes());shutil.copyfile(a.source/'contacts.json',a.out/'contacts.json');(a.out/'manifest_filtered.csv').write_text(f'name,frames,keep,reason\nwebster,{len(q)},0,reference preview; physical validation separate\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
