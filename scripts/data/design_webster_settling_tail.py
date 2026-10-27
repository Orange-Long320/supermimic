"""Screen smooth landing-to-standing references; no dynamic approval is granted."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def right_jacobian(vector):
    angle=np.linalg.norm(vector)
    cross=np.array([[0.,-vector[2],vector[1]],[vector[2],0.,-vector[0]],[-vector[1],vector[0],0.]])
    if angle<1e-5:return np.eye(3)-cross/2+cross@cross/6
    return np.eye(3)-(1-np.cos(angle))/angle**2*cross+(angle-np.sin(angle))/angle**3*(cross@cross)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--standing-trajectory',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--start-frames',type=int,nargs='+',default=[65,75,83])
    parser.add_argument('--durations',type=float,nargs='+',default=[.4,.6,1.,1.5])
    args=parser.parse_args()
    if min(args.durations)<=0:parser.error('Positive durations required')
    args.out.mkdir(parents=True,exist_ok=False)
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.data.physics_motion import export_qpos_motion
    from supermimic.robots.dimensions import PROFILE
    model=load_g1_model();data=mujoco.MjData(model)
    with np.load(args.source) as z:source={k:z[k].copy() for k in z.files}
    with np.load(args.standing_trajectory) as z:end=z['state'][-1,:model.nq].copy()
    fps=float(source['fps'].reshape(-1)[0]);root=source['body_names'].tolist().index(model.body(int(model.jnt_bodyid[0])).name)
    low,high=model.jnt_range[1:].T
    limits=np.array([PROFILE['velocity_limit'][n] for n in PROFILE['joint_names']])
    fields=['joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']
    meshes=[]
    for g in range(model.ngeom):
        if not (model.geom_contype[g] or model.geom_conaffinity[g]):continue
        mesh=int(model.geom_dataid[g]);assert mesh>=0
        a,n=model.mesh_vertadr[mesh],model.mesh_vertnum[mesh]
        body=model.body(int(model.geom_bodyid[g])).name
        group='left' if body=='left_ankle_roll_link' else 'right' if body=='right_ankle_roll_link' else 'nonfoot'
        meshes.append((g,model.mesh_vert[a:a+n].copy(),group))
    reports=[]
    for frame in args.start_frames:
        if not 0<=frame<len(source['joint_pos']):raise ValueError('Start frame outside source')
        initial=np.r_[source['body_pos_w'][frame,root],source['body_quat_w'][frame,root],source['joint_pos'][frame]].astype(float)
        rotation=Rotation.from_quat(initial[[4,5,6,3]])
        velocity=np.r_[source['body_lin_vel_w'][frame,root],rotation.inv().apply(source['body_ang_vel_w'][frame,root]),source['joint_vel'][frame]]
        delta=np.zeros(model.nv);mujoco.mj_differentiatePos(model,delta,1.,initial,end)
        for requested in args.durations:
            count=max(2,round(requested*fps));duration=count/fps
            name=f'frame{frame:03d}-duration{count:03d}';folder=args.out/name;folder.mkdir()
            first=duration*velocity;third=10*delta-6*first;fourth=-15*delta+8*first;fifth=6*delta-3*first
            poses=[];velocities=[];heights=[]
            for i in range(count+1+round(2*fps)):
                u=min(i/count,1.);offset=first*u+third*u**3+fourth*u**4+fifth*u**5
                speed=(first+3*third*u*u+4*fourth*u**3+5*fifth*u**4)/duration
                speed[3:6]=right_jacobian(offset[3:6])@speed[3:6]
                if i>=count:speed[:]=0
                pose=initial.copy();mujoco.mj_integratePos(model,pose,offset,1.)
                poses.append(pose);velocities.append(speed)
                data.qpos[:]=pose;mujoco.mj_kinematics(model,data)
                heights.append([min(float((vertices@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g])[:,2].min())
                                    for g,vertices,kind in meshes if kind==group) for group in ['left','right','nonfoot']])
            poses=np.asarray(poses);velocities=np.asarray(velocities);heights=np.asarray(heights)
            export_qpos_motion(model,poses,folder/'tail-fk.npz',fps,qvel=velocities,source=args.source)
            with np.load(folder/'tail-fk.npz') as z:tail={k:z[k].copy() for k in z.files}
            for key in ['body_names','joint_names']:np.testing.assert_array_equal(tail[key],source[key])
            result={k:v.copy() for k,v in source.items()}
            for key in fields:
                result[key]=np.concatenate([source[key][:frame+1],tail[key][1:]])
                np.testing.assert_array_equal(result[key][:frame+1],source[key][:frame+1])
            result['diagnostic_only']=np.array(True)
            np.savez(folder/'webster.npz',**result)
            hard=float(np.maximum(np.maximum(low-poses[:,7:],poses[:,7:]-high),0).max())
            speed_ratio=float((abs(velocities[:,6:])/limits).max())
            item={'name':name,'start_frame':frame,'duration_s':duration,'hard_violation_rad':hard,
                  'joint_speed_ratio':speed_ratio,'minimum_foot_height_m':float(heights[:,:2].min()),
                  'minimum_nonfoot_height_m':float(heights[:,2].min()),
                  'maximum_lower_sole_height_m':float(heights[:,:2].min(axis=1).max()),
                  'endpoint_pose_error':float(abs(poses[-1]-end).max()),'prefix_unchanged':True,
                  'approved_for_training':False}
            item['kinematic_screen_passed']=bool(hard<1e-9 and speed_ratio<=1 and heights[:,:2].min()>-.004
                                                and heights[:,2].min()>.005 and heights[:,:2].min(axis=1).max()<.02)
            (folder/'DIAGNOSTIC_ONLY.json').write_text(json.dumps(item,indent=2));reports.append(item)
    report={'candidates':reports,'approved_for_training':False,'purpose':__doc__,
            'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
            'standing_trajectory_sha256':hashlib.sha256(args.standing_trajectory.read_bytes()).hexdigest(),
            'note':'Kinematic interpolation only. Contact forces, slip and dynamic tracking remain unverified.'}
    (args.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    (args.out/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
