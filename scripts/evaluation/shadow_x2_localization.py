"""Finite CPU shadow inference from recorded data. No ROS, sockets or commands.

The 10 Hz LIO pelvis pose is held, with its age explicitly checked. This is
not IMU propagation or PC2 online latency validation. Previous actions are
counterfactual: both shadow-feedback and zero-history diagnostics are saved.
"""
from __future__ import annotations
import argparse
from collections import defaultdict, deque
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from supermimic.deploy.pelvis_state import UrdfPelvisMount, pelvis_from_imu, pelvis_angular_velocity, gravity_level_rotation, latest_causal
from supermimic.deploy.torch_policy import TorchScriptPolicy


def stats(a):
    a=np.asarray(a,dtype=float)
    return dict(zip(('min','p50','p95','p99','max'),np.percentile(a,[0,50,95,99,100]).tolist())) if len(a) else None


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','profile','dataset','policy','out'):
        parser.add_argument('--'+key,type=Path,required=True)
    geometry=parser.add_mutually_exclusive_group(required=True)
    geometry.add_argument('--mount-urdf', type=Path,
                          help='Device URDF for pelvis-to-torso FK; policy/anchor FK stays on the training model')
    geometry.add_argument('--legacy-training-mount', action='store_true',
                          help='Explicitly reproduce the previous training-model mount FK for comparison only')
    parser.add_argument('--diagnose-out-of-range',action='store_true',
                        help='logging-only model probe with unchanged out-of-range joints; never validates inputs')
    a=parser.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    # MotionLibrary captures robot dimensions at import time.
    os.environ['SM_ROBOT']='x2_ultra'
    os.environ['SM_ROBOT_SPEC']=str(a.profile.resolve())
    from supermimic.deploy.skill_goal import SkillGoal
    import torch
    torch.set_num_threads(2)
    root=a.root; spec=json.loads(a.profile.read_text()); names=spec['joint_names']; n=len(names)
    if n!=31 or spec['root_body']!='pelvis':raise ValueError('Wrong robot contract')
    if digest(a.policy)!='4424f231fc1896da098b2fca366e8522df6791596638b2b9257ba682520bbe4c':
        raise ValueError('Not the selected 11500 policy')
    if digest(a.dataset/'manifest.json')!='40a2a31c1b84160f2ea6aa26548eb966f2e3c9ec23e10fa46ec1e9b2532b5b7d':
        raise ValueError('Not the selected reference')
    model_path=Path(spec['xml_path'])
    if not model_path.is_absolute():model_path=a.profile.parent/model_path
    model=mujoco.MjModel.from_xml_path(str(model_path)); data=mujoco.MjData(model)
    jids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name) for name in names]
    if min(jids)<0:raise ValueError('Missing model joint')
    qaddr=model.jnt_qposadr[jids]; vaddr=model.jnt_dofadr[jids]
    limits=model.jnt_range[jids]; default=np.array([spec['default_joint_pos'][x] for x in names])
    scale=np.array([spec['action_scale'][x] for x in names])
    torso=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'torso_link')
    prior=json.loads((root/'mount-tf-prior.json').read_text())['tf_static']
    current=json.loads((root/'extracted/tf.json').read_text())
    def transform(row):return np.asarray(row['xyz']),Rotation.from_quat(row['xyzw']).as_matrix()
    ptl,rtl=transform(prior['torso_link -> lidar_chest_front'])
    pli,rli=transform(current['lidar_chest_front -> lidar_imu_chest_front'])
    old=current.get('torso_link -> lidar_chest_front')
    if old is not None and not np.allclose(transform(old)[0],ptl):raise ValueError('Mount TF changed')
    # The moving torso chain uses the training model; the sensor mount uses
    # native TF, never the model's differently oriented inertial-link body.
    def mount(q,qd=None):
        data.qpos[:7]=[0,0,0,1,0,0,0];data.qpos[qaddr]=q
        mujoco.mj_kinematics(model,data)
        rpt=data.xmat[torso].reshape(3,3).copy();ppt=data.xpos[torso].copy()
        rpi=rpt@rtl@rli; ppi=ppt+rpt@(ptl+rtl@pli)
        omega=np.zeros(3)
        if qd is not None:
            mujoco.mj_comPos(model,data)
            jac=np.zeros((3,model.nv));mujoco.mj_jacBody(model,data,None,jac,torso)
            omega=jac[:,vaddr]@qd
        return ppi,rpi,omega
    if a.mount_urdf is not None:
        native_mount=UrdfPelvisMount(a.mount_urdf,names,ptl+rtl@pli,rtl@rli)
        mount=native_mount.mount
    feet=[]
    for name in ('foot_6_collision','foot_12_collision'):
        gid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,name)
        if gid<0:raise ValueError('Original collision foot mesh required')
        mid=int(model.geom_dataid[gid]);start=int(model.mesh_vertadr[mid]);count=int(model.mesh_vertnum[mid])
        if mid<0:raise ValueError('Expected original mesh')
        feet.append((gid,model.mesh_vert[start:start+count].copy()))
    def foot_bottoms(p,r,q):
        data.qpos[:7]=np.r_[p,r.as_quat()[[3,0,1,2]]];data.qpos[qaddr]=q
        mujoco.mj_kinematics(model,data)
        return np.array([np.min((v@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g])[:,2]) for g,v in feet])
    groups=defaultdict(list)
    for line in (root/'capture_latest/robot_state.jsonl').open():
        row=json.loads(line);groups[row['topic']].append(row)
    ages={t:stats([(x['received_ns']-x['header_ns'])*1e-6 for x in rows]) for t,rows in groups.items()}
    sensors=np.load(root/'extracted/sensors.npz',allow_pickle=False);imu=sensors['imu']
    traj=np.loadtxt(root/'replay_latest/trajectory.csv',delimiter=',',skiprows=1,ndmin=2)
    # FAST-LIO initializes rotation to identity and gravity from mean acceleration.
    # Its camera_init frame therefore is not necessarily Z-up on this mounting.
    boot=imu[(imu[:,0]*1e-9>=traj[0,0])&(imu[:,0]*1e-9<traj[0,0]+.5)]
    if len(boot)<50:raise ValueError('Insufficient static gravity initialization')
    boot_indices=np.maximum(0,np.searchsorted(traj[:,0],boot[:,0]*1e-9,side='right')-1)
    up=np.mean(Rotation.from_quat(traj[boot_indices,4:8]).apply(boot[:,5:8]),axis=0)
    leveling=Rotation.from_matrix(gravity_level_rotation(up))
    traj[:,1:4]=leveling.apply(traj[:,1:4])
    traj[:,4:8]=(leveling*Rotation.from_quat(traj[:,4:8])).as_quat()
    avail=np.asarray(json.loads((root/'replay_latest/availability.json').read_text())['capture_clock_available_ns'],dtype=np.int64)
    if len(avail)!=len(traj) or np.any(np.diff(avail)<0):raise ValueError('Invalid replay availability')
    lidar_rows=[{'header_ns':int(r[0]),'received_ns':int(r[1]),'omega':r[2:5]} for r in imu]
    groups['lidar_imu']=lidar_rows
    arrivals={k:np.array([r['received_ns'] for r in rows],dtype=np.int64) for k,rows in groups.items()}
    if any(np.any(np.diff(v)<0) for v in arrivals.values()):raise ValueError('Receive clock moved backwards')
    def select(topic,now,at=None):
        return latest_causal(groups[topic],now,max_age_s=.05,at_ns=at,received_times=arrivals[topic])
    def joints(now,at=None):
        samples=[select('/aima/hal/joint/'+g+'/state',now,at) for g in ('leg','waist','head','arm')]
        byname={};stamps=[]
        for row in samples:
            stamps.append(row['header_ns'])
            for j in row['joints']:
                if j['name'] in byname or j['error_code']:raise ValueError('joint_fault_or_duplicate')
                byname[j['name']]=j
        if set(byname)!=set(names):raise ValueError('joint_name_set')
        q=np.array([byname[x]['position'] for x in names]);qd=np.array([byname[x]['velocity'] for x in names])
        if not np.isfinite(np.r_[q,qd]).all():raise ValueError('nonfinite_joint')
        if not a.diagnose_out_of_range and (np.any(q<limits[:,0]) or np.any(q>limits[:,1])):
            raise ValueError('measured_joint_outside_model_range')
        if max(stamps)-min(stamps)>30_000_000:raise ValueError('joint_skew_over_30ms')
        return q,qd,stamps
    policy=TorchScriptPolicy(a.policy)
    trackers=None; shift=None; origin=None; initial_feet=None; init_q=None
    histories={k:deque(maxlen=10) for k in ('feedback','zero')};previous={k:np.zeros(n) for k in histories}
    outputs=defaultdict(list);deltas=[];timings=[];poses=[];rejected=defaultdict(int);warmup=0
    poseages=[];skews=[];native_gravity_errors=[];step_times=[];previous_frame=None
    input_outside=defaultdict(int)
    first_probe=None
    start=max(min(v[0] for v in arrivals.values()),int(traj[0,0]*1e9))+5_000_000_000
    end=min(v[-1] for v in arrivals.values())
    frames=(a.out/'frames.jsonl').open('x')
    try:
        for now in range(int(start),int(end),20_000_000):
            try:
                q,qd,stamps=joints(now)
                ir=select('lidar_imu',now)
                if max(stamps+[ir['header_ns']])-min(stamps+[ir['header_ns']])>30_000_000:raise ValueError('imu_joint_skew_over_30ms')
                i=np.searchsorted(avail,now,side='right')-1
                while i>=0 and traj[i,0]*1e9>now:i-=1
                if i<0:raise ValueError('no_available_lio')
                poseage=now*1e-9-traj[i,0]
                if poseage>.15:raise ValueError('lio_pose_age_over_150ms')
                past_q,_,_=joints(now,int(traj[i,0]*1e9))
                ppi,rpi,_=mount(past_q)
                pos,rot=pelvis_from_imu(traj[i,1:4],Rotation.from_quat(traj[i,4:8]).as_matrix(),ppi,rpi)
                rot=Rotation.from_matrix(rot)
                _,rpi,chain_omega=mount(q,qd)
                omega=pelvis_angular_velocity(ir['omega'],rpi,chain_omega)
                gravity=rot.inv().apply([0,0,-1])
                if not np.isfinite(np.r_[pos,omega,gravity]).all():raise ValueError('nonfinite_pelvis')
            except ValueError as exc:
                rejected[str(exc)]+=1
                for key in histories:histories[key].clear();previous[key]=np.zeros(n)
                continue
            if shift is None:
                initial_feet=foot_bottoms(pos,rot,q);shift=-float(initial_feet.min());origin=pos.copy();init_q=q.copy()
                initial_pose=np.r_[pos[:2],pos[2]+shift]
                trackers={key:SkillGoal(a.dataset,a.profile,initial_pose,rot.as_quat()[[3,0,1,2]]) for key in histories}
            pos=pos.copy();pos[2]+=shift
            outside=(q<limits[:,0])|(q>limits[:,1])
            for j in np.flatnonzero(outside):input_outside[names[j]]+=1
            poseages.append(poseage*1000);skews.append((max(stamps)-min(stamps))*1e-6)
            poses.append(np.r_[now*1e-9,pos,rot.as_quat(),omega])
            # Native "torso" IMU's frame_id is ambiguous. Compare gravity only;
            # this diagnostic does not assume its yaw or use it as pelvis truth.
            try:
                native=select('/aima/hal/imu/torso/state',now)
                nq=np.asarray(native['orientation_wxyz'])
                ng=Rotation.from_quat(nq[[1,2,3,0]]).inv().apply([0,0,-1])
                native_gravity_errors.append(np.degrees(np.arccos(np.clip(ng@gravity,-1,1))))
            except ValueError:
                pass  # Optional comparison never supplies policy orientation.
            row={'tick_ns':now,'lio_stamp_s':traj[i,0],'lio_age_ms':poseage*1000,
                 'lio_available_capture_clock_ns':int(avail[i]),'joint_header_ns':stamps,
                 'lidar_imu_header_ns':ir['header_ns'],'lidar_imu_received_ns':ir['received_ns'],
                 'pelvis_position':pos.tolist(),'pelvis_quat_wxyz':rot.as_quat()[[3,0,1,2]].tolist(),
                 'pelvis_angular_velocity':omega.tolist(),'joint_position':q.tolist(),'joint_velocity':qd.tolist()}
            row['input_valid_against_model_ranges']=not bool(outside.any())
            row['out_of_range_joints']=[names[j] for j in np.flatnonzero(outside)]
            ready=True
            for key in histories:
                histories[key].append((q-default,qd,omega,gravity,previous[key].copy()))
                if len(histories[key])<10:ready=False;continue
                obs=np.concatenate([frame[term] for term in range(5) for frame in histories[key]])
                goal=trackers[key].goal(pos,rot.as_quat()[[3,0,1,2]],q,'stand')
                if obs.shape!=(990,) or goal.shape!=(729,) or not np.isfinite(np.r_[obs,goal]).all():
                    raise ValueError('Policy input contract failed')
                if first_probe is None:first_probe=(obs.copy(),goal.copy())
                before=time.perf_counter();action=np.asarray(policy.infer(obs.tolist(),goal.tolist()))
                timings.append((time.perf_counter()-before)*1000)
                if action.shape!=(31,) or not np.isfinite(action).all():raise ValueError('Nonfinite or invalid model output')
                target=default+scale*action
                row[key]={'observation':obs.tolist(),'goal':goal.tolist(),'action':action.tolist(),'target_logging_only':target.tolist()}
                outputs[key].append((action,target))
                if key=='feedback':
                    altered=goal.copy();altered[-9]+=0.02
                    perturbed=np.asarray(policy.infer(obs.tolist(),altered.tolist()))
                    deltas.append(float(np.max(np.abs(scale*(perturbed-action)))))
                previous[key]=action if key=='feedback' else np.zeros(n)
                trackers[key].advance()
            if ready:frames.write(json.dumps(row,allow_nan=False)+'\n');step_times.append(now)
            else:warmup+=1
    finally:frames.close()
    if not outputs['feedback']:
        (a.out/'report.json').write_text(json.dumps({'outputs':0,'rejected_ticks':dict(rejected),
            'policy_sha256':digest(a.policy),'strict_input_gates_passed':False,'deployable':False},indent=2)+'\n')
        raise RuntimeError('No valid model output; see report.json age and range gate counts')
    # Independent point-cloud plane diagnostic. It never changes the foot-based
    # registration or preceding inference inputs. Static capture only.
    points=sensors['points'];ts=sensors['point_stamps'];world=[]
    for t in np.unique(ts):
        if t<traj[0,0]+5:continue
        i=np.searchsorted(traj[:,0],t,side='right')-1
        rwi=Rotation.from_quat(traj[i,4:8]);xyz=points[ts==t]
        world.append(rwi.apply((xyz-pli)@rli)+traj[i,1:4])
    cloud=np.concatenate(world);radius=np.linalg.norm(cloud[:,:2]-origin[:2],axis=1)
    cloud=cloud[np.isfinite(cloud).all(axis=1)&(radius>.4)&(radius<4)&(cloud[:,2]<origin[2]-.25)]
    rng=np.random.default_rng(123);best=np.zeros(len(cloud),dtype=bool)
    for _ in range(200):
        if len(cloud)<3:break
        pts=cloud[rng.choice(len(cloud),3,replace=False)]
        normal=np.cross(pts[1]-pts[0],pts[2]-pts[0]);norm=np.linalg.norm(normal)
        if norm<1e-8:continue
        normal/=norm
        if abs(normal[2])<np.cos(np.deg2rad(15)):continue
        mask=np.abs((cloud-pts[0])@normal)<.015
        if mask.sum()>best.sum():best=mask
    plane=None
    if best.sum()>100:
        fit=cloud[best];center=fit.mean(axis=0);_,_,vh=np.linalg.svd(fit-center,full_matrices=False)
        normal=vh[-1]*(1 if vh[-1,2]>0 else -1);d=-normal@center
        zfloor=-(normal[0]*origin[0]+normal[1]*origin[1]+d)/normal[2]
        plane={'normal':normal.tolist(),'d':float(d),'inliers':int(best.sum()),'candidate_points':len(cloud),
               'tilt_deg':float(np.degrees(np.arccos(normal[2]))),
               'residual_rms_m':float(np.sqrt(np.mean((fit@normal+d)**2))),
               'floor_z_at_initial_pelvis':float(zfloor),
               'difference_from_foot_registration_m':float(zfloor+shift)}
    result={}
    for key,values in outputs.items():
        actions=np.array([v[0] for v in values]);targets=np.array([v[1] for v in values])
        outside=(targets<limits[:,0])|(targets>limits[:,1])
        result[key]={'frames':len(values),'all_finite':bool(np.isfinite(actions).all()),
                     'action_abs_max':float(abs(actions).max()),'target_abs_max_rad':float(abs(targets).max()),
                     'frames_outside_model_joint_range':int(np.any(outside,axis=1).sum()),
                     'outside_model_range_by_joint':{name:int(outside[:,i].sum()) for i,name in enumerate(names) if outside[:,i].any()},
                     'max_successive_logged_target_step_rad':float(abs(np.diff(targets,axis=0)).max())}
    probe_obs,probe_goal=first_probe
    identity_anchor=np.array([0.,0.,0.,1.,0.,0.,1.,0.,0.])
    no_error_goal=probe_goal.copy();no_error_goal[-9:]=identity_anchor
    g=probe_goal[:72]
    reference_obs=np.concatenate([np.tile(v,10) for v in (g[:31],g[31:62],g[65:68],g[62:65],np.zeros(31))])
    controls={}
    for name,ob,go in [('measured_proprioception_zero_anchor_error',probe_obs,no_error_goal),
                       ('matched_reference_proprioception_zero_previous_action',reference_obs,no_error_goal)]:
        action=np.asarray(policy.infer(ob.tolist(),go.tolist()));target=default+scale*action
        controls[name]={'action_abs_max':float(abs(action).max()),'target':target.tolist(),
                        'outside_model_range_joints':[names[j] for j in np.flatnonzero((target<limits[:,0])|(target>limits[:,1]))]}
    p=np.array(poses);feedback=np.array([v[1] for v in outputs['feedback']]);zero=np.array([v[1] for v in outputs['zero']])
    report={'scope':'finite server CPU read-only shadow inference; no ROS or network in inference process; no motor outputs',
            'capture_condition':'user: suspended for protection, soles touching ground',
            'selected_checkpoint':11500,'policy_sha256':digest(a.policy),'profile_sha256':digest(a.profile),
            'model_xml_sha256':digest(model_path),'dataset_manifest_sha256':digest(a.dataset/'manifest.json'),
            'mount_fk_source':'device URDF' if a.mount_urdf is not None else 'training model (legacy diagnostic)',
            'mount_urdf_sha256':digest(a.mount_urdf) if a.mount_urdf is not None else None,
            'policy_anchor_fk_source':'unchanged training model',
            'raw_bag_sha256':digest(root/'capture_latest/bag/bag_0.db3'),
            'observation_dim':990,'goal_dim':729,'action_dim':31,'joint_names':names,
            'frequency_hz':50,'history_frames':10,'invented_startup_history':False,
            'first_inference_after_real_history':True,'warmup_ticks':warmup,'rejected_ticks':dict(rejected),
            'input_header_receive_age_ms':ages,'accepted_lio_age_ms':stats(poseages),'accepted_joint_skew_ms':stats(skews),
            'pose_policy':'last available 10Hz LIO pose held, max age 150ms; no extrapolation or IMU propagation',
            'availability_policy':'measured server replay callback time mapped to recorded receive clock; only past received states',
            'initial_foot_bottom_z_local':initial_feet.tolist(),'world_z_translation_m':shift,
            'static_gravity_leveling_up_vector':up.tolist(),'gravity_leveling_matrix':leveling.as_matrix().tolist(),
            'gravity_bootstrap_samples':len(boot),'gravity_bootstrap_before_inference':True,
            'pelvis_height_m':stats(p[:,3]),'pelvis_displacement_from_first_max_m':float(np.linalg.norm(p[:,1:4]-p[0,1:4],axis=1).max()),
            'native_torso_imu_gravity_difference_deg':stats(native_gravity_errors),'ground_plane_cross_check':plane,
            'previous_action_policy':'feedback=own unexecuted outputs, zero=counterfactual zeros; actual action history unavailable',
            'diagnostic_out_of_range_enabled':a.diagnose_out_of_range,
            'input_out_of_model_range_ticks_by_joint':dict(input_outside),
            'strict_input_gates_passed':not bool(input_outside),
            'outputs':result,'feedback_vs_zero_max_target_difference_rad':float(abs(feedback-zero).max()),
            'single_frame_counterfactual_controls_not_robot_results':controls,
            'two_cm_anchor_x_perturbation_max_target_change_rad':stats(deltas),
            'server_cpu_model_call_ms':stats(timings),'model_input_frames_span_s':(step_times[-1]-step_times[0])*1e-9,
            'hardware_limits_validated':False,'motor_zero_direction_validated':False,'pc2_online_latency_validated':False,
            'full_imu_hardware_timestamps_verified':False,'dynamic_accuracy_validated':False,
            'closed_loop_stability_validated':False,'deployable':False}
    np.savetxt(a.out/'pelvis.csv',p,delimiter=',',header='tick_s,x,y,z,qx,qy,qz,qw,wx,wy,wz',comments='')
    (a.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
