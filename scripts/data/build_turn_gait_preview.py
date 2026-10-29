"""Server-only synthetic turning reference, preserving source time and arm motion.

Bend the root path and solve bounded leg IK with stance foot anchors. This is a kinematic design, not a recovered
turning performance or a physics rollout.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--angle-deg', type=float, default=60.)
    p.add_argument('--in-place', action='store_true', help='Fix pelvis XY, reduce fore-aft steps, turn heading exactly by requested angle')
    a = p.parse_args()
    if a.out.exists(): p.error('output directory already exists')
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra', check_assets=True)
    import mujoco
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from supermimic.data.motion_convert import load_g1_model, ResampledMotion, forward_kinematics
    model = load_g1_model()
    data = mujoco.MjData(model)
    src = np.load(a.input)
    fps = float(src['fps'][0])
    names = src['body_names'].tolist()
    joints = src['joint_names'].tolist()
    root = names.index('pelvis')
    pos = src['body_pos_w'][:, root].astype(float)
    quat = src['body_quat_w'][:, root].astype(float)
    q = src['joint_pos'].astype(float).copy()
    h = Rotation.from_quat(quat[0, [1,2,3,0]]).apply([1.,0,0])
    h[2] = 0; h /= np.linalg.norm(h)
    if not 0 < abs(a.angle_deg) <= 90: p.error('angle magnitude must be in (0,90]')
    travel = np.maximum(0, (pos-pos[0]) @ h)
    progress = np.maximum.accumulate(travel)
    if progress[-1] < .1: raise ValueError('source has insufficient walking travel')
    u = progress / progress[-1]
    yaw = np.deg2rad(a.angle_deg) * (3*u*u - 2*u*u*u)
    if a.in_place:
        facing = Rotation.from_quat(quat[:,[1,2,3,0]]).apply(np.tile([1.,0,0],(len(quat),1)))
        source_yaw = np.unwrap(np.arctan2(facing[:,1],facing[:,0]))
        yaw -= source_yaw-source_yaw[0]
    turn = Rotation.from_euler('z', yaw[:,None])
    quat = (turn * Rotation.from_quat(quat[:,[1,2,3,0]])).as_quat()[:,[3,0,1,2]]
    backward = pos.copy()
    delta = np.diff(pos,axis=0)
    delta[:,:2] = Rotation.from_euler('z',((yaw[1:]+yaw[:-1])/2)[:,None]).apply(delta)[:,:2]
    backward[1:] = pos[0] + np.cumsum(delta,axis=0)
    if a.in_place: backward[:,:2] = pos[0,:2]
    feet = [model.body(f'{s}_ankle_roll_link').id for s in ('left','right')]
    original = src['body_pos_w'][:, [names.index(model.body(i).name) for i in feet]].astype(float)
    rel = original - pos[:,None,:]
    if a.in_place:
        neutral = rel[:min(40,len(rel))].mean(axis=0)
        rel -= .85 * ((rel-neutral) @ h)[...,None] * h
    targets = backward[:,None,:] + np.stack([turn.apply(rel[:,k]) for k in range(2)],axis=1)
    foot_rot = Rotation.from_quat(src['body_quat_w'][:,[names.index(model.body(i).name) for i in feet]][:,:,[1,2,3,0]].reshape(-1,4)).as_matrix().reshape(len(q),2,3,3)
    foot_rot = turn.as_matrix()[:,None] @ foot_rot
    stance_runs = []
    # Source ankle height estimates contact; keep only sustained low-foot runs.
    for k in range(2):
        z = original[:,k,2]
        low = z < np.quantile(z,.15) + .025
        if a.in_place: low = z - original[:,:,2].min(axis=1) < .025
        edges = np.diff(np.r_[False,low,False].astype(int))
        runs = [(int(b),int(e)) for b,e in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if e-b>=3]
        if len(runs)<2: raise ValueError('insufficient stance phases for turn synthesis')
        anchors=[]
        rotations=[]
        for b,e in runs:
            anchor = np.mean(targets[b:e,k,:2],axis=0)
            rot = Rotation.from_matrix(foot_rot[(b+e-1)//2,k])
            targets[b:e,k,:2] = anchor
            foot_rot[b:e,k] = rot.as_matrix()
            anchors.append(anchor);rotations.append(rot)
        from scipy.spatial.transform import Slerp
        for j in range(len(runs)-1):
            b,e = runs[j][1]-1,runs[j+1][0]
            t = np.linspace(0,1,e-b+1)
            blend = 3*t*t-2*t*t*t
            targets[b:e+1,k,:2] = (1-blend[:,None])*anchors[j] + blend[:,None]*anchors[j+1]
            rr = Rotation.from_matrix(np.stack([rotations[j].as_matrix(),rotations[j+1].as_matrix()]))
            foot_rot[b:e+1,k] = Slerp([0,1],rr)(blend).as_matrix()
        stance_runs.append(runs)
    leg = np.array([i for i,n in enumerate(joints) if any(k in n for k in ('hip','knee','ankle'))])
    lo, hi = model.jnt_range[leg+1].T
    lo, hi = lo+1e-6, hi-1e-6
    errors = []
    previous = np.clip(q[0,leg],lo,hi)
    for t in range(len(q)):
        source = q[t].copy()
        data.qpos[:3] = backward[t]
        data.qpos[3:7] = quat[t]
        data.qpos[7:] = source
        def residual(x):
            data.qpos[7+leg] = x
            mujoco.mj_kinematics(model,data)
            r = [50*(data.xpos[feet]-targets[t]).ravel()]
            for k,b in enumerate(feet):
                r.append(Rotation.from_matrix(foot_rot[t,k].T @ data.xmat[b].reshape(3,3)).as_rotvec())
            r.extend([.03*(x-source[leg]),.02*(x-previous)])
            return np.concatenate(r)
        fit = least_squares(residual, previous, bounds=(lo,hi),max_nfev=100,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        q[t,leg] = fit.x
        residual(fit.x)
        errors.append(np.linalg.norm(data.xpos[feet]-targets[t],axis=1))
        previous = fit.x.copy()
    # Check actual foot collision vertices and correct only residual floor error.
    foot_geoms = []
    for g in range(model.ngeom):
        if model.geom(g).name.startswith('foot_'):
            mid = model.geom_dataid[g]
            if mid < 0: raise ValueError('expected pinned mesh soles')
            v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]]
            foot_geoms.append((g,v.copy()))
    if not foot_geoms: raise ValueError('missing foot collision meshes')
    lifts=[]
    for t in range(len(q)):
        data.qpos[:] = np.r_[backward[t],quat[t],q[t]]
        mujoco.mj_kinematics(model,data)
        z=min(float((v @ data.geom_xmat[g].reshape(3,3).T + data.geom_xpos[g])[:,2].min()) for g,v in foot_geoms)
        lift=max(0.,-z);backward[t,2]+=lift;lifts.append(lift)
    if max(lifts) > .03 or np.max(errors) > .04:
        raise ValueError(f'candidate needs repair: floor lift {max(lifts)}, IK error {np.max(errors)}')
    dt=1/fps
    dq=(Rotation.from_quat(quat[1:,[1,2,3,0]])*Rotation.from_quat(quat[:-1,[1,2,3,0]]).inv()).as_rotvec()/dt
    ang=np.vstack([dq[:1],(dq[:-1]+dq[1:])/2,dq[-1:]])
    motion=ResampledMotion(fps,backward,quat,q,np.gradient(backward,dt,axis=0),ang,np.gradient(q,dt,axis=0))
    bodies=forward_kinematics(model,motion)
    a.out.mkdir(parents=True)
    arrays=dict(fps=np.array([fps]),joint_pos=q,joint_vel=motion.joint_vel,body_pos_w=bodies.pos_w,body_quat_w=bodies.quat_w,body_lin_vel_w=bodies.lin_vel_w,body_ang_vel_w=bodies.ang_vel_w,joint_names=np.array(joints),body_names=np.array(bodies.body_names),source=np.array(str(a.input)))
    for value in arrays.values():
        if np.issubdtype(value.dtype,np.number) and not np.isfinite(value).all(): raise ValueError('nonfinite output')
    np.savez(a.out/'turn.npz',**arrays)
    report=dict(method=('fixed pelvis XY, reduced fore-aft stepping, relative-height stance anchors and bounded leg IK' if a.in_place else 'curved root path, smooth yaw and source-height stance anchors, bounded leg IK; source frame order retained'),source=str(a.input),source_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),frames=len(q),fps=fps,duration_s=(len(q)-1)/fps,forward_displacement_m=float((pos[-1]-pos[0])@h),turn_angle_deg=a.angle_deg,in_place=a.in_place,max_pelvis_horizontal_drift_m=float(np.linalg.norm(backward[:,:2]-backward[0,:2],axis=1).max()),stance_runs=stance_runs,endpoint_displacement_m=(backward[-1]-backward[0]).tolist(),max_foot_target_error_m=float(np.max(errors)),max_floor_lift_m=max(lifts),max_joint_limit_violation_rad=float(np.maximum(model.jnt_range[1:,0]-q,q-model.jnt_range[1:,1]).clip(0).max()),arms_unchanged=bool(np.array_equal(q[:,[i for i in range(len(joints)) if i not in leg]],src['joint_pos'][:,[i for i in range(len(joints)) if i not in leg]])),physics_validated=False,training_approved=False)
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
