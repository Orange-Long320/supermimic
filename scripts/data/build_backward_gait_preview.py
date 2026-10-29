"""Server-only synthetic backward reference, preserving source time and arm motion.

Reflect fore-aft foot excursions about the standing offsets and reverse forward
root travel; solve bounded leg IK. This is a kinematic design, not a recovered
backward performance or a physics rollout.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
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
    backward = pos - 2 * ((pos-pos[0]) @ h)[:,None] * h
    feet = [model.body(f'{s}_ankle_roll_link').id for s in ('left','right')]
    original = src['body_pos_w'][:, [names.index(model.body(i).name) for i in feet]].astype(float)
    rel = original - pos[:,None,:]
    neutral = rel[:min(40,len(rel))].mean(axis=0)
    targets = backward[:,None,:] + rel - 2 * ((rel-neutral) @ h)[...,None] * h
    foot_rot = Rotation.from_quat(src['body_quat_w'][:,[names.index(model.body(i).name) for i in feet]][:,:,[1,2,3,0]].reshape(-1,4)).as_matrix().reshape(len(q),2,3,3)
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
    np.savez(a.out/'backward.npz',**arrays)
    report=dict(method='reflect fore-aft root travel and foot excursions, bounded leg IK; source frame order retained',source=str(a.input),source_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),frames=len(q),fps=fps,duration_s=(len(q)-1)/fps,forward_displacement_m=float((pos[-1]-pos[0])@h),backward_displacement_m=float((backward[-1]-backward[0])@h),max_foot_target_error_m=float(np.max(errors)),max_floor_lift_m=max(lifts),max_joint_limit_violation_rad=float(np.maximum(model.jnt_range[1:,0]-q,q-model.jnt_range[1:,1]).clip(0).max()),arms_unchanged=bool(np.array_equal(q[:,[i for i in range(len(joints)) if i not in leg]],src['joint_pos'][:,[i for i in range(len(joints)) if i not in leg]])),physics_validated=False,training_approved=False)
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
