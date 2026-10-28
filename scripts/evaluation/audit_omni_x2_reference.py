"""Server-only independent fixed-terrain X2 reference diagnostics.

Uses original X2 collision meshes and motor limits. Estimated unilateral foot
and optional palm contact cones are necessary-condition diagnostics, never a
continuous-execution certificate. No auxiliary movable terrain is retained.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--scene', type=Path, required=True, help='Adapter result.json with scene_boxes')
    p.add_argument('--allow-hand-support', action='store_true')
    p.add_argument('--friction', type=float, default=.8)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists() or a.friction <= 0:
        p.error('New output and positive friction required')
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra', check_assets=True)
    from supermimic.robots.dimensions import PROFILE
    from supermimic.data.contact_wrench import convex_contact_points, solve_contact_effort
    scene = json.loads(a.scene.read_text())
    spec = mujoco.MjSpec.from_file(str(a.model))
    obstacles = [(np.zeros(3), np.array([np.inf, np.inf, 0.]), np.eye(3))]
    spec.worldbody.add_geom(name='omni_audit_ground', type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0,0,.05))
    for i, box in enumerate(scene['scene_boxes']):
        center, half, rot = (np.array(box[k]) for k in ['center','half_size','rotation_world'])
        quat = Rotation.from_matrix(rot).as_quat()[[3,0,1,2]]
        spec.worldbody.add_geom(name=f'omni_audit_box_{i}', type=mujoco.mjtGeom.mjGEOM_BOX, pos=center, size=half, quat=quat)
        obstacles.append((center, half, rot))
    model = spec.compile()
    if (model.nq,model.nv,model.njnt) != (38,37,32):
        raise ValueError('Expected original X2 only, with fixed terrain')
    names = [model.joint(i).name for i in range(1,32)]
    if names != PROFILE['joint_names']:
        raise ValueError('X2 order differs')
    limits = np.array([PROFILE['effort_limit'][n] for n in names])
    speeds = np.array([PROFILE['velocity_limit'][n] for n in names])
    for j,n in enumerate(names,1):
        model.dof_armature[model.jnt_dofadr[j]] = PROFILE['armature'][n]
    data = mujoco.MjData(model)
    with np.load(a.reference, allow_pickle=False) as z:
        q, fps = z['qpos'].copy(), float(np.asarray(z['fps']).reshape(-1)[0])
    if q.ndim != 2 or q.shape[1] != 38 or len(q)<5 or not np.isfinite(q).all() or fps<=0:
        raise ValueError('Invalid reference')
    velocity = np.zeros((len(q),37));acceleration = np.zeros_like(velocity)
    forward,backward = np.zeros(37),np.zeros(37)
    for i in range(1,len(q)-1):
        mujoco.mj_differentiatePos(model,forward,1/fps,q[i],q[i+1])
        mujoco.mj_differentiatePos(model,backward,-1/fps,q[i],q[i-1])
        velocity[i]=(forward+backward)/2;acceleration[i]=(forward-backward)*fps
    velocity[0]=velocity[1];velocity[-1]=velocity[-2]
    robot_geoms=[g for g in range(model.ngeom) if not model.geom(g).name.startswith('omni_audit_') and (model.geom_contype[g] or model.geom_conaffinity[g])]
    terrain_geoms=[g for g in range(model.ngeom) if model.geom(g).name.startswith('omni_audit_')]
    support=[]
    for g in robot_geoms:
        name=model.body(int(model.geom_bodyid[g])).name
        if not (name.endswith('_ankle_roll_link') or (a.allow_hand_support and name.endswith('_wrist_roll_link'))):
            continue
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            vertices=np.array([(x,y,z) for x in (-1,1) for y in (-1,1) for z in (-1,1)])*model.geom_size[g]
        elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh=int(model.geom_dataid[g]);adr,count=model.mesh_vertadr[mesh],model.mesh_vertnum[mesh]
            vertices=model.mesh_vert[adr:adr+count].copy()
        else:
            raise ValueError('Unsupported contact geometry')
        support.append((g,name,vertices))
    rows=[];witness=np.zeros(6);jp,jr=np.zeros((3,37)),np.zeros((3,37))
    max_penetration=0.;max_self=0.;max_hard=0.;flags=model.opt.disableflags
    for i,pose in enumerate(q):
        model.opt.disableflags=flags
        data.qpos[:]=pose;data.qvel[:]=velocity[i];mujoco.mj_forward(model,data)
        max_hard=max(max_hard,float(np.maximum(np.maximum(model.jnt_range[1:,0]-pose[7:],pose[7:]-model.jnt_range[1:,1]),0).max()))
        for g in robot_geoms:
            for obstacle in terrain_geoms:
                max_penetration=max(max_penetration,-float(mujoco.mj_geomDistance(model,data,g,obstacle,.02,witness)))
        for contact in data.contact:
            if int(contact.geom1) in robot_geoms and int(contact.geom2) in robot_geoms:
                max_self=max(max_self,-float(contact.dist))
        if i<2 or i>=len(q)-2:
            continue
        columns=[];patches=[]
        for g,name,local in support:
            world=local@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
            hand=name.endswith('_wrist_roll_link')
            for k,(center,half,rot) in enumerate(obstacles):
                relative=(world-center)@rot
                faces=[(2,1)] if not hand or k==0 else [(axis,sign) for axis in range(3) for sign in (-1,1)]
                for axis,sign in faces:
                    # The box bottom is buried in its support; never create a
                    # fictitious upward-pulling underside hand contact.
                    if axis==2 and sign==-1:continue
                    tangent=[j for j in range(3) if j!=axis]
                    gap=relative[:,axis]*sign-half[axis]
                    mask=(abs(gap)<=.002)&(abs(relative[:,tangent])<=half[tangent]).all(1)
                    points=convex_contact_points(world[mask])
                    if not len(points):continue
                    normal=rot[:,axis]*sign;u,v=rot[:,tangent[0]],rot[:,tangent[1]]
                    rays=np.array([normal+a.friction/np.sqrt(2)*(x*u+y*v) for x,y in [(1,1),(1,-1),(-1,1),(-1,-1)]]).T
                    for point in points:
                        mujoco.mj_jac(model,data,jp,jr,point,int(model.geom_bodyid[g]));columns.append(jp.T@rays)
                    patches.append({'body':name,'surface':k,'face_axis':axis,'face_sign':sign,'extreme_points':len(points)})
        model.opt.disableflags=flags|int(mujoco.mjtDisableBit.mjDSBL_CONTACT)|int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
        data.qvel[:]=0.;data.qacc[:]=0.;mujoco.mj_inverse(model,data);static=data.qfrc_inverse.copy()
        data.qvel[:]=velocity[i];data.qacc[:]=acceleration[i];mujoco.mj_inverse(model,data);dynamic=data.qfrc_inverse.copy()
        fits={}
        for label,required in [('static',static),('dynamic',dynamic)]:
            if columns:
                basis=np.concatenate(columns,axis=1)
                fit=solve_contact_effort(basis,required,limits)
                fits[label]={'feasible_base_wrench':bool(fit.success),'minimum_motor_ratio':float(fit.x[-1]) if fit.success else None,'solver_status':int(fit.status)}
                if fit.success:
                    torque=required[6:]-basis[6:]@fit.x[:-1];fraction=abs(torque)/limits
                    dual=np.asarray(fit.ineqlin.marginals)
                    fits[label]['largest_motor_demands_in_one_optimal_solution']=[
                        {'joint':names[j],'required_torque_nm':float(torque[j]),'physical_limit_nm':float(limits[j]),
                         'motor_ratio':float(fraction[j]),'upper_inequality_dual':float(dual[j]),'lower_inequality_dual':float(dual[j+len(names)])}
                        for j in np.argsort(-fraction)[:5]]
            else:fits[label]={'feasible_base_wrench':False,'minimum_motor_ratio':None,'solver_status':None}
        rows.append({'frame':i,'patches':patches,**fits})
        if i%100==0:print(f'audited {i}/{len(q)}',flush=True)
    a.out.mkdir(parents=True)
    # Save the exact fixed-terrain model, including training armatures.
    mujoco.mj_saveModel(model,str(a.out/'fixed-terrain.mjb'))
    max_speed=float((abs(velocity[:,6:])/speeds).max())
    summary={label:{'feasible_frames':sum(r[label]['feasible_base_wrench'] for r in rows),
        'motor_within_limit_frames':sum(r[label]['feasible_base_wrench'] and r[label]['minimum_motor_ratio']<=1. for r in rows),
        'peak_minimum_motor_ratio':max((r[label]['minimum_motor_ratio'] for r in rows if r[label]['feasible_base_wrench']),default=None)} for label in ['static','dynamic']}
    summary.update(frames=len(q),audited_force_frames=len(rows),max_environment_penetration_m=max_penetration,
        max_native_self_penetration_m=max_self,hard_joint_violation_rad=max_hard,joint_speed_ratio=max_speed)
    necessary=bool(max_penetration<=.001 and max_self<=.001 and max_hard<=1e-6 and max_speed<=1 and summary['dynamic']['motor_within_limit_frames']==len(rows))
    report={'summary':summary,'necessary_diagnostics_passed':necessary,'allow_hand_support':a.allow_hand_support,
        'friction':a.friction,'contact_distance_tolerance_m':.002,'reference_sha256':hashlib.sha256(a.reference.read_bytes()).hexdigest(),
        'rows':rows,'training_approved':False,'scope':'Estimated contact cone and finite-difference inverse dynamics only; no controller execution, terminal hold or policy result'}
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__ == '__main__':
    main()
