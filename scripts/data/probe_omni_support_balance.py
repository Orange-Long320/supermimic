"""Bounded static pose probe for diagnosed X2 support/motor failures.

Keep both source foot poses and every arm joint. Optimize root/legs/waist only
within local domains and original soft bounds. Output is a pose diagnostic,
never a repaired trajectory or approved training reference.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['reference','model','scene','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--frames',nargs='+',type=int,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('New output required')
    import mujoco
    import numpy as np
    from scipy.optimize import minimize,Bounds
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra',check_assets=True)
    from supermimic.robots.dimensions import PROFILE
    from supermimic.data.contact_wrench import convex_contact_points,solve_contact_effort
    m=mujoco.MjModel.from_binary_path(str(a.model));d=mujoco.MjData(m)
    flags=int(m.opt.disableflags);m.opt.disableflags=flags|int(mujoco.mjtDisableBit.mjDSBL_CONTACT)|int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
    # The independent inverse-dynamics model has original inertias and the
    # profile armatures but no servo actuators. This probe never executes a
    # controller: motor inequalities use the unchanged profile effort caps.
    if (m.nq,m.nv,m.njnt)!=(38,37,32) or m.nu not in (0,31):raise ValueError('Expected original X2 inverse-dynamics model')
    with np.load(a.reference) as z:q=z['qpos'].copy()
    if min(a.frames)<0 or max(a.frames)>=len(q):raise ValueError('Frame outside original reference')
    names=[m.joint(i).name.rsplit('/',1)[-1] for i in range(1,m.njnt)]
    if names!=PROFILE['joint_names']:raise ValueError('Joint order changed')
    np.testing.assert_allclose(m.dof_armature[6:],[PROFILE['armature'][n] for n in names],rtol=0,atol=1e-12)
    effort=np.array([PROFILE['effort_limit'][n] for n in names])
    ids=[j+1 for j,n in enumerate(names) if any(k in n for k in ['hip_','knee_','ankle_','waist_'])]
    cols=m.jnt_qposadr[ids];vcols=np.r_[np.arange(6),m.jnt_dofadr[ids]].astype(int)
    root=int(m.jnt_bodyid[0]);feet=[m.body(side+'_ankle_roll_link').id for side in ['left','right']]
    obstacles=[g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]]!=root and (m.geom_contype[g] or m.geom_conaffinity[g])]
    robot=[g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]]==root and (m.geom_contype[g] or m.geom_conaffinity[g])]
    boxes=json.loads(a.scene.read_text())['scene_boxes'];surfaces=[(np.zeros(3),np.array([np.inf,np.inf,0.]),np.eye(3))]
    surfaces += [(np.array(b['center']),np.array(b['half_size']),np.array(b['rotation_world'])) for b in boxes]
    meshparts=[]
    for g in robot:
        if m.geom_bodyid[g] not in feet:continue
        if m.geom_type[g]!=int(mujoco.mjtGeom.mjGEOM_MESH):raise ValueError('Original foot meshes required')
        k=m.geom_dataid[g];start,count=m.mesh_vertadr[k],m.mesh_vertnum[k]
        meshparts.append((g,m.mesh_vert[start:start+count].copy()))
    jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);witness=np.zeros(6)
    a.out.mkdir(parents=True);(a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    report={'source_sha256':sha(a.reference),'model_sha256':sha(a.model),'scene_sha256':sha(a.scene),
        'arms_fixed':True,'both_foot_poses_fixed':True,'pose_diagnostics':[],'approved_for_training':False,'continuous_trajectory_verified':False}
    for frame in a.frames:
        base=q[frame].copy();d.qpos[:]=base;d.qvel[:]=0
        m.opt.disableflags=flags;mujoco.mj_forward(m,d)
        m.opt.disableflags=flags|int(mujoco.mjtDisableBit.mjDSBL_CONTACT)|int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
        target=[(d.xpos[b].copy(),Rotation.from_matrix(d.xmat[b].reshape(3,3))) for b in feet]
        contacts=[];xy=[]
        for g,vertices in meshparts:
            world=vertices@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
            for c,h,r in surfaces:
                local=(world-c)@r;mask=(abs(local[:,2]-h[2])<=.002)&(abs(local[:,:2])<=h[:2]).all(1)
                points=convex_contact_points(world[mask])
                for point in points:contacts.append((int(m.geom_bodyid[g]),point.copy(),r.copy()))
                xy.extend(points[:,:2])
        if len(xy)<3:
            report['pose_diagnostics'].append(dict(frame=frame,status='no_source_support_polygon'));continue
        polygon=ConvexHull(np.asarray(xy)).equations
        centre=m.jnt_range[ids].mean(1);radius=np.diff(m.jnt_range[ids],axis=1).ravel()*.45
        lower=np.r_[np.full(3,-.12),np.full(3,-.15),np.maximum(centre-radius-base[cols],-.5)]
        upper=np.r_[np.full(3,.12),np.full(3,.15),np.minimum(centre+radius-base[cols],.5)]
        # The waist stays close to the source torso. Every shoulder/elbow/wrist
        # is byte-identical; no arm swing is introduced for counterbalance.
        for j,index in enumerate(ids):
            if names[index-1].startswith('waist_'):
                lower[6+j]=max(lower[6+j],-.26);upper[6+j]=min(upper[6+j],.26)
        scales=np.r_[np.full(3,.10),np.full(3,.15),np.full(len(ids),.4)]
        def pose(x):
            value=base.copy();delta=np.zeros(m.nv);delta[vcols]=x;mujoco.mj_integratePos(m,value,delta,1.);return value
        cache={}
        def evaluate(x,force=False):
            key=(x.tobytes(),force)
            if key in cache:return cache[key]
            value=pose(x);d.qpos[:]=value;d.qvel[:]=0
            m.opt.disableflags=flags;mujoco.mj_forward(m,d)
            m.opt.disableflags=flags|int(mujoco.mjtDisableBit.mjDSBL_CONTACT)|int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
            d.qacc[:]=0;mujoco.mj_inverse(m,d)
            equal=np.concatenate([np.r_[(d.xpos[b]-p)*100.,(rotation.inv()*Rotation.from_matrix(d.xmat[b].reshape(3,3))).as_rotvec()] for b,(p,rotation) in zip(feet,target)])
            balance=-(polygon[:,:2]@d.subtree_com[root,:2]+polygon[:,2]+.003)*100
            clearance=np.array([mujoco.mj_geomDistance(m,d,g,o,.02,witness)+.00085 for g in robot for o in obstacles])*100
            constraints=np.r_[balance,clearance]
            fit=None
            if force:
                columns=[]
                for b,point,r in contacts:
                    rays=np.array([r[:,2]+.8/np.sqrt(2)*(u*r[:,0]+v*r[:,1]) for u,v in [(1,1),(1,-1),(-1,1),(-1,-1)]]).T
                    mujoco.mj_jac(m,d,jp,jr,point,b);columns.append(jp.T@rays)
                fit=solve_contact_effort(np.concatenate(columns,axis=1),d.qfrc_inverse.copy(),effort)
                constraints=np.r_[constraints,.95-float(fit.x[-1]) if fit.success else -100.]
            result=(equal,constraints,fit)
            # Limit memory while preserving the usual same-point objective /
            # equality / inequality calls during SLSQP finite differences.
            if len(cache)>100:cache.clear()
            cache[key]=result;return result
        initial=np.clip(np.zeros(len(vcols)),lower+1e-10,upper-1e-10)
        objective=lambda x:.5*float(np.square(x/scales).sum())
        stages=[];x=initial
        for with_force in [False,True]:
            solved=minimize(objective,x,method='SLSQP',bounds=Bounds(lower,upper),
                constraints=[dict(type='eq',fun=lambda x:evaluate(x,with_force)[0]),
                             dict(type='ineq',fun=lambda x:evaluate(x,with_force)[1])],
                options=dict(maxiter=50,ftol=1e-9,eps=1e-5))
            x=solved.x;equal,inequal,fit=evaluate(x,with_force)
            passed=bool(abs(equal).max()<=1e-4 and inequal.min()>=-1e-6)
            stages.append(dict(force_constrained=with_force,solver_success=bool(solved.success),message=str(solved.message),
                iterations=int(solved.nit),max_equal_residual=float(abs(equal).max()),minimum_inequality=float(inequal.min()),passed=passed,
                static_motor_ratio=float(fit.x[-1]) if fit is not None and fit.success else None))
            if not passed:break
        candidate=pose(x);np.savez_compressed(a.out/f'frame-{frame}.npz',qpos=candidate,source_qpos=base)
        unchanged=[7+j for j,n in enumerate(names) if not any(k in n for k in ['hip_','knee_','ankle_','waist_'])]
        np.testing.assert_array_equal(candidate[unchanged],base[unchanged])
        record=dict(frame=frame,stages=stages,root_translation_change_m=float(np.linalg.norm(candidate[:3]-base[:3])),
            max_joint_change_rad=float(abs(candidate[7:]-base[7:]).max()),pose_probe_passed=len(stages)==2 and stages[-1]['passed'])
        report['pose_diagnostics'].append(record);(a.out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(record),flush=True)
    (a.out/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
