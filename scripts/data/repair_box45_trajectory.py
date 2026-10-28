"""Joint trajectory optimization with native convex distances and temporal costs.

Server-only diagnostic. Keeps the box, robot, actuator limits and support-foot
targets; smooths the video orientation/upper body and optimizes all leg/root
translations together. Independent geometry/dynamics audits remain mandatory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import os


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-nfev',type=int,default=60)
    p.add_argument('--check-jacobian',action='store_true')
    p.add_argument('--resume',type=Path,help='Continue the same source/objective from saved solver knots')
    p.add_argument('--support-centred',action='store_true',help='Replace inherited COM drift with support-phase centres')
    p.add_argument('--knot-stride',type=int,default=5)
    p.add_argument('--foot-margin',type=float,default=0.0)
    p.add_argument('--free-torso',action='store_true',help='Optimize root orientation and waist joints for COM balance')
    a=p.parse_args()
    if a.out.exists():p.error('output exists')
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra',check_assets=True)
    import mujoco
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    from scipy.interpolate import make_interp_spline
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    from scipy.spatial.transform import Rotation
    from supermimic.robots.assets import get_spec
    from supermimic.envs.box_scene import FixedBox
    from supermimic.data.physics_motion import differentiate_qpos,export_qpos_motion

    z=np.load(a.source);original=z['qpos'].copy();fps=float(z['fps'])
    center,half=z['box_center'],z['box_half_size']
    box=FixedBox(front_x=float(center[0]-half[0]),center_y=float(center[1]),depth=float(2*half[0]),width=float(2*half[1]),height=float(2*half[2]))
    # The prior candidate has a declared 3s hold. Optimize through its entry;
    # the repeated tail is restored after a C2 stationary endpoint is formed.
    end=len(original)-round(3*fps)-1
    if end<20:raise ValueError('Expected moving segment plus declared 3s hold')
    if a.knot_stride<1 or a.foot_margin<0:raise ValueError('Invalid knot stride or foot margin')
    indices=np.unique(np.r_[np.arange(0,end+1,a.knot_stride),end]).astype(int)
    times=indices/fps;n=len(indices);dt=a.knot_stride/fps
    if not np.allclose(np.diff(times),dt):raise ValueError('Moving endpoint must fall on 10Hz knot grid')
    spec=get_spec();box.add_to_spec(spec)
    spec.worldbody.add_geom(name='repair_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=(0,0,.05),contype=1,conaffinity=1)
    m=spec.compile();d=mujoco.MjData(m)
    box_id=m.geom('box45_collision').id;floor_id=m.geom('repair_floor').id
    bodies=[m.body(s+'_ankle_roll_link').id for s in ('left','right')]
    leg_names=[s+'_'+j+'_joint' for s in ('left','right') for j in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
    if a.free_torso:leg_names += ['waist_'+j+'_joint' for j in ('yaw','roll','pitch')]
    qcols=np.array([m.jnt_qposadr[m.joint(s).id] for s in leg_names],dtype=int)
    vcols=np.r_[np.arange(3),[m.jnt_dofadr[m.joint(s).id] for s in leg_names]].astype(int)
    nr=3 if a.free_torso else 0
    nv=len(vcols)+nr;root=int(m.jnt_bodyid[0])
    geoms=[g for g in range(m.ngeom) if m.geom_bodyid[g]!=0 and (m.geom_contype[g] or m.geom_conaffinity[g])]
    width=nv+12+2+2*len(geoms)
    targets=[];com=[]
    for q in original[indices]:
        d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        targets.append([(d.xpos[b].copy(),d.xmat[b].reshape(3,3).copy()) for b in bodies])
        com.append(d.subtree_com[root].copy())
    com=gaussian_filter1d(np.array(com),2,axis=0,mode='nearest')
    com_weight=40.
    if a.support_centred:
        def sole_center(frame,side):
            d.qpos[:]=original[frame];mujoco.mj_kinematics(m,d);points=[]
            for g in geoms:
                if not m.geom(g).name.startswith('foot_') or side not in m.body(int(m.geom_bodyid[g])).name:continue
                mesh=m.geom_dataid[g]
                if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX:
                    local=np.array([(x,y,z) for x in (-1,1) for y in (-1,1) for z in (-1,1)])*m.geom_size[g]
                else:
                    adr,count=m.mesh_vertadr[mesh],m.mesh_vertnum[mesh];local=m.mesh_vert[adr:adr+count]
                points.extend(local@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])
            points=np.asarray(points);near=points[points[:,2]<points[:,2].min()+.012]
            return (near[:,:2].min(0)+near[:,:2].max(0))/2
        left=sole_center(0,'left');right_ground=sole_center(0,'right')
        right=sole_center(end,'right');left_top=sole_center(end,'left')
        timing=json.loads((Path(__file__).resolve().parents[2]/'configs/video/box45-contacts.json').read_text())
        scale=json.loads((a.source.parent/'fit.json').read_text())['time_scale']
        rl,rd,ll,ld=[timing[s][k]*scale for s,k in [('right','lift_seconds'),('right','land_seconds'),('left','lift_seconds'),('left','land_seconds')]]
        ease=lambda u:(lambda t:t**3*(10+t*(-15+6*t)))(np.clip(u,0,1))
        for i,t in enumerate(times):
            if t<rl:point=(left+right_ground)/2+(left-(left+right_ground)/2)*ease(t/(rl*.9))
            elif t<rd:point=left
            elif t<ll:point=left+(right-left)*ease((t-rd)/(ll-rd))
            elif t<ld:point=right
            else:point=right+((right+left_top)/2-right)*ease((t-ld)/1.)
            com[i,:2]=point
        com_weight=500.
    smooth=gaussian_filter1d(original[:end+1,7:],fps*.15,axis=0,mode='nearest')
    rotation=Rotation.from_quat(original[:end+1,[4,5,6,3]])
    reference_rotation=rotation[0]
    rotvec=gaussian_filter1d((reference_rotation.inv()*rotation).as_rotvec(),fps*.15,axis=0,mode='nearest')
    base=original[indices].copy();base[:,7:]=smooth[indices]
    base[:,3:7]=(reference_rotation*Rotation.from_rotvec(rotvec[indices])).as_quat()[:,[3,0,1,2]]
    initial=np.c_[original[indices,:3],np.zeros((n,nr)),original[indices][:,qcols]]
    reference=gaussian_filter1d(initial,2,axis=0,mode='nearest')
    joint_ids=np.array([m.joint(s).id for s in leg_names])
    ranges=m.jnt_range[joint_ids];mid=ranges.mean(1);radius=np.diff(ranges,axis=1)[:,0]*.9/2
    low=np.tile(np.r_[[-.5,-.7,.25],np.full(nr,-.6),mid-radius+1e-6],(n,1))
    high=np.tile(np.r_[[1.5,.7,1.7],np.full(nr,.6),mid+radius-1e-6],(n,1))
    posture_scale=np.r_[np.full(3,5.),np.full(nr,5.),np.full(len(qcols),.25)]
    acceleration_scale=np.r_[np.full(3,1.),np.full(nr,.2),np.full(len(qcols),.08)]/dt**2
    boundary_scale=np.r_[np.full(3,10.),np.ones(nv-3)]/dt
    total=n*width+(n-2)*nv+2*nv
    jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);witness=np.zeros(6)
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    (a.out/'launch.json').write_text(json.dumps({'argv':[sys.executable,*sys.argv],'pid':os.getpid(),'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest()},indent=2)+'\n')
    cache={};calls=0

    def evaluate(flat):
        nonlocal calls
        if 'x' in cache and np.array_equal(flat,cache['x']):return cache['r'],cache['j']
        x=flat.reshape(n,nv);r=np.zeros(total);jac=lil_matrix((total,n*nv))
        for i in range(n):
            row=i*width;cols=slice(i*nv,(i+1)*nv)
            q=base[i].copy();q[:3]=x[i,:3];q[qcols]=x[i,3+nr:]
            mapping=np.zeros((m.nv,nv));mapping[:3,:3]=np.eye(3)
            mapping[vcols[3:],np.arange(3+nr,nv)]=1
            if nr:
                delta=x[i,3:6];angle=np.linalg.norm(delta)
                skew=np.array([[0,-delta[2],delta[1]],[delta[2],0,-delta[0]],[-delta[1],delta[0],0]])
                c1=.5 if angle<1e-5 else (1-np.cos(angle))/angle**2
                c2=1/6 if angle<1e-5 else (angle-np.sin(angle))/angle**3
                mapping[3:6,3:6]=np.eye(3)-c1*skew+c2*skew@skew
                q[3:7]=(Rotation.from_quat(base[i,[4,5,6,3]])*Rotation.from_rotvec(delta)).as_quat()[[3,0,1,2]]
            d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
            r[row:row+nv]=(x[i]-reference[i])*posture_scale
            jac[row:row+nv,cols]=np.diag(posture_scale);row+=nv
            for b,(target_p,target_r) in zip(bodies,targets[i]):
                mujoco.mj_jacBody(m,d,jp,jr,b)
                r[row:row+3]=(d.xpos[b]-target_p)*1000
                jac[row:row+3,cols]=jp@mapping*1000;row+=3
                error=Rotation.from_matrix(target_r.T@d.xmat[b].reshape(3,3)).as_rotvec()
                skew=np.array([[0,-error[2],error[1]],[error[2],0,-error[0]],[-error[1],error[0],0]])
                angle=np.linalg.norm(error)
                coeff=1/12 if angle<1e-4 else 1/angle**2-(1+np.cos(angle))/(2*angle*np.sin(angle))
                inverse=np.eye(3)-.5*skew+coeff*skew@skew
                r[row:row+3]=error*100
                jac[row:row+3,cols]=inverse@target_r.T@jr@mapping*100;row+=3
            mujoco.mj_jacSubtreeCom(m,d,jp,root)
            r[row:row+2]=(d.subtree_com[root,:2]-com[i,:2])*com_weight
            jac[row:row+2,cols]=jp[:2]@mapping*com_weight;row+=2
            for g in geoms:
                is_foot=m.geom(g).name.startswith('foot_')
                for obstacle,margin in [(box_id,a.foot_margin if is_foot else .006),(floor_id,max(.001,a.foot_margin) if is_foot else .006)]:
                    dist=mujoco.mj_geomDistance(m,d,g,obstacle,.05,witness)
                    if dist<margin:
                        r[row]=(margin-dist)*1500
                        if abs(dist)>1e-9:
                            normal=(witness[3:]-witness[:3])/dist
                            mujoco.mj_jac(m,d,jp,jr,witness[:3],int(m.geom_bodyid[g]))
                            jac[row,cols]=(normal@jp@mapping)*1500
                    row+=1
            assert row==(i+1)*width
        row=n*width
        for i in range(1,n-1):
            r[row:row+nv]=(x[i-1]-2*x[i]+x[i+1])*acceleration_scale
            for offset,mult in [(-1,1),(0,-2),(1,1)]:
                jac[row:row+nv,slice((i+offset)*nv,(i+offset+1)*nv)]=np.diag(acceleration_scale*mult)
            row+=nv
        for first,last in [(0,1),(n-2,n-1)]:
            r[row:row+nv]=(x[last]-x[first])*boundary_scale
            jac[row:row+nv,slice(first*nv,(first+1)*nv)]=-np.diag(boundary_scale)
            jac[row:row+nv,slice(last*nv,(last+1)*nv)]=np.diag(boundary_scale);row+=nv
        calls+=1
        if calls%5==0:print(f'evaluation={calls} cost={np.dot(r,r)/2:.6f} max_residual={abs(r).max():.6f}',flush=True)
        cache.update(x=flat.copy(),r=r,j=jac.tocsr())
        return r,cache['j']

    start=np.clip(initial,low+1e-8,high-1e-8).ravel()
    if a.resume:
        saved=np.load(a.resume)
        if 'x' in saved:
            prior_hash=str(saved['source_sha256'])
            np.testing.assert_array_equal(saved['indices'],indices)
            start=saved['x'].copy()
        else:
            prior_hash=json.loads((a.resume.parent/'launch.json').read_text())['source_sha256']
            prior=saved['qpos'][indices]
            delta=(Rotation.from_quat(base[:,[4,5,6,3]]).inv()*Rotation.from_quat(prior[:,[4,5,6,3]])).as_rotvec() if nr else np.empty((n,0))
            start=np.c_[prior[:,:3],delta,prior[:,qcols]].ravel()
        if prior_hash!=hashlib.sha256(a.source.read_bytes()).hexdigest():raise ValueError('Resume source differs')
        if start.shape!=initial.ravel().shape:raise ValueError('Resume dimension differs')
        projected=np.clip(start,low.ravel()+1e-8,high.ravel()-1e-8)
        (a.out/'resume-projection.json').write_text(json.dumps({'maximum_initial_guess_projection':float(abs(projected-start).max()),'note':'Initial guess only; final trajectory is optimized under bounds and independently audited'},indent=2)+'\n')
        start=projected
    if a.check_jacobian:
        r,jac=evaluate(start);checks=[]
        for i in [0,min(n-1,42),n-1]:
            for column in ((0,2,3,4,5,7,14,20) if nr else (0,2,3,7,14)):
                c=i*nv+column;plus=start.copy();minus=start.copy();plus[c]+=1e-6;minus[c]-=1e-6
                numeric=(evaluate(plus)[0]-evaluate(minus)[0])/2e-6
                analytic=jac[:,c].toarray().ravel()
                checks.append(float(np.linalg.norm(numeric-analytic)/max(1.,np.linalg.norm(numeric))))
        report={'max_relative_column_error':max(checks),'passed':max(checks)<.02}
        (a.out/'jacobian-check.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
        if not report['passed']:raise SystemExit('Jacobian check failed')
        return
    result=least_squares(lambda x:evaluate(x)[0],start,jac=lambda x:evaluate(x)[1],bounds=(low.ravel(),high.ravel()),max_nfev=a.max_nfev,ftol=1e-6,xtol=1e-6,gtol=1e-6,verbose=1,tr_solver='lsmr')
    solved=result.x.reshape(n,nv);knots=base.copy();knots[:,:3]=solved[:,:3];knots[:,qcols]=solved[:,3+nr:]
    if nr:knots[:,3:7]=(Rotation.from_quat(base[:,[4,5,6,3]])*Rotation.from_rotvec(solved[:,3:6])).as_quat()[:,[3,0,1,2]]
    np.savez_compressed(a.out/'solver-knots.npz',x=result.x,indices=indices,source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest())
    dense_times=np.arange(end+1)/fps
    bc=([(1,np.zeros(m.nq-4)),(2,np.zeros(m.nq-4))],[(1,np.zeros(m.nq-4)),(2,np.zeros(m.nq-4))])
    scalars=np.c_[knots[:,:3],knots[:,7:]]
    scalar_curve=make_interp_spline(times,scalars,k=5,bc_type=bc)(dense_times)
    rb=([(1,np.zeros(3)),(2,np.zeros(3))],[(1,np.zeros(3)),(2,np.zeros(3))])
    knot_angles=(reference_rotation.inv()*Rotation.from_quat(knots[:,[4,5,6,3]])).as_rotvec()
    angles=make_interp_spline(times,knot_angles,k=5,bc_type=rb)(dense_times)
    quat=(reference_rotation*Rotation.from_rotvec(angles)).as_quat()[:,[3,0,1,2]]
    q=np.c_[scalar_curve[:,:3],quat,scalar_curve[:,3:]]
    q=np.r_[q,np.repeat(q[-1:],round(3*fps),axis=0)]
    velocity=differentiate_qpos(m,q,fps)
    np.savez_compressed(a.out/'aligned-diagnostic.npz',qpos=q,fps=fps,box_center=center,box_half_size=half)
    export_qpos_motion(m,q,a.out/'motion-fk.npz',fps,qvel=velocity,source=a.source)
    report={'cost':result.cost,'optimality':result.optimality,'nfev':result.nfev,'solver_success':bool(result.success),'frames':len(q),'peak_root_acceleration_m_s2':float(np.linalg.norm(np.diff(q[:,:3],n=2,axis=0)*fps**2,axis=1).max()),'peak_joint_acceleration_rad_s2':float(abs(np.diff(q[:,7:],n=2,axis=0)*fps**2).max()),'box':{'front_x':box.front_x,'center_y':box.center_y,'depth':box.depth,'width':box.width,'height':box.height,'friction':box.friction},'training_approved':False}
    (a.out/'repair.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
