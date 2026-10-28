"""Bounded coupled pose/contact-force refinement of a diagnosed Omni window.

Reuse the established sparse least-squares inverse-dynamics approach. Original
model, gravity, collision meshes, armatures, motor limits and time remain fixed.
Conservative contact vertices are attached to their real source support bodies.
This produces diagnostic candidates; dense independent audits and actual
continuous execution are mandatory before any training use.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['reference','model','scene','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--start-frame',type=int,required=True,help='Inclusive50Hz fixed boundary')
    p.add_argument('--end-frame',type=int,required=True,help='Inclusive50Hz fixed boundary')
    p.add_argument('--max-nfev',type=int,default=20)
    p.add_argument('--pose-seed-dir',type=Path,help='Optional independently checked static pose probes, used only as solver initialization')
    a=p.parse_args()
    if a.out.exists() or not 1<=a.max_nfev<=40:p.error('New output and at most40 evaluations required')
    a.out.mkdir(parents=True)
    (a.out/'pid').write_text(str(os.getpid()))
    (a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    import mujoco
    import numpy as np
    from scipy.optimize import least_squares,nnls
    from scipy.sparse import lil_matrix
    from scipy.spatial.transform import Rotation
    from scipy.interpolate import CubicSpline
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra',check_assets=True)
    from supermimic.robots.dimensions import PROFILE
    from supermimic.data.physics_motion import differentiate_qpos
    from supermimic.data.contact_wrench import solve_contact_effort
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only diagnostic required')
    m=mujoco.MjModel.from_binary_path(str(a.model));d=mujoco.MjData(m)
    if (m.nq,m.nv,m.njnt)!=(38,37,32) or m.nu not in (0,31) or m.neq:raise ValueError('Original floating X2 required')
    names=[m.joint(i).name.rsplit('/',1)[-1] for i in range(1,m.njnt)]
    if names!=PROFILE['joint_names']:raise ValueError('Unexpected joint order')
    np.testing.assert_allclose(m.opt.gravity,[0,0,-9.81],rtol=0,atol=1e-12)
    np.testing.assert_allclose(m.dof_armature[6:],[PROFILE['armature'][n] for n in names],rtol=0,atol=1e-12)
    with np.load(a.reference) as z:
        original=z['qpos'].copy();source_fps=float(np.asarray(z['fps']).ravel()[0])
    stride=round(source_fps/50)
    if stride<1 or abs(source_fps/stride-50)>1e-9:raise ValueError('Exact50Hz source subsampling required')
    q=original[::stride].copy();start,end=a.start_frame,a.end_frame;fps=50.
    if not 2<=start<end<len(q)-2:raise ValueError('Window needs two unchanged neighboring states on each side')
    flags=int(m.opt.disableflags);no_contacts=flags|int(mujoco.mjtDisableBit.mjDSBL_CONTACT)|int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
    root=int(m.jnt_bodyid[0]);weight=float(m.body_mass.sum()*9.81)
    scales=np.r_[np.full(3,weight),np.full(3,weight*.7)]
    effort=np.array([PROFILE['effort_limit'][n] for n in names]);speed=np.array([PROFILE['velocity_limit'][n] for n in names])
    robot=[g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]]==root and (m.geom_contype[g] or m.geom_conaffinity[g])]
    terrain=[g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]]!=root and (m.geom_contype[g] or m.geom_conaffinity[g])]
    feet=[m.body(s+'_ankle_roll_link').id for s in ['left','right']]
    hands=[m.body(s+'_wrist_roll_link').id for s in ['left','right']]
    scene=json.loads(a.scene.read_text());surfaces=[(np.zeros(3),np.array([np.inf,np.inf,0.]),np.eye(3))]
    surfaces += [(np.array(b['center']),np.array(b['half_size']),np.array(b['rotation_world'])) for b in scene['scene_boxes']]
    parts=[]
    for g in robot:
        if m.geom_bodyid[g] not in feet+hands:continue
        if m.geom_type[g]!=int(mujoco.mjtGeom.mjGEOM_MESH):raise ValueError('Original physical support meshes required')
        mesh=m.geom_dataid[g];first,count=m.mesh_vertadr[mesh],m.mesh_vertnum[mesh]
        parts.append((g,m.mesh_vert[first:first+count].copy()))
    contacts=[];targets=[];directions=np.array([[1,1],[1,-1],[-1,1],[-1,-1]])
    for frame in range(start,end+1):
        m.opt.disableflags=flags;d.qpos[:]=q[frame];d.qvel[:]=0;mujoco.mj_forward(m,d)
        support=[];active=set()
        for g,vertices in parts:
            body=int(m.geom_bodyid[g]);world=vertices@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
            for k,(c,h,r) in enumerate(surfaces):
                local=(world-c)@r
                faces=[(2,1)] if body in feet or k==0 else [(axis,sign) for axis in range(3) for sign in [-1,1] if (axis,sign)!=(2,-1)]
                for axis,sign in faces:
                    tangent=[j for j in range(3) if j!=axis]
                    mask=(abs(local[:,axis]*sign-h[axis])<=.002)&(abs(local[:,tangent])<=h[tangent]).all(1)
                    points=world[mask]
                    if not len(points):continue
                    # A conservative subset of real vertices, never a larger
                    # rectangle or fictitious patch outside collision meshes.
                    picked=np.unique(np.argmax((points@r)[:,tangent]@directions.T,axis=0))
                    rays=np.array([sign*r[:,axis]+.8/np.sqrt(2)*(u*r[:,tangent[0]]+v*r[:,tangent[1]]) for u,v in directions]).T
                    for point in points[picked]:support.append((body,(point-d.xpos[body])@d.xmat[body].reshape(3,3),rays.copy()))
                    active.add(body)
        contacts.append(support)
        targets.append([(b,d.xpos[b].copy(),Rotation.from_matrix(d.xmat[b].reshape(3,3))) for b in feet+sorted(active.intersection(hands))])
    width=37;n=end-start-1;count=end-start+1
    sizes=[4*len(c) for c in contacts];offsets=np.r_[n*width,n*width+np.cumsum(sizes)].astype(int)
    initial=np.zeros(offsets[-1]);low=np.full_like(initial,-np.inf);high=np.full_like(initial,np.inf)
    centre=m.jnt_range[1:].mean(1);radius=np.diff(m.jnt_range[1:],axis=1).ravel()*.45
    domains=np.array([.26 if any(k in name for k in ['shoulder','elbow','wrist','waist','head','neck']) else .5 for name in names])
    for i,frame in enumerate(range(start+1,end)):
        lo=np.r_[np.full(3,-.12),np.full(3,-.15),np.maximum(centre-radius-q[frame,7:],-domains)]
        hi=np.r_[np.full(3,.12),np.full(3,.15),np.minimum(centre+radius-q[frame,7:],domains)]
        low[i*width:(i+1)*width]=lo;high[i*width:(i+1)*width]=hi
    low[n*width:]=0;high[n*width:]=4.
    if np.any(low>=high):raise ValueError('Empty unchanged-soft-limit domain')
    seeds=[]
    if a.pose_seed_dir:
        seed_report=json.loads((a.pose_seed_dir/'report.json').read_text())
        if seed_report['source_sha256']!=hashlib.sha256(a.reference.read_bytes()).hexdigest():raise ValueError('Static seeds use a different source')
        knots=[float(start)];deltas=[np.zeros(37)]
        for item in sorted(seed_report['pose_diagnostics'],key=lambda x:x['frame']):
            frame=item['frame'];location=frame/stride
            if not start<location<end or not item.get('pose_probe_passed'):continue
            path=a.pose_seed_dir/f'frame-{frame}.npz'
            with np.load(path) as z:source=z['source_qpos'];candidate=z['qpos']
            np.testing.assert_array_equal(source,original[frame]);delta=np.zeros(37)
            mujoco.mj_differentiatePos(m,delta,1.,source,candidate)
            knots.append(location);deltas.append(delta);seeds.append(str(path))
        knots.append(float(end));deltas.append(np.zeros(37))
        if not seeds:raise ValueError('No successful static seeds within this window')
        curve=CubicSpline(knots,np.asarray(deltas),axis=0,bc_type=((1,np.zeros(37)),(1,np.zeros(37))))
        initial[:n*width]=curve(np.arange(start+1,end)).ravel()
    initial=np.clip(initial,low+1e-10,high-1e-10)
    initial_q=q.copy()
    for i,frame in enumerate(range(start+1,end)):
        mujoco.mj_integratePos(m,initial_q[frame],initial[i*width:(i+1)*width],1.)
    tangent=np.zeros(m.nv);backward=np.zeros(m.nv);jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);witness=np.zeros(6)
    def dynamics(qs,frame,points):
        d.qpos[:]=qs[frame]
        mujoco.mj_differentiatePos(m,tangent,1/fps,qs[frame],qs[frame+1])
        mujoco.mj_differentiatePos(m,backward,-1/fps,qs[frame],qs[frame-1])
        d.qvel[:]=(tangent+backward)/2;m.opt.disableflags=flags;mujoco.mj_forward(m,d)
        self_pen=max([-float(c.dist) for c in d.contact if c.geom1 in robot and c.geom2 in robot]+[0.])
        m.opt.disableflags=no_contacts;d.qacc[:]=(tangent-backward)*fps;mujoco.mj_inverse(m,d)
        bases=[]
        for body,local,rays in points:
            point=d.xpos[body]+d.xmat[body].reshape(3,3)@local;mujoco.mj_jac(m,d,jp,jr,point,body);bases.append(jp.T@rays)
        return d.qfrc_inverse.copy(),np.concatenate(bases,axis=1) if bases else np.zeros((37,0)),self_pen
    for i,frame in enumerate(range(start,end+1)):
        required,basis,_=dynamics(initial_q,frame,contacts[i])
        if basis.shape[1]:
            coefficients,_=nnls(basis[:6]/scales[:,None],required[:6]/scales,maxiter=10000)
            initial[offsets[i]:offsets[i+1]]=coefficients/weight
    initial=np.clip(initial,low+1e-10,high-1e-10)
    def unpack(x):
        qs=q.copy()
        for i,frame in enumerate(range(start+1,end)):
            mujoco.mj_integratePos(m,qs[frame],x[i*width:(i+1)*width],1.)
        return qs
    blocks=[];row=0
    for i in range(count):
        length=6+31+31+sizes[i]+6*len(targets[i])+len(robot)*len(terrain)+1+width
        blocks.append((row,row+length));row+=length
    smooth_count=end-start+3
    sparsity=lil_matrix((row+smooth_count*width,len(initial)),dtype=int)
    for i,frame in enumerate(range(start,end+1)):
        begin,finish=blocks[i]
        for f in range(max(start+1,frame-1),min(end-1,frame+1)+1):sparsity[begin:finish,(f-start-1)*width:(f-start)*width]=1
        sparsity[begin:finish,offsets[i]:offsets[i+1]]=1
    for i in range(smooth_count):
        frame=start-1+i
        for f in range(max(start+1,frame-1),min(end-1,frame+1)+1):sparsity[row+i*width:row+(i+1)*width,(f-start-1)*width:(f-start)*width]=1
    def residual(x):
        qs=unpack(x);values=[]
        for i,frame in enumerate(range(start,end+1)):
            required,basis,self_pen=dynamics(qs,frame,contacts[i]);coeff=x[offsets[i]:offsets[i+1]]
            force=required-basis@(coeff*weight)
            values.extend(force[:6]/scales*200.)
            values.extend(np.maximum(abs(force[6:])/effort-.95,0)*30.)
            values.extend(np.maximum(abs(d.qvel[6:])/speed-1.,0)*30.)
            values.extend(coeff*.01)
            for body,p,rotation in targets[i]:
                values.extend((d.xpos[body]-p)*2000.)
                values.extend((rotation.inv()*Rotation.from_matrix(d.xmat[body].reshape(3,3))).as_rotvec()*200.)
            values.extend(max(0.,-float(mujoco.mj_geomDistance(m,d,g,o,.02,witness))-.00085)*10000. for g in robot for o in terrain)
            values.append(max(0.,self_pen-.00085)*10000.)
            delta=np.zeros(37);mujoco.mj_differentiatePos(m,delta,1.,q[frame],qs[frame])
            values.extend(delta*np.r_[np.full(3,10.),np.full(3,5.),np.ones(31)])
        section=qs[start-2:end+3];source=q[start-2:end+3];corrections=[]
        for nominal,pose in zip(source,section):
            delta=np.zeros(37);mujoco.mj_differentiatePos(m,delta,1.,nominal,pose);corrections.append(delta)
        values.extend((np.diff(corrections,n=2,axis=0)*np.r_[np.full(3,500.),np.full(3,50.),np.full(31,5.)]).ravel())
        return np.asarray(values)
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    record=dict(pid=os.getpid(),reference_sha256=sha(a.reference),model_sha256=sha(a.model),scene_sha256=sha(a.scene),
        frames50hz=[start,end],fixed_boundary_frames=True,time_scale=1.,friction=.8,variables=len(initial),residuals=sparsity.shape[0],
        source_fps=source_fps,contact_subset='At most4 actual mesh directional extrema per physical face; conservative subset',
        static_initialization_sha256={path:sha(path) for path in seeds},
        scope=__doc__,approved_for_training=False,continuous_execution_verified=False)
    (a.out/'as-run.py').write_bytes(Path(__file__).read_bytes());(a.out/'launch.json').write_text(json.dumps(record,indent=2))
    started=time.monotonic();iteration=0
    def checkpoint(x):
        nonlocal iteration
        iteration+=1
        qs=unpack(x);np.savez_compressed(a.out/'iteration-latest.npz',qpos=qs,fps=np.array([fps]),variables=x)
        print(json.dumps(dict(iteration=iteration,elapsed_seconds=time.monotonic()-started,cost=float(np.square(residual(x)).sum()/2))),flush=True)
    r0=residual(initial);print(json.dumps(dict(initial_cost=float(r0@r0/2),**record)),flush=True)
    solved=least_squares(residual,initial,bounds=(low,high),jac_sparsity=sparsity.tocsr(),
        x_scale='jac',tr_options={'maxiter':100},max_nfev=a.max_nfev,callback=checkpoint,ftol=1e-6,xtol=1e-7,gtol=1e-6)
    final=unpack(solved.x);velocity=differentiate_qpos(m,final,fps)
    np.testing.assert_array_equal(final[:start+1],q[:start+1]);np.testing.assert_array_equal(final[end:],q[end:])
    np.savez_compressed(a.out/'candidate-qpos.npz',qpos=final,qvel=velocity,fps=np.array([fps]),variables=solved.x)
    rows=[]
    for i,frame in enumerate(range(start,end+1)):
        required,basis,_=dynamics(final,frame,contacts[i]);force=required-basis@(solved.x[offsets[i]:offsets[i+1]]*weight)
        fit=solve_contact_effort(basis,required,effort) if basis.shape[1] else None
        rows.append(dict(frame=frame,root_residual_normalized=float(abs(force[:6]/scales).max()),motor_ratio=float((abs(force[6:])/effort).max()),
            contact_lp_feasible=bool(fit is not None and fit.success),minimum_motor_ratio=float(fit.x[-1]) if fit is not None and fit.success else None))
    record.update(solver_success=bool(solved.success),message=str(solved.message),nfev=int(solved.nfev),cost=float(solved.cost),
        initial_cost=float(r0@r0/2),elapsed_seconds=time.monotonic()-started,rows=rows)
    (a.out/'report.json').write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
