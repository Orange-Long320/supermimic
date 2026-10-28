"""Minimal smooth shoulder counterbalance for actual support-polygon gaps.

Root, legs, contact timing and checked standing endpoint remain byte-identical.
The arm offset is bounded and C2 in time; output still requires dense force and
geometry audits and actual continuous execution.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--scan',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--diagnose-only',action='store_true')
    p.add_argument('--motor-reserve',type=float,help='Static exact-contact peak motor fraction; no motor limits are changed')
    a=p.parse_args()
    if a.out.exists():p.error('output exists')
    if a.motor_reserve is not None and not 0<a.motor_reserve<1:p.error('Motor reserve must be between zero and one')
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra',check_assets=True)
    import mujoco
    import numpy as np
    from scipy.spatial import ConvexHull
    from scipy.optimize import minimize,Bounds
    from supermimic.robots.dimensions import PROFILE
    from supermimic.envs.box_scene import FixedBox
    from supermimic.data.physics_motion import differentiate_qpos,export_qpos_motion
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.data.contact_wrench import solve_contact_effort,convex_contact_points

    z=np.load(a.source);original=z['qpos'];fps=float(z['fps']);q=original.copy()
    center,half=z['box_center'],z['box_half_size']
    box=FixedBox(front_x=float(center[0]-half[0]),center_y=float(center[1]),depth=float(half[0]*2),width=float(half[1]*2),height=float(half[2]*2))
    failed=json.loads(a.scan.read_text())['summary']['None']['failed_frames']
    if not failed:raise ValueError('No static wrench or motor failure to repair')
    lo=max(0,min(failed)-round(.6*fps));hi=min(len(q)-1,max(failed)+round(.8*fps));middle=float(np.mean(failed))
    frames=np.arange(lo,hi+1)
    ease=lambda t:(lambda u:u**3*(10+u*(-15+6*u)))(np.clip(t,0,1))
    weights=np.where(frames<=middle,ease((frames-lo)/(middle-lo)),1-ease((frames-middle)/(hi-middle)))
    m=mujoco.MjModel.from_binary_path(str(a.model));m.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_CONTACT)|int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
    d=mujoco.MjData(m);root=int(m.jnt_bodyid[0])
    names=[s+'_shoulder_'+j+'_joint' for s in ('left','right') for j in ('pitch','roll')]
    all_names=[m.joint(i).name.rsplit('/',1)[-1] for i in range(1,m.njnt)]
    joint_ids=np.array([all_names.index(name)+1 for name in names]);cols=m.jnt_qposadr[joint_ids]
    upper=[i for i,name in enumerate(all_names) if not any(k in name for k in ('hip_','knee_','ankle_'))]
    caps=np.array([PROFILE['effort_limit'][all_names[i]] for i in upper])
    foot_geoms=[];arm_geoms=[]
    for g in range(m.ngeom):
        leaf=m.geom(g).name.rsplit('/',1)[-1]
        if leaf.startswith('foot_'):
            mesh=m.geom_dataid[g]
            if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX:
                local=np.array([(x,y,z) for x in (-1,1) for y in (-1,1) for z in (-1,1)])*m.geom_size[g]
            else:
                adr,count=m.mesh_vertadr[mesh],m.mesh_vertnum[mesh];local=m.mesh_vert[adr:adr+count]
            foot_geoms.append((g,local))
        if m.geom_bodyid[g] and any(k in m.body(int(m.geom_bodyid[g])).name for k in ('shoulder','elbow','wrist')) and (m.geom_contype[g] or m.geom_conaffinity[g]):arm_geoms.append(g)
    obstacles=[m.geom('box45_collision').id]+[g for g in range(m.ngeom) if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE]
    polygons=[];contact_bases=[]
    jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
    rays=np.array([[x*box.friction/np.sqrt(2),y*box.friction/np.sqrt(2),1] for x,y in ((1,1),(1,-1),(-1,1),(-1,-1))]).T
    all_caps=np.array([PROFILE['effort_limit'][name] for name in all_names])
    for i in frames:
        d.qpos[:]=q[i];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);points=[];columns=[]
        for g,local in foot_geoms:
            world=local@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
            inside=(world[:,0]>=box.front_x)&(world[:,0]<=box.front_x+box.depth)&(abs(world[:,1]-box.center_y)<=box.width/2)
            mask=((~inside)&(abs(world[:,2])<.012))|(inside&(abs(world[:,2]-box.height)<.012))
            points.extend(world[mask,:2])
            if a.motor_reserve is not None:
                for point in convex_contact_points(world[mask]):
                    mujoco.mj_jac(m,d,jp,jr,point,int(m.geom_bodyid[g]));columns.append(jp.T@rays)
        polygons.append(ConvexHull(np.asarray(points)).equations)
        contact_bases.append(np.concatenate(columns,axis=1) if columns else None)
    low=np.full(4,-.5);high=np.full(4,.5)
    limits=m.jnt_range[joint_ids];mid=limits.mean(1);radius=np.diff(limits,axis=1)[:,0]*.9/2
    for i,w in zip(frames,weights):
        if w<1e-6:continue
        low=np.maximum(low,(mid-radius-q[i,cols])/w);high=np.minimum(high,(mid+radius-q[i,cols])/w)
    if np.any(low>=high):raise ValueError('No bounded shoulder correction')
    cache={};calls=0;witness=np.zeros(6)
    def constraints(amplitude):
        nonlocal calls
        if 'x' in cache and np.array_equal(amplitude,cache['x']):return cache['c']
        c=[];labels=[]
        for i,w,polygon,basis in zip(frames,weights,polygons,contact_bases):
            d.qpos[:]=q[i];d.qpos[cols]+=w*amplitude;d.qvel[:]=0;d.qacc[:]=0;mujoco.mj_inverse(m,d)
            c.extend(-(polygon[:,:2]@d.subtree_com[root,:2]+polygon[:,2]+.003*w)/.05)
            labels.extend([('support',int(i),k) for k in range(len(polygon))])
            gravity=d.qfrc_inverse[6:][upper]/caps
            c.extend(.95-gravity);c.extend(.95+gravity)
            labels.extend([('gravity',int(i),all_names[j],sign) for sign in ('upper','lower') for j in upper])
            if a.motor_reserve is not None:
                fit=solve_contact_effort(basis,d.qfrc_inverse.copy(),all_caps) if basis is not None else None
                c.append(a.motor_reserve-float(fit.x[-1]) if fit is not None and fit.success else -10.)
                labels.append(('exact_static_motor',int(i)))
            for g in arm_geoms:
                for obstacle in obstacles:
                    c.append((mujoco.mj_geomDistance(m,d,g,obstacle,.05,witness)-.003)/.05)
                    labels.append(('geometry',int(i),m.geom(g).name,m.geom(obstacle).name))
        cache.update(x=amplitude.copy(),c=np.asarray(c),labels=labels);calls+=1
        if calls%10==0:print('eval',calls,'minimum_constraint',float(cache['c'].min()),flush=True)
        return cache['c']
    initial=np.clip(np.zeros(4),low,high);values=constraints(initial)
    print('initial_worst',float(values.min()),cache['labels'][int(values.argmin())],flush=True)
    if a.diagnose_only:
        a.out.mkdir(parents=True,exist_ok=False)
        worst=[{'value':float(values[k]),'label':cache['labels'][k]} for k in np.argsort(values)[:10]]
        (a.out/'diagnostic.json').write_text(json.dumps(worst,indent=2)+'\n');print(json.dumps(worst));return
    options={'maxiter':100,'ftol':1e-10}
    if a.motor_reserve is not None:options['eps']=1e-5  # Resolve LP solver tolerance in numerical constraint derivatives.
    result=minimize(lambda x:.5*x@x,initial,jac=lambda x:x,method='SLSQP',bounds=Bounds(low,high),constraints=[{'type':'ineq','fun':constraints}],options=options)
    minimum=float(constraints(result.x).min())
    if not result.success or minimum< -1e-6:
        a.out.mkdir(parents=True,exist_ok=False)
        (a.out/'failure.json').write_text(json.dumps({'message':str(result.message),'amplitude_rad':result.x.tolist(),'minimum_constraint':minimum,'worst':cache['labels'][int(cache['c'].argmin())]},indent=2)+'\n')
        raise RuntimeError(f'Counterbalance rejected: {result.message}; constraint={minimum}; worst={cache["labels"][int(cache["c"].argmin())]}')
    q[np.ix_(frames,cols)]+=weights[:,None]*result.x[None,:]
    unchanged=np.ones(m.nq,dtype=bool);unchanged[cols]=False
    np.testing.assert_array_equal(q[:,unchanged],original[:,unchanged])
    a.out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(a.out/'aligned-diagnostic.npz',qpos=q,fps=fps,box_center=center,box_half_size=half)
    raw=load_g1_model();velocity=differentiate_qpos(raw,q,fps)
    export_qpos_motion(raw,q,a.out/'motion-fk.npz',fps,qvel=velocity,source=a.source)
    report={'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),'joints':names,'amplitude_rad':result.x.tolist(),'window_frames':[int(lo),int(hi)],'minimum_constraint':minimum,'root_and_legs_unchanged':True,'training_approved':False,'box':{'front_x':box.front_x,'center_y':box.center_y,'depth':box.depth,'width':box.width,'height':box.height,'friction':box.friction}}
    (a.out/'balance.json').write_text(json.dumps(report,indent=2)+'\n');(a.out/'as-run.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(report,indent=2))
    (a.out/'constraints.json').write_text(json.dumps({'static_motor_fraction':a.motor_reserve,'optimizer_options':options},indent=2)+'\n')


if __name__=='__main__':main()
