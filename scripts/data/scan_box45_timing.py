"""Check uniform timing factors using exact floating-base wrench LPs.

Geometry and motor limits remain fixed. This is an inverse-dynamics diagnostic;
even a passing factor needs dense resampling audits and continuous execution.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--scales',type=float,nargs='+',default=[1,2,3,4,8])
    p.add_argument('--write-scale',type=float)
    p.add_argument('--write-best',action='store_true',help='Export the smallest tested factor passing every sampled wrench/motor check')
    a=p.parse_args()
    if a.out.exists():p.error('output exists')
    if any(s<1 for s in a.scales):p.error('Timing factors must be at least one')
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra',check_assets=True)
    import mujoco
    import numpy as np
    from scipy.interpolate import make_interp_spline
    from scipy.spatial.transform import Rotation
    from supermimic.envs.box_scene import FixedBox
    from supermimic.data.contact_wrench import solve_contact_effort,convex_contact_points
    from supermimic.robots.dimensions import PROFILE

    z=np.load(a.source);q=z['qpos'];fps=float(z['fps']);center,half=z['box_center'],z['box_half_size']
    box=FixedBox(front_x=float(center[0]-half[0]),center_y=float(center[1]),depth=float(2*half[0]),width=float(2*half[1]),height=float(2*half[2]))
    m=mujoco.MjModel.from_binary_path(str(a.model));m.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_CONTACT)|int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
    d=mujoco.MjData(m)
    if q.ndim!=2 or q.shape[1]!=m.nq:raise ValueError('Model/reference dimensions differ')
    effort=np.array([PROFILE['effort_limit'][n] for n in PROFILE['joint_names']])
    v=np.zeros((len(q),m.nv));acc=np.zeros_like(v);forward=np.zeros(m.nv);back=np.zeros(m.nv)
    for i in range(1,len(q)-1):
        mujoco.mj_differentiatePos(m,forward,1/fps,q[i],q[i+1]);mujoco.mj_differentiatePos(m,back,-1/fps,q[i],q[i-1])
        v[i]=(forward+back)/2;acc[i]=(forward-back)*fps
    feet={}
    for g in range(m.ngeom):
        if not m.geom(g).name.rsplit('/',1)[-1].startswith('foot_'):continue
        mesh=m.geom_dataid[g]
        if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX:
            points=np.array([(x,y,z) for x in (-1,1) for y in (-1,1) for z in (-1,1)])*m.geom_size[g]
        else:
            adr,count=m.mesh_vertadr[mesh],m.mesh_vertnum[mesh];points=m.mesh_vert[adr:adr+count]
        feet[g]=points
    rays=np.array([[x*box.friction/np.sqrt(2),y*box.friction/np.sqrt(2),1] for x,y in ((1,1),(1,-1),(-1,1),(-1,-1))]).T
    jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
    factors=[None,*map(float,a.scales)];rows={str(s):[] for s in factors}
    for i in range(2,len(q)-2):
        d.qpos[:]=q[i];d.qvel[:]=0;d.qacc[:]=0;mujoco.mj_inverse(m,d);columns=[]
        for g,local in feet.items():
            world=local@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
            inside=(world[:,0]>=box.front_x)&(world[:,0]<=box.front_x+box.depth)&(abs(world[:,1]-box.center_y)<=box.width/2)
            near=world[((~inside)&(abs(world[:,2])<.012))|(inside&(abs(world[:,2]-box.height)<.012))]
            near=convex_contact_points(near)
            for point in near:
                mujoco.mj_jac(m,d,jp,jr,point,int(m.geom_bodyid[g]));columns.append(jp.T@rays)
        basis=np.concatenate(columns,axis=1) if columns else None
        for s in factors:
            d.qvel[:]=0 if s is None else v[i]/s;d.qacc[:]=0 if s is None else acc[i]/s**2
            mujoco.mj_inverse(m,d)
            fit=solve_contact_effort(basis,d.qfrc_inverse.copy(),effort) if basis is not None else None
            rows[str(s)].append({'frame':i,'wrench_feasible':bool(fit is not None and fit.success),'minimum_motor_ratio':float(fit.x[-1]) if fit is not None and fit.success else None})
        if i%100==0:print(f'checked={i}/{len(q)-4}',flush=True)
    summary={}
    for s in factors:
        r=rows[str(s)];good=[x for x in r if x['wrench_feasible']];bad=[x['frame'] for x in r if not x['wrench_feasible'] or x['minimum_motor_ratio']>1]
        summary[str(s)]={'frames':len(r),'wrench_feasible_frames':len(good),'peak_exact_motor_ratio':max((x['minimum_motor_ratio'] for x in good),default=None),'failed_frames':bad,'passed':not bad}
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'scan.json').write_text(json.dumps({'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),'summary':summary,'rows':rows,'limitations':'Pose-sampled unilateral friction-pyramid support within12mm; not executed robot dynamics.'},indent=2)+'\n')
    (a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summary,indent=2))
    if a.write_best:
        passing=[s for s in factors if s is not None and summary[str(s)]['passed']]
        a.write_scale=min(passing) if passing else None
    if a.write_scale is not None:
        s=a.write_scale
        if str(s) not in summary or not summary[str(s)]['passed']:raise ValueError('Requested factor did not pass the pose-sampled inverse audit')
        end=len(q)-round(3*fps)-1;t=np.arange(end+1)/fps
        dense=np.minimum(np.arange(round(end*s)+1)/(fps*s),t[-1])
        scalars=np.c_[q[:end+1,:3],q[:end+1,7:]]
        bc=([(1,np.zeros(scalars.shape[1])),(2,np.zeros(scalars.shape[1]))],[(1,np.zeros(scalars.shape[1])),(2,np.zeros(scalars.shape[1]))])
        scalar=make_interp_spline(t,scalars,k=5,bc_type=bc)(dense)
        first=Rotation.from_quat(q[0,[4,5,6,3]]);angles=(first.inv()*Rotation.from_quat(q[:end+1,[4,5,6,3]])).as_rotvec()
        rb=([(1,np.zeros(3)),(2,np.zeros(3))],[(1,np.zeros(3)),(2,np.zeros(3))])
        quat=(first*Rotation.from_rotvec(make_interp_spline(t,angles,k=5,bc_type=rb)(dense))).as_quat()[:,[3,0,1,2]]
        poses=np.c_[scalar[:,:3],quat,scalar[:,3:]];poses=np.r_[poses,np.repeat(poses[-1:],round(3*fps),axis=0)]
        np.savez_compressed(a.out/'aligned-diagnostic.npz',qpos=poses,fps=fps,box_center=center,box_half_size=half)
        from supermimic.data.physics_motion import differentiate_qpos,export_qpos_motion
        from supermimic.data.motion_convert import load_g1_model
        raw=load_g1_model();velocity=differentiate_qpos(raw,poses,fps)
        export_qpos_motion(raw,poses,a.out/'motion-fk.npz',fps,qvel=velocity,source=a.source)
        (a.out/'timing.json').write_text(json.dumps({'scale':s,'frames':len(poses),'fps':fps,'box':{'front_x':box.front_x,'center_y':box.center_y,'depth':box.depth,'width':box.width,'height':box.height,'friction':box.friction},'training_approved':False,'status':'Needs dense resampling audits and continuous execution'},indent=2)+'\n')


if __name__=='__main__':main()
