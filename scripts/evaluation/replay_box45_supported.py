"""Native joint servo with support-consistent reference feedforward.

Estimated support forces only compute joint torque offsets; they are never
applied to the simulator. Actual motion uses the unchanged native joint motors,
gravity and contact law, with no state resets after initialization.
"""
import argparse
import hashlib
import json
from pathlib import Path
import os


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--box',type=Path,required=True)
    p.add_argument('--start-frame',type=int,default=0)
    p.add_argument('--seconds',type=float,default=3)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('output exists')
    import mujoco
    import numpy as np
    from supermimic.envs.box_scene import FixedBox
    from supermimic.data.contact_wrench import solve_contact_effort
    from supermimic.robots.dimensions import PROFILE
    m=mujoco.MjModel.from_binary_path(str(a.model));actual=mujoco.MjData(m)
    inverse=mujoco.MjModel.from_binary_path(str(a.model))
    inverse.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_CONTACT)|int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
    desired=mujoco.MjData(inverse)
    z=np.load(a.reference);q=z['qpos'];v=z['qvel'];fps=float(z['fps'].reshape(-1)[0])
    box=FixedBox(**json.loads(a.box.read_text())['box'])
    if not 0<=a.start_frame<len(q):raise ValueError('Invalid start frame')
    cols=m.jnt_qposadr[m.actuator_trnid[:,0]];vcols=m.jnt_dofadr[m.actuator_trnid[:,0]]
    np.testing.assert_array_equal(vcols,np.arange(6,m.nv))
    effort=np.array([PROFILE['effort_limit'][name] for name in PROFILE['joint_names']])
    kp=m.actuator_gainprm[:,0];kd=-m.actuator_biasprm[:,2]
    foot_geoms=[g for g in range(m.ngeom) if m.geom(g).name.rsplit('/',1)[-1].startswith('foot_')]
    meshes={}
    for g in foot_geoms:
        mesh=m.geom_dataid[g]
        if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX:
            local=np.array([(x,y,z) for x in (-1,1) for y in (-1,1) for z in (-1,1)])*m.geom_size[g]
        elif mesh>=0:
            adr,count=m.mesh_vertadr[mesh],m.mesh_vertnum[mesh];local=m.mesh_vert[adr:adr+count].copy()
        else:raise ValueError('Unsupported foot geometry')
        meshes[g]=local
    acceleration=np.gradient(v,1/fps,axis=0)
    rays=np.array([[x/np.sqrt(2),y/np.sqrt(2),1] for x,y in ((1,1),(1,-1),(-1,1),(-1,-1))]).T
    jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);cache={}
    def feedforward(frame):
        if frame in cache:return cache[frame]
        desired.qpos[:]=q[frame];desired.qvel[:]=v[frame];desired.qacc[:]=acceleration[frame]
        mujoco.mj_inverse(inverse,desired);required=desired.qfrc_inverse.copy();columns=[]
        for g,local in meshes.items():
            world=local@desired.geom_xmat[g].reshape(3,3).T+desired.geom_xpos[g]
            inside=(world[:,0]>=box.front_x)&(world[:,0]<=box.front_x+box.depth)&(abs(world[:,1]-box.center_y)<=box.width/2)
            mask=((~inside)&(abs(world[:,2])<.012))|(inside&(abs(world[:,2]-box.height)<.012))
            points=world[mask]
            if len(points)>12:points=points[np.linspace(0,len(points)-1,12).astype(int)]
            for point in points:
                mujoco.mj_jac(inverse,desired,jp,jr,point,int(inverse.geom_bodyid[g]));columns.append(jp.T@rays)
        if not columns:raise RuntimeError(f'No planned support at frame {frame}')
        basis=np.concatenate(columns,axis=1);fit=solve_contact_effort(basis,required,effort)
        if not fit.success or fit.x[-1]>1:raise RuntimeError(f'Reference wrench/motor infeasible at {frame}: {fit.message}')
        tau=(required-basis@fit.x[:-1])[6:]
        cache[frame]=(tau,float(fit.x[-1]));return cache[frame]
    actual.qpos[:]=q[a.start_frame];actual.qvel[:]=v[a.start_frame]
    mujoco.mj_forward(m,actual)
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    root=int(m.jnt_bodyid[0]);rows=[];states=[];controls=[];status='completed';failure=None
    bounds=m.jnt_range[1:];jac=np.zeros((3,m.nv))
    for i in range(round(a.seconds/m.opt.timestep)):
        f=min(a.start_frame+i*m.opt.timestep*fps,len(q)-1);lo=int(f);hi=min(lo+1,len(q)-1);blend=f-lo
        try:
            tau,ratio=feedforward(lo);tau2,_=feedforward(hi)
        except RuntimeError as e:status='reference_rejected';failure=str(e);break
        target=q[lo]*(1-blend)+q[hi]*blend;target_v=v[lo]*(1-blend)+v[hi]*blend
        ff=tau*(1-blend)+tau2*blend
        actual.ctrl[:]=target[cols]+(ff+kd*target_v[vcols])/kp
        controls.append(actual.ctrl.copy());mujoco.mj_step(m,actual);mujoco.mj_forward(m,actual)
        mujoco.mj_jacSubtreeCom(m,actual,jac,root)
        hard=float(np.maximum(np.maximum(bounds[:,0]-actual.qpos[7:],actual.qpos[7:]-bounds[:,1]),0).max())
        row={'time_s':(i+1)*m.opt.timestep,'root_height_m':float(actual.qpos[2]),'root_up_z':float(actual.xmat[root].reshape(3,3)[2,2]),'hard_violation_rad':hard,'motor_ratio':float((abs(actual.actuator_force)/effort).max()),'com_speed_m_s':float(np.linalg.norm(jac@actual.qvel)),'reference_motor_ratio':ratio}
        rows.append(row);states.append(np.r_[actual.qpos.copy(),actual.qvel.copy()])
        if hard>.001 or row['root_up_z']<.3 or row['root_height_m']<.22:status='safety_stop';break
    np.savez_compressed(a.out/'trajectory.npz',state=states,controls=controls,dt=m.opt.timestep)
    report={'status':status,'failure':failure,'seconds_executed':len(rows)*m.opt.timestep,'maximum_hard_violation_rad':max((r['hard_violation_rad'] for r in rows),default=0),'maximum_motor_ratio':max((r['motor_ratio'] for r in rows),default=0),'last':rows[-1] if rows else None,'rows':rows,'state_resets_after_initialization':0,'external_root_forces':False,'policy_evaluation':False,'approved_for_training':False,'pid':os.getpid(),'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),'reference_sha256':hashlib.sha256(a.reference.read_bytes()).hexdigest()}
    (a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'}))


if __name__=='__main__':main()
