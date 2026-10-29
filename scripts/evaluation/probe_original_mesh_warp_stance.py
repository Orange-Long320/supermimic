"""Single-world actual-training-engine stance probe; original mesh only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['model','target','feedback','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--seconds',type=float,default=5.)
    a=p.parse_args()
    if a.out.exists() or not 2<=a.seconds<=5:p.error('New output and 2–5 seconds required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='3':raise ValueError('This bounded probe is assigned to physical GPU3 only')
    import mujoco
    import numpy as np
    import warp as wp
    import mujoco_warp as mw
    m=mujoco.MjModel.from_binary_path(str(a.model));q=np.load(a.target)['candidate'];f=np.load(a.feedback);N=f['N'];K=f['K']
    for side in ['left','right']:
        body=m.body('robot/'+side+'_ankle_roll_link').id
        ids=[g for g in range(m.ngeom) if m.geom_bodyid[g]==body and (m.geom_contype[g] or m.geom_conaffinity[g])]
        if len(ids)!=1 or m.geom_type[ids[0]]!=mujoco.mjtGeom.mjGEOM_MESH:raise ValueError('Original foot collision mesh required')
    a.out.mkdir(parents=True);(a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    (a.out/'launch.json').write_text(json.dumps({'pid':os.getpid(),'physical_gpu':3,'nworld':1,'configuration':{k:str(v) for k,v in vars(a).items()}},indent=2))
    wp.init();wp.set_device('cuda:0');wm=mw.put_model(m)
    nominal=f['nominal_control'];mujoco_data=mujoco.MjData(m);mujoco_data.qpos[:]=q;mujoco_data.ctrl[:]=nominal;mujoco.mj_forward(m,mujoco_data)
    report={'scope':__doc__,'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),'physical_geometry_changed':False,'physical_gpu':3,'nworld':1,'trials':{}}
    for mode in ['fixed','contact-feedback']:
        host=mujoco.MjData(m);host.qpos[:]=q;host.ctrl[:]=nominal
        wd=mw.put_data(m,host,nworld=1,nconmax=256,nccdmax=256,njmax=1024)
        mw.forward(wm,wd);wp.synchronize();states=[];controls=[];forces=[];constraint=[];status='completed';tangent=np.zeros(m.nv);start=time.monotonic()
        ctrl_buffer=wp.array(nominal[None,:].astype(np.float32),device=wd.qpos.device)
        for step in range(round(a.seconds/m.opt.timestep)):
            qactual=wd.qpos.numpy()[0].astype(float);vactual=wd.qvel.numpy()[0].astype(float)
            u=nominal.copy()
            if mode=='contact-feedback':
                mujoco.mj_differentiatePos(m,tangent,1.,q,qactual);error=np.r_[N.T@tangent,N.T@vactual];u-=K@error
            ctrl_buffer.assign(u[None,:].astype(np.float32));wp.copy(wd.ctrl,ctrl_buffer);mw.step(wm,wd);wp.synchronize()
            pose=wd.qpos.numpy()[0].astype(float);vel=wd.qvel.numpy()[0].astype(float)
            states.append(np.r_[pose,vel]);controls.append(u);forces.append(wd.actuator_force.numpy()[0].copy());constraint.append(wd.qfrc_constraint.numpy()[0].copy())
            hard=float(np.maximum(np.maximum(m.jnt_range[1:,0]-pose[7:],pose[7:]-m.jnt_range[1:,1]),0).max())
            mujoco_data.qpos[:]=pose;mujoco.mj_kinematics(m,mujoco_data);up=float(mujoco_data.xmat[int(m.jnt_bodyid[0])].reshape(3,3)[2,2]);drift=float(np.linalg.norm(pose[:3]-q[:3]))
            if step%100==0:print(json.dumps({'mode':mode,'time_s':(step+1)*m.opt.timestep,'hard':hard,'drift':drift,'up':up}),flush=True)
            if hard>.001 or drift>.15 or up<.9 or not np.isfinite(states[-1]).all():status='safety_stop';break
        np.savez_compressed(a.out/f'{mode}.npz',state=np.asarray(states),controls=np.asarray(controls),actuator_force=np.asarray(forces),qfrc_constraint=np.asarray(constraint),dt=m.opt.timestep)
        report['trials'][mode]={'status':status,'seconds':len(states)*m.opt.timestep,'wall_seconds':time.monotonic()-start,'last_hard':hard,'last_drift':drift,'last_up':up,'state_resets_after_initialization':0}
        (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['trials'][mode]),flush=True)


if __name__=='__main__':main()
