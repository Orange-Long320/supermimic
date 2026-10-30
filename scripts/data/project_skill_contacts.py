"""Experimental bounded stance-foot projection of staged gait candidates.

This is a kinematic proposal, not a dynamic contact or stability guarantee.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('Output exists')
    from supermimic.data.motion_convert import load_g1_model,ResampledMotion,forward_kinematics
    m=load_g1_model();d=mujoco.MjData(m);manifest=json.loads((a.input/'manifest.json').read_text())
    if manifest['parameters']['release_lead']<10:raise ValueError('Need ten committed frames')
    with np.load(a.input/'candidate.npz') as z:
        names=list(z['joint_names']);body_names=list(z['body_names']);root=body_names.index('pelvis');fps=float(z['fps'][0])
        pos=z['body_pos_w'][:,root].astype(float);quat=z['body_quat_w'][:,root].astype(float);q=z['joint_pos'].astype(float)
    with np.load(a.input/'continue.npz') as z:
        cpos=z['body_pos_w'][:,root].astype(float);cquat=z['body_quat_w'][:,root].astype(float);cq=z['joint_pos'].astype(float)
    feet=[m.body(f'{s}_ankle_roll_link').id for s in ('left','right')]
    leg=np.array([i for i,n in enumerate(names) if any(x in n for x in ('hip','knee','ankle'))])
    jids=np.array([m.joint(n).id for n in names]);lo=m.jnt_range[jids[leg],0]+1e-5;hi=m.jnt_range[jids[leg],1]-1e-5
    geoms=[]
    for side in ('left','right'):
        found=[]
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g])!=m.body(side+'_ankle_roll_link').id or not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
            if m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
            mid=m.geom_dataid[g];verts=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]].copy();found.append((g,verts))
        if not found:raise ValueError('Missing original foot collision mesh: '+side)
        geoms.append(found)
    def sole():
        return np.array([min(float((v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])[:,2].min()) for g,v in group) for group in geoms])
    def project(pos,quat,q):
        n=len(q);outp=pos.copy();outq=q.copy();anchor=[None,None];anchor_rot=[None,None];prev=None;errors=[];support=[];ground_shift=[]
        for i in range(n):
            d.qpos[:]=np.r_[pos[i],quat[i],q[i]];mujoco.mj_kinematics(m,d); rawfoot=d.xpos[feet].copy();rawsole=sole()
            # Select the lower foot; double support only if both have similar clearance.
            stance=rawsole<=rawsole.min()+.015
            target=rawfoot.copy();target[:,2]-=rawsole.min()
            for s in range(2):
                if stance[s]:
                    if anchor[s] is None:
                        anchor[s]=np.r_[rawfoot[s,:2],rawfoot[s,2]-rawsole[s]]
                        anchor_rot[s]=d.xmat[feet[s]].reshape(3,3).copy()
                    target[s]=anchor[s]
                else:anchor[s]=None;anchor_rot[s]=None
            initial=np.r_[pos[i,2]-rawsole.min(),np.clip(q[i,leg],lo,hi)] if prev is None else prev.copy()
            initial[0]=np.clip(initial[0],pos[i,2]-.18,pos[i,2]+.03)
            lower=np.r_[pos[i,2]-.18,lo];upper=np.r_[pos[i,2]+.03,hi]
            def residual(x):
                d.qpos[:]=np.r_[pos[i],quat[i],q[i]];d.qpos[2]=x[0];d.qpos[7+leg]=x[1:];mujoco.mj_kinematics(m,d)
                diff=(d.xpos[feet]-target)*np.where(stance[:,None],60.,8.)
                heights=sole()
                ground=np.where(stance,heights,np.minimum(heights,0.))
                orient=np.concatenate([Rotation.from_matrix(anchor_rot[s].T@d.xmat[feet[s]].reshape(3,3)).as_rotvec() if stance[s] else np.zeros(3) for s in range(2)])
                smooth=np.zeros_like(x[1:]) if prev is None else .03*(x[1:]-prev[1:])
                return np.r_[diff.ravel(),150*ground,4*orient,.04*(x[1:]-np.clip(q[i,leg],lo,hi)),smooth,.1*(x[0]-(pos[i,2]-rawsole.min()))]
            fit=least_squares(residual,initial,bounds=(lower,upper),max_nfev=40,ftol=1e-7,xtol=1e-7,gtol=1e-7)
            residual(fit.x);outp[i,2]=fit.x[0];outq[i,leg]=fit.x[1:];prev=fit.x
            errors.append(np.linalg.norm(d.xpos[feet]-target,axis=1));support.append(stance.copy());ground_shift.append(outp[i,2]-pos[i,2])
        return outp,outq,np.asarray(errors),np.asarray(support),np.asarray(ground_shift)
    # Project the full continuing path once; the candidate shares exactly the
    # committed prefix. The braking tail is projected independently below.
    cp,cq,ce,cs,cshift=project(cpos,cquat,cq)
    prefix=manifest['release_frame']+manifest['parameters']['release_lead']+1
    pp,pq,pe,ps,shift=project(pos,quat,q)
    pp[:prefix]=cp[:prefix];pq[:prefix]=cq[:prefix];quat[:prefix]=cquat[:prefix]
    def save(name,pp,qq,joints):
        dt=1/fps; dq=(Rotation.from_quat(qq[1:,[1,2,3,0]])*Rotation.from_quat(qq[:-1,[1,2,3,0]]).inv()).as_rotvec()/dt
        ang=np.vstack([dq[:1],(dq[:-1]+dq[1:])/2,dq[-1:]])
        motion=ResampledMotion(fps,pp,qq,joints,np.gradient(pp,dt,axis=0),ang,np.gradient(joints,dt,axis=0))
        b=forward_kinematics(m,motion)
        np.savez(a.out/(name+'.npz'),fps=np.array([fps]),joint_pos=joints,joint_vel=motion.joint_vel,body_pos_w=b.pos_w,body_quat_w=b.quat_w,body_lin_vel_w=b.lin_vel_w,body_ang_vel_w=b.ang_vel_w,joint_names=np.array(names),body_names=np.array(b.body_names),source=np.array(str(a.input)))
    a.out.mkdir(parents=True)
    save('candidate',pp,quat,pq);save('continue',cp,cquat,cq)
    manifest['candidate_sha256']=hashlib.sha256((a.out/'candidate.npz').read_bytes()).hexdigest()
    manifest['contact_projection']={'source_candidate_sha256':hashlib.sha256((a.input/'candidate.npz').read_bytes()).hexdigest(),
        'max_ankle_target_error_m':float(pe.max()),'support_max_ankle_target_error_m':float(pe[ps].max()),
        'min_max_root_height_shift_m':[float(shift.min()),float(shift.max())],
        'prefix_frames_shared':prefix,'note':'Kinematic ankle proxy and mesh clearance; support is inferred, not measured force.'}
    manifest['scope']='Contact-projected experimental candidate; dynamic validity and stop acceptance unverified'
    np.savez(a.out/'projection-diagnostics.npz',target_error=pe,support=ps,root_height_shift=shift)
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest['contact_projection'],indent=2))

if __name__=='__main__':main()
