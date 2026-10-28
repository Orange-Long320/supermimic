"""Joint pose/contact-force refinement of an existing Webster settling candidate.

Uses the unchanged robot inertias, armatures, soft posture limits and effort
limits. The contact cone is an offline approximation; native replay is separate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares, nnls
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=Path,required=True,help='qpos.npz with a full motion and fps')
    p.add_argument('--template',type=Path,required=True,help='NPZ metadata/contact annotation source')
    p.add_argument('--start-frame',type=int,default=83)
    p.add_argument('--end-frame',type=int,default=143)
    p.add_argument('--max-nfev',type=int,default=100)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resume',type=Path)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    (args.out/'launch.json').write_text(json.dumps({'pid':os.getpid(),'host':socket.gethostname(),
        'command':[sys.executable,*sys.argv],'python':sys.version,'mujoco':mujoco.__version__,
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.data.physics_motion import export_qpos_motion,differentiate_qpos
    from supermimic.data.contact_wrench import solve_contact_effort
    from supermimic.robots.dimensions import PROFILE
    from supermimic.robots.assets import get_robot_cfg
    m=load_g1_model();d=mujoco.MjData(m)
    for name in PROFILE['joint_names']:
        m.dof_armature[m.jnt_dofadr[m.joint(name).id]]=PROFILE['armature'][name]
    m.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_CONTACT)|int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
    seed=np.load(args.seed);reference=seed['qpos'].copy();fps=float(seed['fps'].reshape(-1)[0])
    start,end=args.start_frame,args.end_frame
    if not 2<=start<end<len(reference)-2:raise ValueError('Invalid refinement interval')
    names=list(PROFILE['joint_names'])
    np.testing.assert_array_equal(seed['joint_names'],names)
    columns=np.array([m.jnt_qposadr[m.joint(n).id] for n in names
        if any(n.startswith(s) for s in ['left_hip_','right_hip_','left_knee','right_knee','left_ankle_','right_ankle_','waist_'])])
    joint_ids=np.array([m.dof_jntid[c-1] for c in columns]);width=6+len(columns)
    factor=get_robot_cfg().articulation.soft_joint_pos_limit_factor
    ranges=m.jnt_range[joint_ids];centre=ranges.mean(axis=1);half=np.diff(ranges,axis=1)[:,0]*factor/2
    lower_j,upper_j=centre-half+.005,centre+half-.005
    effort=np.array([PROFILE['effort_limit'][n] for n in names])
    velocity_limits=np.array([PROFILE['velocity_limit'][n] for n in names])
    root=int(m.jnt_bodyid[0]);weight=float(m.body_subtreemass[root]*9.81)
    scales=np.r_[np.full(3,weight),np.full(3,weight*.7)]
    rays=np.array([[x/np.sqrt(2),y/np.sqrt(2),1.] for x,y in [(1,1),(1,-1),(-1,1),(-1,-1)]]).T
    feet={s:m.body(s+'_ankle_roll_link').id for s in ['left','right']}
    meshes={}
    for side,b in feet.items():
        parts=[]
        for g in range(m.ngeom):
            if m.geom_bodyid[g]!=b or not(m.geom_contype[g] or m.geom_conaffinity[g]):continue
            k=int(m.geom_dataid[g]);a,n=m.mesh_vertadr[k],m.mesh_vertnum[k]
            parts.append((g,m.mesh_vert[a:a+n].copy()))
        meshes[side]=parts
    def forward(q):
        d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
    def vertices(side):
        return np.concatenate([v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g] for g,v in meshes[side]])
    references=[];contacts=[]
    for frame in range(start,end+1):
        forward(reference[frame]);refs={s:(d.xpos[b].copy(),Rotation.from_matrix(d.xmat[b].reshape(3,3))) for s,b in feet.items()}
        points=[]
        for side,b in feet.items():
            world=vertices(side);near=world[world[:,2]<.012]
            if len(near):
                # Actual collision vertices at directional extremes, without
                # inventing a rectangular support patch outside the mesh.
                directions=np.array([[1,1],[1,-1],[-1,1],[-1,-1]],dtype=float)
                picked=np.unique(np.argmax(near[:,:2]@directions.T,axis=0))
                local=(near[picked]-d.xpos[b])@d.xmat[b].reshape(3,3)
                points.extend((b,x) for x in local)
        references.append(refs);contacts.append(points)
    jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv))
    tangent=np.zeros(m.nv);backward=np.zeros(m.nv)
    def inverse_and_basis(qs,frame,points):
        q=qs[frame];d.qpos[:]=q
        mujoco.mj_differentiatePos(m,tangent,1/fps,q,qs[frame+1])
        mujoco.mj_differentiatePos(m,backward,-1/fps,q,qs[frame-1])
        d.qvel[:]=(tangent+backward)/2;d.qacc[:]=(tangent-backward)*fps
        mujoco.mj_inverse(m,d);required=d.qfrc_inverse.copy();bases=[]
        for b,local in points:
            point=d.xpos[b]+d.xmat[b].reshape(3,3)@local
            mujoco.mj_jac(m,d,jp,jr,point,b);bases.append(jp.T@rays)
        basis=np.concatenate(bases,axis=1) if bases else np.zeros((m.nv,0))
        return required,basis
    n=end-start-1
    count=end-start+1
    force_sizes=[4*len(points) for points in contacts]
    force_offsets=np.r_[n*width,n*width+np.cumsum(force_sizes)]
    total=int(force_offsets[-1])
    initial=np.zeros(total);low=np.full(total,-np.inf);high=np.full(total,np.inf)
    rotations=Rotation.from_quat(reference[:,[4,5,6,3]])
    for i,frame in enumerate(range(start+1,end)):
        block=slice(i*width,(i+1)*width)
        initial[block]=np.r_[reference[frame,:3],np.zeros(3),reference[frame,columns]]
        low[block]=np.r_[reference[frame,:3]-[.12,.12,.10],np.full(3,-.2),lower_j]
        high[block]=np.r_[reference[frame,:3]+[.12,.12,.10],np.full(3,.2),upper_j]
    if args.resume:
        resumed=np.load(args.resume)['variables']
        if resumed.shape!=initial.shape:raise ValueError('Resume variable shape mismatch')
        initial=resumed
    else:
        for i,frame in enumerate(range(start,end+1)):
            required,basis=inverse_and_basis(reference,frame,contacts[i])
            if basis.shape[1]:
                coefficients,_=nnls(basis[:6]/scales[:,None],required[:6]/scales,maxiter=10000)
                initial[force_offsets[i]:force_offsets[i+1]]=coefficients/weight
    low[n*width:]=0.;high[n*width:]=4.
    def unpack(x):
        qs=reference.copy()
        for i,frame in enumerate(range(start+1,end)):
            values=x[i*width:(i+1)*width];qs[frame,:3]=values[:3]
            qs[frame,3:7]=(rotations[frame]*Rotation.from_rotvec(values[3:6])).as_quat()[[3,0,1,2]]
            qs[frame,columns]=values[6:]
        return qs
    blocks=[];row=0
    for i,frame in enumerate(range(start,end+1)):
        rows=6+(m.nv-6)+force_sizes[i]+12+width
        blocks.append((row,row+rows));row+=rows
    smooth_rows=(end-start+3)*width
    sparsity=lil_matrix((row+smooth_rows,total),dtype=int)
    for i,frame in enumerate(range(start,end+1)):
        a,b=blocks[i]
        for f in range(max(start+1,frame-1),min(end-1,frame+1)+1):
            j=f-start-1;sparsity[a:b,j*width:(j+1)*width]=1
        sparsity[a:b,force_offsets[i]:force_offsets[i+1]]=1
    for i in range(end-start+3):
        frame=start-1+i
        for f in range(max(start+1,frame-1),min(end-1,frame+1)+1):
            j=f-start-1;sparsity[row+i*width:row+(i+1)*width,j*width:(j+1)*width]=1
    def residual(x):
        qs=unpack(x);result=[]
        for i,frame in enumerate(range(start,end+1)):
            required,basis=inverse_and_basis(qs,frame,contacts[i])
            coefficients=x[force_offsets[i]:force_offsets[i+1]]*weight
            force=required-basis@coefficients
            # Motor hinges cannot be reduced by inventing an actuated root.
            result.extend(force[:6]/scales*200)
            result.extend(np.maximum(abs(force[6:])/effort-.95,0)*30)
            result.extend(x[force_offsets[i]:force_offsets[i+1]]*.01)
            for side,b in feet.items():
                target_p,target_r=references[i][side]
                result.extend((d.xpos[b]-target_p)*1500)
                result.extend((target_r.inv()*Rotation.from_matrix(d.xmat[b].reshape(3,3))).as_rotvec()*200)
            delta=(rotations[frame].inv()*Rotation.from_quat(qs[frame,[4,5,6,3]])).as_rotvec()
            result.extend(np.r_[(qs[frame,:3]-reference[frame,:3])*10,delta*5,
                                (qs[frame,columns]-reference[frame,columns])])
        # Include the fixed incoming samples and terminal hold in acceleration
        # regularization, so the seam cannot disappear outside the objective.
        section=qs[start-2:end+3]
        angles=(rotations[start].inv()*Rotation.from_quat(section[:,[4,5,6,3]])).as_rotvec()
        coordinates=np.c_[section[:,:3],angles,section[:,columns]]
        smooth=np.diff(coordinates,n=2,axis=0)*np.r_[np.full(3,500),np.full(3,50),np.full(len(columns),5)]
        result.extend(smooth.ravel())
        return np.asarray(result)
    iteration={'value':0}
    def checkpoint(x):
        iteration['value']+=1
        if iteration['value']%5==0:
            r=residual(x)
            np.savez(args.out/f'optimization-{iteration["value"]:03d}.npz',qpos=unpack(x),variables=x)
            print(json.dumps({'iteration':iteration['value'],'cost':float(r@r/2)}),flush=True)
    solved=least_squares(residual,np.clip(initial,low,high),bounds=(low,high),
        jac_sparsity=sparsity.tocsr(),x_scale='jac',tr_options={'maxiter':100},max_nfev=args.max_nfev,
        callback=checkpoint,ftol=1e-6,xtol=1e-7,gtol=1e-6,verbose=1)
    qs=unpack(solved.x);v=differentiate_qpos(m,qs,fps)
    np.savez(args.out/'qpos.npz',qpos=qs,qvel=v,fps=np.array([fps]),joint_names=np.array(names),variables=solved.x)
    export_qpos_motion(m,qs,args.out/'motion-fk.npz',fps,qvel=v,source=args.seed)
    with np.load(args.out/'motion-fk.npz') as z:arrays={k:z[k].copy() for k in z.files}
    np.savez(args.out/'webster.npz',**arrays,diagnostic_only=np.array(True),approved_for_training=np.array(False))
    rows=[]
    for i,frame in enumerate(range(start,end+1)):
        required,basis=inverse_and_basis(qs,frame,contacts[i])
        force=required-basis@solved.x[force_offsets[i]:force_offsets[i+1]]*weight
        lp=solve_contact_effort(basis,required,effort) if basis.shape[1] else None
        rows.append({'frame':frame,'root_residual_normalized':float(abs(force[:6]/scales).max()),
                     'motor_ratio':float((abs(force[6:])/effort).max()),
                     'contact_lp_feasible':bool(lp is not None and lp.success),
                     'contact_lp_peak_motor_ratio':float(lp.x[-1]) if lp is not None and lp.success else None})
    full_range=m.jnt_range[1:];c=full_range.mean(1);h=np.diff(full_range,axis=1)[:,0]*factor/2
    report={'configuration':{k:str(x) if isinstance(x,Path) else x for k,x in vars(args).items()},
            'seed_sha256':hashlib.sha256(args.seed.read_bytes()).hexdigest(),
            'output_sha256':hashlib.sha256((args.out/'webster.npz').read_bytes()).hexdigest(),
            'optimizer_success':bool(solved.success),'nfev':int(solved.nfev),'cost':float(solved.cost),
            'soft_violation_rad':float(np.maximum(abs(qs[:,7:]-c)-h,0).max()),
            'maximum_adjacent_speed_ratio':float((abs(np.diff(qs[:,7:],axis=0))*fps/velocity_limits).max()),
            'maximum_root_residual':max(x['root_residual_normalized'] for x in rows),
            'maximum_motor_ratio':max(x['motor_ratio'] for x in rows),
            'feasible_contact_frames':sum(x['contact_lp_feasible'] for x in rows),'contact_frames':len(rows),
            'rows':rows,'approved_for_training':False,'dynamic_tracking_verified':False,
            'limitations':'Fixed candidate contact schedule and approximate point-force friction cone. Independent geometry, dense contact audit and unchanged native dynamics remain necessary.'}
    (args.out/'result.json').write_text(json.dumps(report,indent=2))
    (args.out/'contacts.json').write_bytes((args.template.parent/'contacts.json').read_bytes())
    print(json.dumps({k:x for k,x in report.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
