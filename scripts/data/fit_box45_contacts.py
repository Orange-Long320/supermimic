"""Fit a diagnostic X2 guide to ground/45cm box contact targets on the server.

Keeps source upper-body joints and root orientation, solves root translation and
leg IK within soft limits. This creates a new constrained guide, not a calibrated
video reconstruction or a dynamically validated expert. Never approves training.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import socket
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--max-nfev',type=int,default=80)
    parser.add_argument('--time-scale',type=float,default=2.0,help='Slow the guide before appending terminal hold')
    parser.add_argument('--toe-support',action='store_true',help='Allow trailing left heel lift during support transfer')
    parser.add_argument('--collision-clearance',action='store_true',help='Penalize robot hull vertices inside the box')
    args=parser.parse_args()
    if args.out.exists():parser.error('output already exists')
    if args.time_scale<1 or args.max_nfev<1:parser.error('time-scale >=1 and positive max-nfev required')
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra',check_assets=True)
    import mujoco
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from scipy.spatial import ConvexHull
    from supermimic.robots.assets import get_spec
    from supermimic.robots.dimensions import PROFILE
    from supermimic.envs.box_scene import FixedBox
    from supermimic.data.physics_motion import differentiate_qpos,export_qpos_motion
    from supermimic.data.motion_convert import resample_motion

    root=Path(__file__).resolve().parents[2]
    timing_path=root/'configs/video/box45-contacts.json'
    timing=json.loads(timing_path.read_text())
    source=np.load(args.source)
    original=source['qpos'].copy();fps=float(source['fps'])
    model=get_spec().compile();data=mujoco.MjData(model)
    feet={s:model.body(s+'_ankle_roll_link').id for s in ('left','right')}
    meshes={s:[] for s in feet}
    for g in range(model.ngeom):
        side=next((s for s,b in feet.items() if model.geom_bodyid[g]==b and model.geom(g).name.startswith('foot_')),None)
        if side is None:continue
        if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX:
            v=np.array([(x,y,z) for x in (-1,1) for y in (-1,1) for z in (-1,1)])*model.geom_size[g]
        else:
            mesh=model.geom_dataid[g]
            if mesh<0:raise ValueError('Unsupported foot geometry')
            start,count=model.mesh_vertadr[mesh],model.mesh_vertnum[mesh]
            v=model.mesh_vert[start:start+count].copy()
        meshes[side].append((g,v))
    if not all(meshes.values()):raise ValueError('Missing native foot meshes')
    def update(q):
        data.qpos[:]=q;mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
    def vertices(side):
        return np.concatenate([v@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g] for g,v in meshes[side]])
    update(original[0])
    sole_local={};toe_local={};ground={}
    for side,body in feet.items():
        v=vertices(side);rot=data.xmat[body].reshape(3,3)
        local=(v-data.xpos[body])@rot
        sole_local[side]=np.r_[(local[:,:2].min(0)+local[:,:2].max(0))/2,local[:,2].min()]
        toe_local[side]=sole_local[side].copy();toe_local[side][0]=local[:,0].max()
        ground[side]=np.r_[(v[:,:2].min(0)+v[:,:2].max(0))/2,.003]
    front=max(vertices(s)[:,0].max() for s in feet)+.12
    box=FixedBox(front_x=float(front),center_y=float(np.mean([p[1] for p in ground.values()])))
    collision_vertices=[]
    if args.collision_clearance:
        for g in range(model.ngeom):
            if model.geom_bodyid[g]==0 or not (model.geom_contype[g] or model.geom_conaffinity[g]):continue
            mesh=model.geom_dataid[g]
            if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX:
                local=np.array([(x,y,z) for x in (-1,1) for y in (-1,1) for z in (-1,1)])*model.geom_size[g]
            elif mesh>=0:
                start,count=model.mesh_vertadr[mesh],model.mesh_vertnum[mesh]
                local=model.mesh_vert[start:start+count].copy();local=local[ConvexHull(local).vertices]
            else:raise ValueError('Unsupported body collision geometry')
            collision_vertices.append((g,local,0.0 if model.geom(g).name.startswith('foot_') else .005))
    top={s:np.array([front+.27,box.center_y+(.12 if s=='left' else -.12),box.height+.003]) for s in feet}
    columns=np.array([int(model.jnt_qposadr[model.joint(s+'_'+j+'_joint').id]) for s in feet for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']])
    bounds=model.jnt_range[1:];centre=bounds.mean(1);half=np.diff(bounds,axis=1)[:,0]*.9/2
    low=np.r_[[-1,-1,.25],(centre-half)[columns-7]+1e-6]
    high=np.r_[[2,1,1.7],(centre+half)[columns-7]-1e-6]
    smooth=lambda x: (lambda u:u**3*(10+u*(-15+6*u)))(float(np.clip(x,0,1)))
    def foot_target(side,t):
        lift,land=[timing[side][k] for k in ('lift_seconds','land_seconds')]
        u=(t-lift)/(land-lift)
        pitch=.65*smooth((t-1.9)/.3) if args.toe_support and side=='left' else 0.0
        rotation=Rotation.from_euler('y',pitch)
        toe_world=ground[side]+toe_local[side]-sole_local[side]
        lift_center=toe_world+Rotation.from_euler('y',.65).apply(sole_local[side]-toe_local[side]) if args.toe_support and side=='left' else ground[side]
        if u<=0:return (toe_world-rotation.apply(toe_local[side]),rotation) if pitch else (ground[side]-sole_local[side],Rotation.identity())
        if u>=1:return top[side]-sole_local[side],Rotation.identity()
        rotation=Rotation.from_euler('y',(.65*(1-smooth(u/.35))) if args.toe_support and side=='left' else 0.0)
        apex=box.height+.09
        if u<.35:
            p=lift_center.copy();p[2]+=(apex-p[2])*smooth(u/.35)
        elif u<.75:
            p=lift_center+(top[side]-lift_center)*smooth((u-.35)/.4);p[2]=apex
        else:
            p=top[side].copy();p[2]+=(apex-p[2])*(1-smooth((u-.75)/.25))
        return p-rotation.apply(sole_local[side]),rotation
    initial_com=data.subtree_com[int(model.jnt_bodyid[0])].copy()
    initial_com[:2]=(ground['left'][:2]+ground['right'][:2])/2
    n=min(len(original),round(3.4*fps)+1)
    poses=[];errors=[];previous=None
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    (args.out/'contacts.json').write_bytes(timing_path.read_bytes())
    (args.out/'launch.json').write_text(json.dumps({'pid':__import__('os').getpid(),'host':socket.gethostname(),'argv':[sys.executable,*sys.argv]},indent=2)+'\n')
    for i,base in enumerate(original[:n]):
        t=i/fps;targets={s:foot_target(s,t) for s in feet}
        if t<1.0:com_xy=initial_com[:2]+(ground['left'][:2]-initial_com[:2])*smooth(t/.9)
        elif t<1.9:com_xy=ground['left'][:2]
        elif t<2.2:com_xy=ground['left'][:2]+(top['right'][:2]-ground['left'][:2])*smooth((t-1.9)/.3)
        elif t<2.7:com_xy=top['right'][:2]
        else:com_xy=top['right'][:2]+((top['left'][:2]+top['right'][:2])/2-top['right'][:2])*smooth((t-2.7)/.5)
        root_z=original[0,2]+box.height*smooth((t-2.15)/.65)
        init=np.r_[base[:3],base[columns]] if previous is None else previous
        def pose(x):
            q=base.copy();q[:3]=x[:3];q[columns]=x[3:];return q
        def residual(x):
            update(pose(x));r=[]
            for side,body in feet.items():
                target_p,target_r=targets[side]
                r.extend((data.xpos[body]-target_p)*500)
                r.extend((target_r.inv()*Rotation.from_matrix(data.xmat[body].reshape(3,3))).as_rotvec()*100)
            for g,local,margin in collision_vertices:
                points=local@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
                delta=np.abs(points-np.asarray(box.center))-np.asarray(box.half_size)
                distance=np.linalg.norm(np.maximum(delta,0),axis=1)+np.minimum(delta.max(1),0)
                r.extend(np.maximum(margin-distance,0)*1000)
            r.extend((data.subtree_com[int(model.jnt_bodyid[0])][:2]-com_xy)*30)
            r.append((x[2]-root_z)*5)
            r.extend((x[3:]-base[columns])*.2)
            if previous is not None:r.extend((x-previous)*.2)
            return np.asarray(r)
        fit=least_squares(residual,np.clip(init,low+1e-8,high-1e-8),bounds=(low,high),max_nfev=args.max_nfev,ftol=1e-7,xtol=1e-7,gtol=1e-7)
        previous=fit.x;poses.append(pose(fit.x));update(poses[-1])
        errors.append(max(np.linalg.norm(data.xpos[b]-targets[s][0]) for s,b in feet.items()))
        if i%25==0:print(f'frame={i}/{n} foot_target_error_m={errors[-1]:.6f}',flush=True)
    guide=np.asarray(poses)
    slowed=resample_motion(np.c_[guide[:,:3],guide[:,[4,5,6,3]],guide[:,7:]],fps/args.time_scale,fps)
    q=np.c_[slowed.root_pos,slowed.root_quat,slowed.joint_pos]
    q=np.r_[q,np.repeat(q[-1:],round(3*fps),axis=0)]
    v=differentiate_qpos(model,q,fps)
    np.savez_compressed(args.out/'aligned-diagnostic.npz',qpos=q,fps=fps,box_center=box.center,box_half_size=box.half_size)
    export_qpos_motion(model,q,args.out/'motion-fk.npz',fps,qvel=v,source=args.source)
    report={'source':str(args.source.resolve()),'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
            'box':asdict(box),'frames':len(q),'fps':fps,'max_foot_target_error_m':float(max(errors)),
            'max_leg_change_rad':float(abs(guide[:,columns]-original[:n,columns]).max()),
            'time_scale':args.time_scale,
            'toe_support':args.toe_support,'collision_clearance':args.collision_clearance,
            'max_joint_velocity_rad_s':float(abs(v[:,6:]).max()),'terminal_hold_seconds':3,
            'training_approved':False,'status':'contact IK guide only; collision, speed, torque and continuous dynamics unvalidated'}
    (args.out/'fit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
