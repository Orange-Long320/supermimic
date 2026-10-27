"""Build a coordinated COM, planted-foot and joint guide on the native X2 model."""
import argparse,hashlib,json
from pathlib import Path
import mujoco
import numpy as np
from scipy.interpolate import BPoly
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--equilibrium',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--seconds',type=float,default=.6)
    p.add_argument('--rise',type=float,default=.04);p.add_argument('--advance',type=float,default=0.)
    p.add_argument('--lateral',type=float,default=0.)
    p.add_argument('--entry-vx',type=float,default=0.);p.add_argument('--launch-vx',type=float,default=0.)
    p.add_argument('--launch-vz',type=float,default=0.);p.add_argument('--root-pitch',type=float,default=0.)
    p.add_argument('--root-roll',type=float,default=0.)
    p.add_argument('--root-pitch-velocity',type=float,default=0.);p.add_argument('--toe-angle',type=float,default=0.)
    p.add_argument('--swing-hip',type=float);p.add_argument('--swing-knee',type=float)
    p.add_argument('--swing-hip-velocity',type=float,default=0.)
    p.add_argument('--ramped-launch',action='store_true',help='Use constant-jerk COM acceleration instead of zero acceleration at the launch boundary')
    p.add_argument('--rotation-delay-fraction',type=float,default=0.)
    p.add_argument('--toe-velocity',type=float,default=0.)
    p.add_argument('--toe-delay-fraction',type=float,default=0.)
    p.add_argument('--arm-pitch',type=float)
    p.add_argument('--arm-pitch-velocity',type=float,default=0.)
    p.add_argument('--launch-momentum',type=float,nargs=3,help='Solve endpoint root rates for a specified world angular momentum')
    p.add_argument('--fps',type=float,default=200.);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    program_source=Path(__file__).read_bytes();(args.out/'as-run.py').write_bytes(program_source)
    from supermimic.kinematics.support_reference import SupportPoseIK,root_rate_for_momentum
    from supermimic.data.physics_motion import differentiate_qpos
    from supermimic.robots.dimensions import PROFILE
    m=mujoco.MjModel.from_binary_path(str(args.model));d=mujoco.MjData(m)
    with np.load(args.equilibrium) as z:
        q0=z['qpos'].copy();assert z['joint_names'].tolist()==list(PROFILE['joint_names'])
    root=int(m.jnt_bodyid[0]);foot=m.body('robot/left_ankle_roll_link').id;d.qpos[:]=q0;mujoco.mj_forward(m,d)
    com0=d.subtree_com[root].copy();foot0=d.xpos[foot].copy();footrot=Rotation.from_matrix(d.xmat[foot].reshape(3,3))
    rootrot=Rotation.from_quat(q0[3:7][[1,2,3,0]]);meshes=[];foot_points=[]
    for g in range(m.ngeom):
        if m.body_rootid[int(m.geom_bodyid[g])]!=root or not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
        k=m.geom_dataid[g];assert k>=0;a,n=m.mesh_vertadr[k],m.mesh_vertnum[k]
        vertices=m.mesh_vert[a:a+n].astype(float);side=1 if m.geom_bodyid[g]==foot else 2 if 'foot_' in m.geom(g).name else 0
        meshes.append((g,vertices,side))
        if side==1:foot_points.extend((vertices@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]-foot0)@footrot.as_matrix())
    foot_points=np.asarray(foot_points);world=footrot.apply(foot_points)+foot0;sole=float(world[:,2].min())
    low=world[world[:,2]<sole+.01];edge=low[low[:,0]>low[:,0].max()-.008]
    pivot=np.array([low[:,0].max(),edge[:,1].mean(),sole])
    duration=args.seconds;times=np.linspace(0,duration,round(duration*args.fps)+1)
    if len(times)<3 or duration<=0:raise ValueError('At least three reference frames are required')
    effective_fps=(len(times)-1)/duration
    def curve(first,last,velocity0=None,velocity1=None):
        return BPoly.from_derivatives([0,duration],[[first,np.zeros_like(first) if velocity0 is None else velocity0,np.zeros_like(first)],
            [last,np.zeros_like(last) if velocity1 is None else velocity1,np.zeros_like(last)]])
    com_curve=curve(com0,com0+[args.advance,args.lateral,args.rise],np.array([args.entry_vx,0,0]),np.array([args.launch_vx,0,args.launch_vz]))
    pitch_curve=curve(np.zeros(3),np.array([args.root_roll,args.root_pitch,0]),velocity1=np.array([0,args.root_pitch_velocity,0]))
    if args.ramped_launch:
        if args.root_roll:raise ValueError('Ramped launch roll timing is not specified')
        v0=np.array([args.entry_vx,0,0]);v1=np.array([args.launch_vx,0,args.launch_vz])
        displacement=(2*v0+v1)*duration/3
        requested=np.array([args.advance,args.lateral,args.rise])
        if not np.allclose(displacement,requested,atol=1e-8):
            raise ValueError(f'Constant-jerk launch displacement must be {displacement.tolist()}')
        com_curve=BPoly.from_derivatives([0,duration],[[com0,v0,np.zeros(3)],
            [com0+requested,v1,2*(v1-v0)/duration]])
        delay=duration*args.rotation_delay_fraction;span=duration-delay
        if not np.isclose(args.root_pitch,args.root_pitch_velocity*span/3,atol=1e-8):
            raise ValueError('Root pitch must equal final angular velocity times rotation duration / 3')
        root_data=[[np.zeros(3),np.zeros(3),np.zeros(3)]];root_knots=[0.]
        if delay:root_knots.append(delay);root_data.append([np.zeros(3),np.zeros(3),np.zeros(3)])
        root_knots.append(duration);root_data.append([np.array([0,args.root_pitch,0]),np.array([0,args.root_pitch_velocity,0]),
            np.array([0,2*args.root_pitch_velocity/span,0])])
        pitch_curve=BPoly.from_derivatives(root_knots,root_data)
    toe_curve=curve(np.array(0.),np.array(args.toe_angle))
    if args.toe_velocity:
        toe_delay=duration*args.toe_delay_fraction;span=duration-toe_delay
        if not np.isclose(args.toe_angle,args.toe_velocity*span/3,atol=1e-8):
            raise ValueError('Ramped toe angle must equal final toe speed times toe duration / 3')
        knots=[0.];data=[[np.array(0.),np.array(0.),np.array(0.)]]
        if toe_delay:knots.append(toe_delay);data.append([np.array(0.),np.array(0.),np.array(0.)])
        knots.append(duration);data.append([np.array(args.toe_angle),np.array(args.toe_velocity),np.array(2*args.toe_velocity/span)])
        toe_curve=BPoly.from_derivatives(knots,data)
    swing=q0.copy();names=list(PROFILE['joint_names'])
    for suffix,value in [('hip_pitch_joint',args.swing_hip),('knee_joint',args.swing_knee)]:
        if value is not None:swing[7+names.index('right_'+suffix)]=value
    swing_v=np.zeros(m.nq);swing_v[7+names.index('right_hip_pitch_joint')]=args.swing_hip_velocity
    if args.arm_pitch is not None:
        for side in ['left','right']:
            swing[7+names.index(side+'_shoulder_pitch_joint')]=args.arm_pitch
            swing_v[7+names.index(side+'_shoulder_pitch_joint')]=args.arm_pitch_velocity
    swing_curve=curve(q0,swing,velocity1=swing_v);ik=SupportPoseIK(m,q0,foot);poses=[];errors=[];footposes=[];heights=[]
    def foot_pose(t):
        turn=Rotation.from_rotvec([0,float(toe_curve(t)),0]);fr=turn*footrot;fp=pivot+turn.apply(foot0-pivot)
        fp[2]+=sole-(fr.apply(foot_points)+fp)[:,2].min()
        return fp,fr
    endpoint_solve=None
    if args.launch_momentum:
        for stage in np.linspace(0,duration,13):
            fp,fr=foot_pose(stage)
            endpoint,endpoint_error=ik.fit(swing_curve(stage),com_curve(stage),fp,fr,Rotation.from_rotvec(pitch_curve(stage))*rootrot)
        angle=np.array([args.root_roll,args.root_pitch,0])
        if endpoint_error['com_error_m']>.002 or endpoint_error['root_orientation_error_rad']>.02:
            raise ValueError('Endpoint IK did not satisfy constraints for momentum matching')
        eps=1e-6;p0,r0=foot_pose(duration-eps);p1,r1=foot_pose(duration)
        twist=np.r_[(p1-p0)/eps,(r1*r0.inv()).as_rotvec()/eps]
        rates,end_velocity,residual=root_rate_for_momentum(m,endpoint,foot,swing_v[7:],com_curve.derivative()(duration),
            twist,angle,np.asarray(args.launch_momentum))
        if args.ramped_launch:
            root_data[-1]=[angle,rates,2*rates/(duration-delay)]
            pitch_curve=BPoly.from_derivatives(root_knots,root_data)
        else:pitch_curve=curve(np.zeros(3),angle,velocity1=rates)
        endpoint_solve={'root_rotation_vector_rate':rates.tolist(),'constraint_residual':residual,
            'maximum_joint_speed_ratio':float((abs(end_velocity[6:])/np.array([PROFILE['velocity_limit'][n] for n in names])).max())}
        ik.previous=q0[ik.columns].copy()
    for t in times:
        fp,fr=foot_pose(t)
        posture=swing_curve(t)
        if args.arm_pitch is not None:
            for side in ['left','right']:
                column=7+names.index(side+'_shoulder_pitch_joint');delta=args.arm_pitch-q0[column]
                if args.arm_pitch_velocity:
                    arm_span=3*delta/args.arm_pitch_velocity
                    if not 0<arm_span<=duration:raise ValueError('Ramped arm motion has an invalid duration')
                    u=max(0.,t-(duration-arm_span))/arm_span
                    posture[column]=q0[column]+delta*u**3
                else:posture[column]=swing_curve(t)[column]
        q,error=ik.fit(posture,com_curve(t),fp,fr,Rotation.from_rotvec(pitch_curve(t))*rootrot)
        poses.append(q);errors.append(error);footposes.append(np.r_[fp,fr.as_quat()]);d.qpos[:]=q;mujoco.mj_kinematics(m,d)
        heights.append([min(float((vertices@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])[:,2].min()) for g,vertices,kind in meshes if kind==side) for side in [1,2,0]])
    q=np.asarray(poses);v=differentiate_qpos(m,q,effective_fps);acc=np.gradient(v,1/effective_fps,axis=0,edge_order=2)
    com=[];cv=[];mom=[]
    for pose,vel in zip(q,v):
        d.qpos[:]=pose;d.qvel[:]=vel;mujoco.mj_forward(m,d);jp=np.zeros((3,m.nv));hj=np.zeros_like(jp)
        mujoco.mj_jacSubtreeCom(m,d,jp,root);mujoco.mj_angmomMat(m,d,hj,root)
        com.append(d.subtree_com[root].copy());cv.append(jp@vel);mom.append(hj@vel)
    cv=np.asarray(cv);mom=np.asarray(mom);feet=np.asarray(footposes);fv=np.zeros((len(times),6))
    fv[:,:3]=np.gradient(feet[:,:3],1/effective_fps,axis=0,edge_order=2)
    for i in range(1,len(times)-1):fv[i,3:]=(Rotation.from_quat(feet[i+1,3:])*Rotation.from_quat(feet[i-1,3:]).inv()).as_rotvec()*effective_fps/2
    fv[0,3:]=(Rotation.from_quat(feet[1,3:])*Rotation.from_quat(feet[0,3:]).inv()).as_rotvec()*effective_fps
    fv[-1,3:]=(Rotation.from_quat(feet[-1,3:])*Rotation.from_quat(feet[-2,3:]).inv()).as_rotvec()*effective_fps
    np.savez(args.out/'reference.npz',time=times,qpos=q,qvel=v,qacc=acc,com=np.asarray(com),com_velocity=cv,
        com_acceleration=np.gradient(cv,1/effective_fps,axis=0,edge_order=2),momentum=mom,momentum_rate=np.gradient(mom,1/effective_fps,axis=0,edge_order=2),
        foot_position=feet[:,:3],foot_quat=feet[:,3:],foot_velocity=fv,foot_acceleration=np.gradient(fv,1/effective_fps,axis=0,edge_order=2),
        joint_names=np.array(names),heights=np.asarray(heights))
    lo,hi=m.jnt_range[1:].T;limits=np.array([PROFILE['velocity_limit'][n] for n in names])
    report={'purpose':__doc__,'configuration':{k:str(x) if isinstance(x,Path) else x for k,x in vars(args).items()},
        'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),'equilibrium_sha256':hashlib.sha256(args.equilibrium.read_bytes()).hexdigest(),
        'effective_fps':effective_fps,'source_sha256':hashlib.sha256(program_source).hexdigest(),
        'endpoint_momentum_solve':endpoint_solve,
        'max_com_error':max(x['com_error_m'] for x in errors),'max_root_error':max(x['root_orientation_error_rad'] for x in errors),
        'hard_violation':float(np.maximum(np.maximum(lo-q[:,7:],q[:,7:]-hi),0).max()),
        'maximum_joint_speed_ratio':float((abs(v[:,6:])/limits).max()),'minimum_heights':np.asarray(heights).min(0).tolist(),
        'start_com_velocity':cv[0].tolist(),'end_com_velocity':cv[-1].tolist(),'end_momentum':mom[-1].tolist(),
        'ik':errors,'approved_for_training':False,'note':'Kinematics only. Native dynamic feasibility and full Webster remain unverified.'}
    (args.out/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:x for k,x in report.items() if k!='ik'}),flush=True)


if __name__=='__main__':main()
