"""Robot-designed Webster guide: fixed left sole, crouch, swing and tucked flight.

Geometry and an aerial centroidal fit are only guide checks. This file never
approves training: executable motion still needs motor-bounded project rollouts.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import BPoly, CubicHermiteSpline
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ready-config",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--preparation",type=float,default=.9)
    p.add_argument("--flight",type=float,default=.4)
    p.add_argument("--launch-com-z",type=float,default=.68)
    p.add_argument("--crouch-com-z",type=float,default=.50)
    p.add_argument("--initial-com-z",type=float,default=.64)
    p.add_argument("--com-advance",type=float,default=.15)
    p.add_argument("--swing",choices=("backward","forward"),default="backward")
    p.add_argument("--stance-only",action="store_true",
                   help="Audit support geometry without expensive aerial shooting")
    p.add_argument("--joint-bounds",choices=("hard","soft"),default="hard",
                   help="Use actual mechanical ranges; soft limits are a separate posture penalty")
    p.add_argument("--joint-margin",type=float,default=.02)
    p.add_argument("--tuck-hip",type=float,default=-2.)
    p.add_argument("--tuck-knee",type=float,default=2.1)
    p.add_argument("--tuck-waist",type=float,default=0.)
    p.add_argument("--projection-iterations",type=int,default=2)
    p.add_argument("--root-orientation-slack",type=float,default=.6,
                   help="Allow contact IK to adjust torso orientation (radians per axis)")
    args=p.parse_args()
    from supermimic.data.motion_convert import load_g1_model,convert_csv_to_npz
    from supermimic.data.centroidal_projection import project_flight
    from supermimic.data.contact_wrench import solve_contact_effort
    from supermimic.robots.dimensions import PROFILE
    m=load_g1_model();d=mujoco.MjData(m);root=int(m.jnt_bodyid[0])
    ready=np.array(json.loads(args.ready_config.read_text())["initial_qpos"],dtype=float)
    names=list(PROFILE["joint_names"])
    effort=np.array([PROFILE["effort_limit"][n] for n in names])
    for n in names:m.dof_armature[m.jnt_dofadr[m.joint(n).id]]=PROFILE["armature"][n]
    j=lambda name:7+names.index(name)
    centre=m.jnt_range[1:].mean(axis=1)
    soft_half=.45*np.diff(m.jnt_range[1:],axis=1).ravel()
    soft_lower,soft_upper=centre-soft_half,centre+soft_half
    hard_lower,hard_upper=m.jnt_range[1:].T
    lower=(hard_lower if args.joint_bounds=="hard" else soft_lower)+args.joint_margin
    upper=(hard_upper if args.joint_bounds=="hard" else soft_upper)-args.joint_margin
    feet={side:m.body(f"{side}_ankle_roll_link").id for side in ("left","right")}
    meshes=[]
    for g in range(m.ngeom):
        if m.geom(g).name.startswith("foot_"):
            k=m.geom_dataid[g];a,n=m.mesh_vertadr[k],m.mesh_vertnum[k]
            meshes.append((g,m.mesh_vert[a:a+n].copy()))
    def fk(row):
        d.qpos[:]=row;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        foot={s:(d.xpos[b].copy(),Rotation.from_matrix(d.xmat[b].reshape(3,3))) for s,b in feet.items()}
        sole=min((v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])[:,2].min() for g,v in meshes)
        return d.subtree_com[root].copy(),foot,float(sole)
    ready[2]+=.005-fk(ready)[2]
    com_ready,anchors,_=fk(ready)
    # The inherited ankle orientation leaves the main sole face tilted by
    # several degrees. Flatten the largest downward hull face, rather than
    # preserving that orientation and balancing on its lowest edge.
    flattening={}
    for side,body in feet.items():
        for g,vertices in meshes:
            if m.geom_bodyid[g]!=body:continue
            hull=ConvexHull(vertices.astype(float));tri=vertices[hull.simplices]
            areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
            planes,inverse=np.unique(np.round(hull.equations,5),axis=0,return_inverse=True)
            area=np.bincount(inverse,weights=areas)
            world_normals=planes[:,:3]@d.geom_xmat[g].reshape(3,3).T
            eligible=np.flatnonzero(world_normals[:,2]<-.8)
            if not len(eligible):raise ValueError(f"No downward sole face: {side}")
            normal=world_normals[eligible[np.argmax(area[eligible])]]
            normal/=np.linalg.norm(normal)
            correction=Rotation.align_vectors(np.array([[0.,0.,-1.]]),normal[None])[0]
            position,orientation=anchors[side]
            world=vertices@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
            new_orientation=correction*orientation
            moved=correction.apply(world-position)+position
            new_position=position.copy();new_position[2]+=.005-moved[:,2].min()
            anchors[side]=(new_position,new_orientation)
            flattening[side]=float(correction.magnitude())
    indices={s:np.array([j(f"{s}_{n}_joint") for n in ["hip_pitch","hip_roll","hip_yaw","knee","ankle_pitch","ankle_roll"]]) for s in feet}
    errors=[]
    def fit_pose(reference,target,active,previous=None):
        columns=np.concatenate([np.arange(3),*[indices[s] for s in active]])
        lo=np.concatenate([[-3,-3,.15],*[lower[indices[s]-7] for s in active]])
        hi=np.concatenate([[3,3,1.2],*[upper[indices[s]-7] for s in active]])
        if previous is not None:
            delta=np.r_[np.full(3,.08),np.full(len(columns)-3,.35)]
            lo=np.maximum(lo,previous[columns]-delta);hi=np.minimum(hi,previous[columns]+delta)
        # Fixing torso orientation as well as sole pose and COM overconstrains
        # the bounded leg chain. Solve orientation too, with a posture prior.
        count=len(columns)
        orientation=Rotation.from_quat(reference[[4,5,6,3]])
        slack=args.root_orientation_slack
        lo=np.r_[lo,np.full(3,-slack)];hi=np.r_[hi,np.full(3,slack)]
        def unpack(x):
            row=reference.copy();row[columns]=x[:count]
            row[3:7]=(orientation*Rotation.from_rotvec(x[count:])).as_quat()[[3,0,1,2]]
            return row
        def residual(x):
            row=unpack(x)
            com,foot,_=fk(row)
            values=[(com-target)*2000]
            for side in active:
                position,rotation=foot[side];want_position,want_rotation=anchors[side]
                values += [(position-want_position)*2000,(want_rotation.inv()*rotation).as_rotvec()*300]
            values += [(x[:count]-reference[columns])*.03,x[count:]*.1]
            return np.concatenate(values)
        initial=np.r_[reference[columns],np.zeros(3)]
        if previous is not None:
            initial[:count]=previous[columns]
            initial[count:]=(orientation.inv()*Rotation.from_quat(previous[[4,5,6,3]])).as_rotvec()
        initial=np.clip(initial,lo,hi)
        fit=least_squares(residual,initial,bounds=(lo,hi),max_nfev=60,
                          ftol=1e-9,xtol=1e-9,gtol=1e-8)
        row=unpack(fit.x)
        physical=residual(fit.x)[:3+6*len(active)]
        errors.append({"com_error_m":float(np.linalg.norm(physical[:3])/2000),
            "maximum_scaled_constraint":float(np.abs(physical).max()),"nfev":int(fit.nfev),
            "root_orientation_adjustment_rad":float(np.linalg.norm(fit.x[count:])),
            "foot_position_error_m":max(float(np.linalg.norm(physical[3+6*k:6+6*k])/2000) for k in range(len(active)))})
        return row
    def static_support(row):
        d.qpos[:]=row;d.qvel[:]=0;d.qacc[:]=0;mujoco.mj_inverse(m,d)
        required=d.qfrc_inverse.copy();columns=[]
        rays=np.array([[a/np.sqrt(2),b/np.sqrt(2),1] for a,b in [(1,1),(1,-1),(-1,1),(-1,-1)]]).T
        jp,jr0=np.zeros((3,m.nv)),np.zeros((3,m.nv))
        for g,v in meshes:
            if m.geom_bodyid[g]!=feet["left"]:continue
            world=v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
            near=world[world[:,2]<world[:,2].min()+.003]
            if len(near)>12:near=near[np.linspace(0,len(near)-1,12).astype(int)]
            for point in near:
                mujoco.mj_jac(m,d,jp,jr0,point,int(m.geom_bodyid[g]));columns.append(jp.T@rays)
        return solve_contact_effort(np.concatenate(columns,axis=1),required,effort)
    hip_values=[-.6,-.8,-.8,.8] if args.swing=="backward" else [.2,.25,.25,-2.]
    knee_values=[1.5,2.1,2.1,.65] if args.swing=="backward" else [1.2,1.5,1.5,.35]
    swing_speed=6. if args.swing=="backward" else -6.
    start=ready.copy()
    start[j("right_hip_pitch_joint")]=hip_values[0];start[j("right_knee_joint")]=knee_values[0]
    start[3:7]=Rotation.from_euler("xyz",[-.12,.05,0]).as_quat()[[3,0,1,2]]
    first_com=np.r_[anchors["left"][0][:2]+[.03,0],args.initial_com_z]
    start=fit_pose(start,first_com,["left"])
    # Neutral, feet-flat crouched landing, independently of the video pose.
    landing=ready.copy();landing[3:7]=Rotation.from_euler("y",6.).as_quat()[[3,0,1,2]]
    for side in feet:
        landing[j(f"{side}_hip_pitch_joint")]=-.4
        landing[j(f"{side}_knee_joint")]=1.2
        landing[j(f"{side}_shoulder_pitch_joint")]=-.7
        landing[j(f"{side}_elbow_joint")]=-1.4
    ankle_columns=np.array([j(f"{s}_{n}_joint") for s in feet for n in ["hip_yaw","ankle_pitch","ankle_roll"]])
    def landing_orientation(x):
        row=landing.copy();row[ankle_columns]=x
        _,poses,_=fk(row)
        return np.concatenate([(anchors[s][1].inv()*poses[s][1]).as_rotvec() for s in feet])
    fit=least_squares(landing_orientation,landing[ankle_columns],
        bounds=(lower[ankle_columns-7],upper[ankle_columns-7]),max_nfev=40)
    landing[ankle_columns]=fit.x
    landing[2]+=.005-fk(landing)[2]
    landing_z=fk(landing)[0][2]
    fps=50.;takeoff=round(args.preparation*fps);land=takeoff+round(args.flight*fps)
    duration=takeoff/fps;air_duration=(land-takeoff)/fps
    vertical_velocity=(landing_z-args.launch_com_z)/air_duration+.5*9.81*air_duration
    advance=args.com_advance;horizontal_velocity=3*advance/duration
    end_com=np.r_[first_com[:2]+[advance,0],args.launch_com_z]
    land_com=end_com+np.array([horizontal_velocity*air_duration,0,landing_z-args.launch_com_z])
    landing[:3]+=land_com-fk(landing)[0]
    recover=40
    q=np.tile(ready,(land+recover+1,1))
    q[land]=landing
    omega=np.array([0.,5.,0.])
    shoot=[]
    for refinement in range(args.projection_iterations):
        times=np.array([0,.35*duration/.9,.55*duration/.9,duration])
        end_vector=np.array([0.,1.,0.]);angle=np.linalg.norm(end_vector)
        cross=np.array([[0,0,1],[0,0,0],[-1,0,0]],dtype=float)
        jr=np.eye(3)-(1-np.cos(angle))/angle**2*cross+(angle-np.sin(angle))/angle**3*cross@cross
        vectors=[Rotation.from_euler("xyz",[-.12,.05,0]).as_rotvec(),
                 Rotation.from_euler("xyz",[-.20,.15,0]).as_rotvec(),
                 Rotation.from_euler("xyz",[-.20,.15,0]).as_rotvec(),end_vector]
        rotation_curve=BPoly.from_derivatives(times,[[v,np.zeros(3),np.zeros(3)] if k<3 else
            [v,np.linalg.solve(jr,omega),np.zeros(3)] for k,v in enumerate(vectors)])
        z_curve=CubicHermiteSpline(times,[first_com[2],args.crouch_com_z,args.crouch_com_z,args.launch_com_z],[0,0,0,vertical_velocity])
        hip_curve=CubicHermiteSpline(times,hip_values,[0,0,0,swing_speed])
        knee_curve=CubicHermiteSpline(times,knee_values,[0,0,0,-2])
        arm_curve=CubicHermiteSpline(times,[.2,.7,.7,-.7],[0,0,0,-4])
        for frame in range(takeoff+1):
            t=frame/fps;u=t/duration
            row=start.copy()
            row[3:7]=Rotation.from_rotvec(rotation_curve(t)).as_quat()[[3,0,1,2]]
            row[j("right_hip_pitch_joint")]=hip_curve(t)
            row[j("right_knee_joint")]=knee_curve(t)
            for side in feet:row[j(f"{side}_shoulder_pitch_joint")]=arm_curve(t)
            desired=np.r_[first_com[:2]+[advance*u**3,0],float(z_curve(t))]
            q[frame]=fit_pose(row,desired,["left"],q[frame-1] if frame else None)
        if args.stance_only:
            stance=q[:takeoff+1]
            heights={side:[] for side in feet}
            for row in stance:
                fk(row)
                for side,body in feet.items():
                    heights[side].append(min(float((v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])[:,2].min())
                        for g,v in meshes if m.geom_bodyid[g]==body))
            static=static_support(stance[0])
            args.out.mkdir(parents=True,exist_ok=False)
            np.savez(args.out/"stance-qpos.npz",qpos=stance,fps=np.array([fps]))
            report={"approved_for_training":False,"stance_only":True,
                "arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                "maximum_com_error_m":max(e["com_error_m"] for e in errors),
                "maximum_foot_position_error_m":max(e["foot_position_error_m"] for e in errors),
                "maximum_scaled_constraint":max(e["maximum_scaled_constraint"] for e in errors),
                "maximum_hard_violation_rad":float(np.maximum(np.maximum(hard_lower-stance[:,7:],stance[:,7:]-hard_upper),0).max()),
                "maximum_soft_violation_rad":float(np.maximum(np.maximum(soft_lower-stance[:,7:],stance[:,7:]-soft_upper),0).max()),
                "minimum_foot_height_m":{side:min(values) for side,values in heights.items()},
                "initial_static_contact_feasible":bool(static.success),
                "initial_static_peak_motor_ratio":float(static.x[-1]) if static.success else None,
                "maximum_adjacent_joint_change_rad":float(np.abs(np.diff(stance[:,7:],axis=0)).max())}
            (args.out/"ik.json").write_text(json.dumps(errors))
            (args.out/"stance-report.json").write_text(json.dumps(report,indent=2))
            print(json.dumps(report),flush=True)
            return
        v_launch=(3*q[takeoff,7:]-4*q[takeoff-1,7:]+q[takeoff-2,7:])*fps/2
        tuck=q[takeoff,7:].copy()
        for side in feet:
            tuck[j(f"{side}_hip_pitch_joint")-7]=args.tuck_hip
            tuck[j(f"{side}_knee_joint")-7]=args.tuck_knee
            tuck[j(f"{side}_ankle_pitch_joint")-7]=-.4
            tuck[j(f"{side}_elbow_joint")-7]=-1.4
        tuck[j("waist_pitch_joint")-7]=args.tuck_waist
        tuck=np.clip(tuck,lower,upper)
        joint_curve=CubicHermiteSpline([0,.4*air_duration,.6*air_duration,air_duration],
            np.array([q[takeoff,7:],tuck,tuck,landing[7:]]),
            np.array([v_launch,np.zeros(len(names)),np.zeros(len(names)),np.zeros(len(names))]))
        for frame in range(takeoff+1,land+1):
            t=(frame-takeoff)/fps
            q[frame,7:]=np.clip(joint_curve(t),lower,upper)
            q[frame,3:7]=Rotation.from_euler("y",1+5*t/air_duration).as_quat()[[3,0,1,2]]
            target=end_com+np.array([horizontal_velocity*t,0,vertical_velocity*t-.5*9.81*t*t])
            q[frame,:3]+=target-fk(q[frame])[0]
        q[land]=landing
        # Supply a benign fixed tail to the spline used by centroidal shooting.
        q[land+1:]=landing
        q,momentum,omega=project_flight(m,q,takeoff,land,fps,np.array([0.,20.,0.]))
        shoot.append({"iteration":refinement,"momentum":momentum.tolist(),"launch_omega_body":omega.tolist()})
    # Recovery keeps both feet planted and absorbs downward landing velocity.
    _,land_feet,_=fk(q[land]);anchors=land_feet
    centre_xy=np.mean([v[0][:2] for v in anchors.values()],axis=0)
    falling_velocity=vertical_velocity-9.81*air_duration
    recovery_z=CubicHermiteSpline([0,.12,.8],[landing_z,landing_z-.11,com_ready[2]],
                                  [falling_velocity,0,0])
    recovery_xy=CubicHermiteSpline([0,.8],[land_com[:2],centre_xy],[[horizontal_velocity,0],[0,0]])
    recovery_angle=CubicHermiteSpline([0,.8],[6.,2*np.pi],[float(omega[1]),0])
    for k in range(1,recover+1):
        t=k/fps;u=t/.8;w=u*u*(3-2*u)
        row=landing.copy();row[7:]=landing[7:]*(1-w)+ready[7:]*w
        row[3:7]=Rotation.from_euler("y",recovery_angle(t)).as_quat()[[3,0,1,2]]
        desired=np.r_[recovery_xy(t),float(recovery_z(t))]
        q[land+k]=fit_pose(row,desired,["left","right"],q[land+k-1])
    # Static check of the actual one-leg initial support, including motor limits.
    static=static_support(q[0])
    args.out.mkdir(parents=True,exist_ok=False)
    np.savetxt(args.out/"webster.csv",np.c_[q[:,:3],q[:,[4,5,6,3]],q[:,7:]],delimiter=",",fmt="%.10f")
    convert_csv_to_npz(args.out/"webster.csv",args.out/"webster.npz",fps,fps)
    report={"purpose":"Robot-designed kinematic guide, not a trained policy or dynamic feasibility certificate",
        "arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        "approved_for_training":False,"ready_sha256":hashlib.sha256(args.ready_config.read_bytes()).hexdigest(),
        "frames":len(q),"takeoff_frame":takeoff,"landing_frame":land,"preparation_seconds":duration,
        "flight_seconds":air_duration,"launch_com_velocity":[horizontal_velocity,0,vertical_velocity],
        "centroidal_fits":shoot,"initial_static_contact_feasible":bool(static.success),
        "sole_anchor_flattening_rad":flattening,
        "initial_static_peak_motor_ratio":float(static.x[-1]) if static.success else None,
        "maximum_ik_com_error_m":max(e["com_error_m"] for e in errors),
        "maximum_ik_scaled_constraint":max(e["maximum_scaled_constraint"] for e in errors),
        "maximum_adjacent_joint_change_rad":float(np.abs(np.diff(q[:,7:],axis=0)).max()),
        "limitations":"Requires independent geometry/dynamics audits and repeated motor-bounded execution in project physics before short training."}
    (args.out/"GUIDE_ONLY.json").write_text(json.dumps(report,indent=2))
    (args.out/"ik.json").write_text(json.dumps(errors))
    (args.out/"contacts.json").write_text(json.dumps({"reference_fps":fps,"takeoff_reference_frame":takeoff,
        "landing_reference_frame":land,"gravity_m_per_s2":9.81,"annotation":"Designed fixed-left-foot support and ballistics; not source-video contact measurements"},indent=2))
    print(json.dumps(report),flush=True)


if __name__=="__main__":main()
