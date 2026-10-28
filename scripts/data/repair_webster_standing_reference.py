"""Repair wrist jitter and fit a planted-foot landing-to-standing reference.

This constructs a kinematic candidate, not a physical success certificate.
Run on the project server through scripts/robot.py x2_ultra.
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
from scipy.interpolate import BPoly
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix


def smoothstep(u):
    u = np.clip(u, 0., 1.)
    return u**3 * (10 + u * (-15 + 6 * u))


def bounded_settling_curve(start,end,rate,duration,low,high,min_duration):
    """Preserve entry velocity using bounded quintic Bezier control points."""
    if min(start,end)<low-1e-8 or max(start,end)>high+1e-8:
        raise ValueError('Posture endpoint outside soft limits')
    seconds=duration
    if abs(rate)>1e-9:
        gap=(high-start) if rate>0 else (start-low)
        seconds=min(seconds,2.5*max(gap-1e-6,0)/abs(rate))
    if seconds<min_duration:
        raise ValueError('Incoming rate cannot settle inside limits at this sample rate; start the transition earlier')
    curve=BPoly.from_derivatives([0,seconds],[[start,rate,0.],[end,0.,0.]])
    return curve,seconds


def smooth_wrists(poses, names, strength=30.):
    """Penalize second differences without moving the first/last wrist pose."""
    result = poses.copy()
    n = len(poses)
    second = np.diff(np.eye(n), n=2, axis=0)
    matrix = np.eye(n) + strength * second.T @ second
    for joint, name in enumerate(names):
        if 'wrist' not in name:
            continue
        original = poses[:, 7 + joint]
        fixed = np.array([0, n - 1])
        free = np.arange(1, n - 1)
        rhs = original[free] - matrix[np.ix_(free, fixed)] @ original[fixed]
        result[free, 7 + joint] = np.linalg.solve(matrix[np.ix_(free, free)], rhs)
    return result


class Geometry:
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.root = int(model.jnt_bodyid[0])
        self.feet = {s: model.body(s + '_ankle_roll_link').id for s in ['left', 'right']}
        self.meshes = []
        for g in range(model.ngeom):
            if not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue
            k = int(model.geom_dataid[g])
            if int(model.geom_type[g])==int(mujoco.mjtGeom.mjGEOM_BOX):
                vertices=np.array([[x,y,z] for x in [-1.,1.] for y in [-1.,1.] for z in [-1.,1.]])*model.geom_size[g]
            elif k>=0:
                a,n=model.mesh_vertadr[k],model.mesh_vertnum[k];vertices=model.mesh_vert[a:a+n].copy()
            else:raise ValueError('Unsupported robot collision geometry')
            group = next((s for s, b in self.feet.items() if model.geom_bodyid[g] == b), 'nonfoot')
            self.meshes.append((g, vertices, group))

    def update(self, pose):
        self.data.qpos[:] = pose
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def vertices(self, side):
        d = self.data
        return np.concatenate([v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                               for g, v, s in self.meshes if s == side])

    def height(self):
        return [float(self.vertices(s)[:, 2].min()) for s in ['left', 'right', 'nonfoot']]

    def foot(self, side):
        b = self.feet[side]
        return self.data.xpos[b].copy(), Rotation.from_matrix(self.data.xmat[b].reshape(3, 3))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--standing', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--start-frame', type=int, default=83)
    p.add_argument('--seconds', type=float, default=1.2)
    p.add_argument('--wrist-strength', type=float, default=30.)
    p.add_argument('--max-nfev', type=int, default=100)
    p.add_argument('--resume',type=Path,help='Warm start from a saved optimization qpos sequence')
    p.add_argument('--match-entry-velocity',action='store_true',help='Match incoming joint and foot twists and include the incoming samples in smoothing')
    args = p.parse_args()
    if args.wrist_strength<0 or args.max_nfev<1 or args.seconds<=.2:
        p.error('Nonnegative smoothing strength, positive evaluations and seconds > .2 required')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'as-run.py').write_bytes(Path(__file__).read_bytes())
    (args.out/'launch.json').write_text(json.dumps({'pid':os.getpid(),'hostname':socket.gethostname(),
        'command':[sys.executable,*sys.argv],'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'python':sys.version,'mujoco':mujoco.__version__,'numpy':np.__version__,
        'sole_box_config':os.environ.get('SM_SOLE_BOX_CONFIG'),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
    if os.environ.get('SM_SOLE_BOX_CONFIG'):
        (args.out/'sole-boxes-as-run.json').write_bytes(Path(os.environ['SM_SOLE_BOX_CONFIG']).read_bytes())
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.data.physics_motion import differentiate_qpos, export_qpos_motion
    from supermimic.robots.dimensions import PROFILE
    m = load_g1_model()
    geo = Geometry(m)
    z = np.load(args.source)
    names = z['joint_names'].tolist()
    if names != list(PROFILE['joint_names']):
        raise ValueError('Unexpected joint order')
    fps = float(z['fps'].reshape(-1)[0])
    root = z['body_names'].tolist().index(m.body(geo.root).name)
    original = np.c_[z['body_pos_w'][:, root], z['body_quat_w'][:, root], z['joint_pos']].astype(float)
    source = smooth_wrists(original, names, args.wrist_strength)
    # Preserve the original COM in every frame, including the ballistic flight.
    com_error = []; com_targets = []
    for i in range(len(source)):
        geo.update(original[i]); target = geo.data.subtree_com[geo.root].copy()
        com_targets.append(target)
        geo.update(source[i]); source[i, :3] += target - geo.data.subtree_com[geo.root]
        geo.update(source[i]); com_error.append(float(np.linalg.norm(target - geo.data.subtree_com[geo.root])))
    # Smoothing wrist orientation can lower the hand. Correct the nearby arm
    # posture with a smooth, small offset rather than changing the flight COM.
    arm_columns = np.array([7+j for j,n in enumerate(names) if 'shoulder' in n or 'elbow' in n])
    smoothed = source.copy()
    arm_changes = np.zeros((len(source),len(arm_columns)))
    for _ in range(3):
        proposal = arm_changes.copy()
        for i in range(len(source)):
            def corrected(values):
                q = smoothed[i].copy(); q[arm_columns] += values
                geo.update(q); q[:3] += com_targets[i] - geo.data.subtree_com[geo.root]
                geo.update(q)
                return q
            corrected(arm_changes[i])
            if geo.height()[2] >= .009:
                continue
            def clearance(values):
                corrected(values)
                return np.r_[values*.01, max(.012-geo.height()[2],0)*100]
            fit = least_squares(clearance, arm_changes[i], bounds=(-.1,.1),
                                max_nfev=60, diff_step=1e-4)
            proposal[i] = fit.x
        arm_changes = gaussian_filter1d(proposal,1.,axis=0,mode='nearest')
    for i in range(len(source)):
        source[i] = smoothed[i]
        source[i,arm_columns] += arm_changes[i]
        geo.update(source[i]); source[i,:3] += com_targets[i] - geo.data.subtree_com[geo.root]
        geo.update(source[i]); com_error.append(float(np.linalg.norm(com_targets[i]-geo.data.subtree_com[geo.root])))
    frame = args.start_frame
    if not 2 <= frame < len(source) or args.seconds <= .2:
        raise ValueError('Invalid transition boundary or duration')
    initial = source[frame].copy()
    end = np.load(args.standing)['qpos'].copy()
    if end.shape != initial.shape:
        raise ValueError('Standing pose dimension mismatch')
    geo.update(initial)
    initial_com = geo.data.subtree_com[geo.root].copy()
    initial_feet = {s: geo.foot(s) for s in geo.feet}
    vertices = geo.vertices('left')
    # Preserve an actual lowest contact point while leveling the support foot.
    pivot = vertices[np.argmin(vertices[:, 2])].copy()
    lp, lr = initial_feet['left']
    local_pivot = lr.inv().apply(pivot - lp)
    geo.update(end)
    final_feet = {s: geo.foot(s) for s in geo.feet}
    velocity = differentiate_qpos(m, source, fps)[frame]
    geo.update(initial)
    jac = np.zeros((3, m.nv))
    mujoco.mj_jacSubtreeCom(m, geo.data, jac, geo.root)
    com_velocity = jac @ velocity
    # Choose a forward capture location from COM momentum, not root return.
    capture_time = np.sqrt(initial_com[2] / 9.81)
    geo.update(end)
    final_com_unshifted = geo.data.subtree_com[geo.root].copy()
    end[:2] += initial_com[:2] + capture_time*com_velocity[:2] - final_com_unshifted[:2]
    # Do not place the right landing foot across/too close to the left support
    # foot. A raw lateral capture target can create an unreachable narrow stance.
    geo.update(end)
    heading = Rotation.from_quat(end[[4,5,6,3]]).as_euler('xyz')[2]
    lateral = Rotation.from_euler('z',heading).apply([0.,1.,0.])
    right_lateral = float(geo.foot('right')[0] @ lateral)
    allowed_lateral = float(initial_feet['left'][0] @ lateral) - .20
    end[:2] -= max(right_lateral-allowed_lateral,0)*lateral[:2]
    geo.update(end)
    final_feet = {s: geo.foot(s) for s in geo.feet}
    final_com = geo.data.subtree_com[geo.root].copy()
    right_floor_height=float(geo.vertices('right')[:,2].min())
    final_pivot = final_feet['left'][0] + final_feet['left'][1].apply(local_pivot)
    count = round(args.seconds * fps)
    duration = count / fps
    stop_time = min(duration*.6, 2*capture_time)
    support_com = final_com.copy()
    support_com += Rotation.from_euler('z',heading).apply([0.,-.09,0.])
    curve = BPoly.from_derivatives([0, stop_time, duration*.7, duration],
        [[initial_com, com_velocity, np.zeros(3)], [support_com, np.zeros(3), np.zeros(3)],
         [support_com, np.zeros(3), np.zeros(3)], [final_com, np.zeros(3), np.zeros(3)]])
    rot0 = Rotation.from_quat(initial[[4, 5, 6, 3]])
    rot1 = Rotation.from_quat(end[[4, 5, 6, 3]])
    columns = np.array([m.jnt_qposadr[m.joint(side+'_' + s + '_joint').id]
        for side in ['left','right']
        for s in ['hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll']])
    joint_ids = np.array([int(m.dof_jntid[c-1]) for c in columns])
    from supermimic.robots.assets import get_robot_cfg
    factor = get_robot_cfg().articulation.soft_joint_pos_limit_factor
    centre = m.jnt_range[joint_ids].mean(axis=1)
    half = np.diff(m.jnt_range[joint_ids],axis=1)[:,0]*factor/2
    lower, upper = centre-half+.01, centre+half-.01
    fits = []; targets = []
    geo.update(initial)
    left_local = initial_feet['left'][1].inv().apply(geo.vertices('left')-initial_feet['left'][0])
    right_local = initial_feet['right'][1].inv().apply(geo.vertices('right')-initial_feet['right'][0])
    jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv))
    mujoco.mj_jac(m,geo.data,jp,jr,pivot,geo.feet['left']);pivot_velocity=jp@velocity
    left_omega=lr.inv().apply(jr@velocity)
    mujoco.mj_jacBody(m,geo.data,jp,jr,geo.feet['right'])
    right_velocity=jp@velocity;right_omega=initial_feet['right'][1].inv().apply(jr@velocity)
    def boundary_curve(first,last,seconds,rate):
        return BPoly.from_derivatives([0,seconds],[[first,rate,np.zeros_like(first)],[last,np.zeros_like(first),np.zeros_like(first)]])
    if args.match_entry_velocity:
        all_ranges=m.jnt_range[1:];all_centre=all_ranges.mean(axis=1)
        all_half=np.diff(all_ranges,axis=1)[:,0]*factor/2
        joint_curves=[];joint_durations=[]
        for j,name in enumerate(names):
            if 7+j in columns:
                joint_curves.append(boundary_curve(np.array(initial[7+j]),np.array(end[7+j]),duration,np.array(velocity[6+j])))
                joint_durations.append(duration)
            else:
                curve_j,seconds_j=bounded_settling_curve(initial[7+j],end[7+j],velocity[6+j],duration,
                    all_centre[j]-all_half[j],all_centre[j]+all_half[j],2/fps)
                joint_curves.append(curve_j);joint_durations.append(seconds_j)
        root_angle_curve=boundary_curve(np.zeros(3),(rot0.inv()*rot1).as_rotvec(),duration,velocity[3:6])
        flat_time=min(.25,duration*.4)
        left_angle_curve=boundary_curve(np.zeros(3),(lr.inv()*final_feet['left'][1]).as_rotvec(),flat_time,left_omega)
        rp,rr=initial_feet['right'];ep,er=final_feet['right']
        right_pos_curve=boundary_curve(rp,ep,duration*.45,right_velocity)
        right_angle_curve=boundary_curve(np.zeros(3),(rr.inv()*er).as_rotvec(),duration*.45,right_omega)
        plant_time=min(.12,duration*.2);plant_end=pivot.copy();plant_end[:2]+=.5*plant_time*pivot_velocity[:2]
        plant_end[2]=final_pivot[2]
        pivot_curve=boundary_curve(pivot,plant_end,plant_time,pivot_velocity)
    for i in range(1, count + 1):
        t = i / fps; u = t / duration; a = float(smoothstep(u))
        flat = float(smoothstep(t / min(.25, duration * .4)))
        foot_rot = lr * Rotation.from_rotvec((lr.inv() * final_feet['left'][1]).as_rotvec() * flat)
        step = float(smoothstep((t-duration*.55)/(duration*.3)))
        contact = (1-step)*pivot + step*final_pivot
        # The left foot stays planted while the right lands, then takes a step.
        contact[2] = (1-flat)*pivot[2] + flat*final_pivot[2] + .025*np.sin(np.pi*step)**2
        foot_pos = contact - foot_rot.apply(local_pivot)
        # Right foot lands early, then its target remains fixed.
        swing = float(smoothstep(t / (duration * .45)))
        rp, rr = initial_feet['right']; ep, er = final_feet['right']
        right_pos = (1-swing)*rp + swing*ep
        right_rot = rr * Rotation.from_rotvec((rr.inv()*er).as_rotvec()*swing)
        right_pos[2] = max(right_pos[2], right_floor_height-right_rot.apply(right_local)[:,2].min())
        root_rot = rot0 * Rotation.from_rotvec((rot0.inv()*rot1).as_rotvec()*a)
        posture = initial.copy(); posture[7:] = (1-a)*initial[7:] + a*end[7:]
        if args.match_entry_velocity:
            foot_rot=lr*Rotation.from_rotvec(left_angle_curve(min(t,flat_time)))
            planted=pivot_curve(min(t,plant_time))
            contact=(1-step)*planted+step*final_pivot
            contact[2]+=.025*np.sin(np.pi*step)**2
            foot_pos=contact-foot_rot.apply(local_pivot)
            foot_pos[2]=max(foot_pos[2],right_floor_height-foot_rot.apply(left_local)[:,2].min())
            right_pos=right_pos_curve(min(t,duration*.45))
            right_rot=rr*Rotation.from_rotvec(right_angle_curve(min(t,duration*.45)))
            right_pos[2]=max(right_pos[2],right_floor_height-right_rot.apply(right_local)[:,2].min())
            root_rot=rot0*Rotation.from_rotvec(root_angle_curve(t))
            posture[7:]=[float(c(min(t,seconds))) for c,seconds in zip(joint_curves,joint_durations)]

        targets.append((foot_pos,foot_rot,right_pos,right_rot,root_rot,posture))
    # Optimize the complete transition together. Independent frame IK admits
    # branch changes and deep crouches even with nearly exact foot placement.
    root_curve=BPoly.from_derivatives([0,stop_time,duration],
        [[initial[:3],velocity[:3],np.zeros(3)],[end[:3],np.zeros(3),np.zeros(3)],
         [end[:3],np.zeros(3),np.zeros(3)]])
    n=count-1; width=18
    nominal=np.array([np.r_[root_curve((i+1)/fps),np.zeros(3),targets[i][5][columns]] for i in range(n)])
    low=np.tile(np.r_[[-.15,-.15,-.12],np.full(3,-.2),lower],(n,1))
    high=np.tile(np.r_[[.15,.15,.12],np.full(3,.2),upper],(n,1))
    low[:,:3]+=nominal[:,:3];high[:,:3]+=nominal[:,:3]
    low[:,2]=np.maximum(low[:,2],.45)
    def unpack(values):
        values=values.reshape(n,width);result=[initial]
        for i,x in enumerate(values):
            q=targets[i][5].copy();q[:3]=x[:3]
            q[3:7]=(targets[i][4]*Rotation.from_rotvec(x[3:6])).as_quat()[[3,0,1,2]]
            q[columns]=x[6:];result.append(q)
        result.append(end)
        return np.asarray(result)
    def residual_all(values):
        qs=unpack(values);r=[]
        for i,q in enumerate(qs[1:-1]):
            lp,lr_target,rp,rr_target,root_target,posture=targets[i]
            geo.update(q);left=geo.foot('left');right=geo.foot('right')
            r.extend(np.r_[(left[0]-lp)*10000,(lr_target.inv()*left[1]).as_rotvec()*2000,
                (right[0]-rp)*10000,(rr_target.inv()*right[1]).as_rotvec()*2000,
                (q[:3]-nominal[i,:3])*500,
                np.asarray(values).reshape(n,width)[i,3:6]*200,
                (q[columns]-posture[columns])*2])
        rotation=Rotation.from_quat(qs[:,[4,5,6,3]])
        angles=(rot0.inv()*rotation).as_rotvec()
        coordinates=np.c_[qs[:,:3],angles,qs[:,columns]]
        if args.match_entry_velocity:
            section=np.concatenate([source[frame-2:frame],qs,np.repeat(end[None],2,axis=0)])
            angles=(rot0.inv()*Rotation.from_quat(section[:,[4,5,6,3]])).as_rotvec()
            coordinates=np.c_[section[:,:3],angles,section[:,columns]]
        r.extend((np.diff(coordinates,n=2,axis=0)*np.r_[np.full(3,1000),np.full(3,100),np.full(12,20)]).ravel())
        return np.asarray(r)
    temporal=n+4 if args.match_entry_velocity else n
    sparsity=lil_matrix((n*30+temporal*18,n*width),dtype=int)
    for i in range(n):
        sparsity[i*30:(i+1)*30,i*width:(i+1)*width]=1
    for i in range(temporal):
        center=i-1 if args.match_entry_velocity else i+1
        for t in range(max(1,center-1),min(count-1,center+1)+1):
            j=t-1;sparsity[n*30+i*18:n*30+(i+1)*18,j*width:(j+1)*width]=1
    progress={'iteration':0}
    def checkpoint(x):
        progress['iteration']+=1
        if progress['iteration']%10==0:
            np.savez(args.out/f'optimization-{progress["iteration"]:03d}.npz',qpos=unpack(x))
            print(json.dumps({'iteration':progress['iteration']}),flush=True)
    seed=nominal.copy()
    if args.resume:
        warm=np.load(args.resume)['qpos']
        if warm.shape!=(count+1,m.nq):raise ValueError('Warm-start trajectory shape mismatch')
        for i,q in enumerate(warm[1:-1]):
            seed[i,:3]=q[:3];seed[i,3:6]=(targets[i][4].inv()*Rotation.from_quat(q[[4,5,6,3]])).as_rotvec()
            seed[i,6:]=q[columns]
    solved=least_squares(residual_all,np.clip(seed,low,high).ravel(),
        bounds=(low.ravel(),high.ravel()),jac_sparsity=sparsity.tocsr(),
        tr_options={'maxiter':100},callback=checkpoint,
        max_nfev=args.max_nfev,ftol=1e-7,xtol=1e-7,gtol=1e-6,verbose=1)
    poses=unpack(solved.x)
    for i,q in enumerate(poses[1:]):
        geo.update(q);target=targets[i];right=geo.foot('right')
        fits.append({'frame':frame+i+1,'com_error_m':float(np.linalg.norm(geo.data.subtree_com[geo.root]-curve((i+1)/fps))),
            'root_error_rad':float((target[4].inv()*Rotation.from_quat(q[[4,5,6,3]])).magnitude()),
            'right_position_error_m':float(np.linalg.norm(right[0]-target[2])),
            'right_rotation_error_rad':float((target[3].inv()*right[1]).magnitude())})
    full = np.concatenate([source[:frame], poses, np.repeat(poses[-1:,:], round(2*fps), axis=0)])
    v = differentiate_qpos(m, full, fps)
    export_qpos_motion(m, full, args.out/'motion-fk.npz', fps, qvel=v, source=args.source)
    with np.load(args.out/'motion-fk.npz') as exported:
        arrays={k:exported[k].copy() for k in exported.files}
    np.savez(args.out/'webster.npz',**arrays,diagnostic_only=np.array(True),approved_for_training=np.array(False))
    np.savez(args.out/'qpos.npz',qpos=full,qvel=v,fps=np.array([fps]),joint_names=np.array(names))
    heights = []; foot_positions = []; com = []
    for q in full:
        geo.update(q); heights.append(geo.height()); com.append(geo.data.subtree_com[geo.root].copy())
        foot_positions.append([geo.foot(s)[0] for s in ['left','right']])
    heights=np.asarray(heights); foot_positions=np.asarray(foot_positions); com=np.asarray(com)
    limits=np.array([PROFILE['velocity_limit'][n] for n in names])
    lo,hi=m.jnt_range[1:].T
    centre=(lo+hi)/2;half=(hi-lo)*factor/2
    wrists=[j for j,n in enumerate(names) if 'wrist' in n]
    def jitter(q):
        return {'max_step_rad':float(abs(np.diff(q[:,7+np.array(wrists)],axis=0)).max()),
                'rms_second_difference_rad':float(np.sqrt(np.mean(np.diff(q[:,7+np.array(wrists)],n=2,axis=0)**2)))}
    report={'configuration':{k:str(x) if isinstance(x,Path) else x for k,x in vars(args).items()},
            'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
            'standing_sha256':hashlib.sha256(args.standing.read_bytes()).hexdigest(),
            'output_sha256':hashlib.sha256((args.out/'webster.npz').read_bytes()).hexdigest(),
            'frames':len(full),'fps':fps,'transition_frame':frame,'transition_seconds':duration,
            'optimizer_success':bool(solved.success),'optimizer_nfev':int(solved.nfev),
            'shortened_posture_curves_s':{name:seconds for name,seconds in zip(names,joint_durations) if seconds<duration-1e-8} if args.match_entry_velocity else {},
            'wrist_before':jitter(original),'wrist_after':jitter(source),
            'maximum_arm_clearance_adjustment_rad':float(abs(arm_changes).max()),
            'prefix_com_preservation_max_error_m':max(com_error),
            'hard_violation_rad':float(np.maximum(np.maximum(lo-full[:,7:],full[:,7:]-hi),0).max()),
            'soft_violation_rad':float(np.maximum(abs(full[:,7:]-centre)-half,0).max()),
            'maximum_joint_speed_ratio':float((abs(v[:,6:])/limits).max()),
            'maximum_adjacent_joint_speed_ratio':float((abs(np.diff(full[:,7:],axis=0))*fps/limits).max()),
            'entry_generalized_velocity_difference_max':float(abs(v[frame]-velocity).max()),
            'minimum_heights_m':heights.min(0).tolist(),
            'transition_minimum_heights_m':heights[frame:].min(0).tolist(),
            'transition_maximum_lower_foot_height_m':float(heights[frame:,:2].min(axis=1).max()),
            'root_net_horizontal_displacement_m':(poses[-1,:2]-poses[0,:2]).tolist(),
            'root_horizontal_path_m':float(np.linalg.norm(np.diff(poses[:,:2],axis=0),axis=1).sum()),
            'endpoint_pose_difference_max':float(abs(poses[-1]-end).max()),
            'maximum_capture_com_guide_deviation_m':max(x['com_error_m'] for x in fits),
            'fit_max_right_position_error_m':max(x['right_position_error_m'] for x in fits),
            'fit_max_right_rotation_error_rad':max(x['right_rotation_error_rad'] for x in fits),
            'ik':fits,'approved_for_training':False,'dynamic_tracking_verified':False,
            'limitations':'Kinematic correction. Contact-force, friction and continuous native dynamics require separate checks.'}
    report['kinematic_screen_passed']=bool(report['hard_violation_rad']<1e-9 and report['soft_violation_rad']<1e-9
        and report['maximum_adjacent_joint_speed_ratio']<=1 and heights[:,:2].min()>-.004
        and heights[:,2].min()>.005 and report['transition_maximum_lower_foot_height_m']<.006
        and report['fit_max_right_position_error_m']<.003
        and report['fit_max_right_rotation_error_rad']<.01)
    (args.out/'result.json').write_text(json.dumps(report,indent=2))
    (args.out/'contacts.json').write_bytes((args.source.parent/'contacts.json').read_bytes())
    (args.out/'manifest_filtered.csv').write_text(f'name,frames,keep,reason\nwebster,{len(full)},1,reference repair candidate; dynamic validation pending\n')
    print(json.dumps({k:x for k,x in report.items() if k!='ik'}),flush=True)


if __name__=='__main__':
    main()
