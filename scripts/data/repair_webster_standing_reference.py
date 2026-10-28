"""Repair wrist jitter and fit a planted-foot landing-to-standing reference.

This constructs a kinematic candidate, not a physical success certificate.
Run on the project server through scripts/robot.py x2_ultra.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import BPoly
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def smoothstep(u):
    u = np.clip(u, 0., 1.)
    return u**3 * (10 + u * (-15 + 6 * u))


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
            if k < 0:
                raise ValueError('Expected robot collision meshes')
            a, n = model.mesh_vertadr[k], model.mesh_vertnum[k]
            group = next((s for s, b in self.feet.items() if model.geom_bodyid[g] == b), 'nonfoot')
            self.meshes.append((g, model.mesh_vert[a:a+n].copy(), group))

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
    p.add_argument('--start-frame', type=int, default=73)
    p.add_argument('--seconds', type=float, default=.8)
    p.add_argument('--wrist-strength', type=float, default=30.)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'as-run.py').write_bytes(Path(__file__).read_bytes())
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.data.physics_motion import differentiate_qpos, export_qpos_motion
    from supermimic.kinematics.support_reference import SupportPoseIK
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
    com_error = []
    for i in range(len(source)):
        geo.update(original[i]); target = geo.data.subtree_com[geo.root].copy()
        geo.update(source[i]); source[i, :3] += target - geo.data.subtree_com[geo.root]
        geo.update(source[i]); com_error.append(float(np.linalg.norm(target - geo.data.subtree_com[geo.root])))
    frame = args.start_frame
    if not 2 <= frame < len(source) - 2 or args.seconds <= .2:
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
    final_pivot = final_feet['left'][0] + final_feet['left'][1].apply(local_pivot)
    shift = pivot[:2] - final_pivot[:2]
    end[:2] += shift
    geo.update(end)
    final_feet = {s: geo.foot(s) for s in geo.feet}
    final_com = geo.data.subtree_com[geo.root].copy()
    target_pivot_z = float((final_feet['left'][0] + final_feet['left'][1].apply(local_pivot))[2])
    count = round(args.seconds * fps)
    duration = count / fps
    velocity = differentiate_qpos(m, source, fps)[frame]
    geo.update(initial)
    jac = np.zeros((3, m.nv))
    mujoco.mj_jacSubtreeCom(m, geo.data, jac, geo.root)
    com_velocity = jac @ velocity
    curve = BPoly.from_derivatives([0, duration],
        [[initial_com, com_velocity, np.zeros(3)], [final_com, np.zeros(3), np.zeros(3)]])
    rot0 = Rotation.from_quat(initial[[4, 5, 6, 3]])
    rot1 = Rotation.from_quat(end[[4, 5, 6, 3]])
    ik = SupportPoseIK(m, initial, geo.feet['left'])
    right_columns = np.array([m.jnt_qposadr[m.joint('right_' + s + '_joint').id]
                             for s in ['hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll']])
    columns = np.r_[ik.columns, right_columns]
    joint_ids = np.array([int(m.dof_jntid[c-1]) for c in columns])
    lower, upper = m.jnt_range[joint_ids, 0] + .01, m.jnt_range[joint_ids, 1] - .01
    previous = initial[columns].copy()
    poses = [initial.copy()]
    fits = []
    for i in range(1, count + 1):
        t = i / fps; u = t / duration; a = float(smoothstep(u))
        flat = float(smoothstep(t / min(.25, duration * .4)))
        foot_rot = lr * Rotation.from_rotvec((lr.inv() * final_feet['left'][1]).as_rotvec() * flat)
        contact = pivot.copy(); contact[2] += flat * (target_pivot_z - pivot[2])
        foot_pos = contact - foot_rot.apply(local_pivot)
        # Right foot lands early, then remains exactly at the terminal target.
        swing = float(smoothstep(t / (duration * .65)))
        rp, rr = initial_feet['right']; ep, er = final_feet['right']
        right_pos = (1-swing)*rp + swing*ep
        right_rot = rr * Rotation.from_rotvec((rr.inv()*er).as_rotvec()*swing)
        root_rot = rot0 * Rotation.from_rotvec((rot0.inv()*rot1).as_rotvec()*a)
        posture = initial.copy(); posture[7:] = (1-a)*initial[7:] + a*end[7:]

        def pose(values):
            q = posture.copy(); q[right_columns] = values[6:]
            return ik.place(q, values[:6], foot_pos, foot_rot)[0]

        def residual(values):
            q = pose(values); geo.update(q); d = geo.data
            current = Rotation.from_quat(q[[4,5,6,3]])
            right_current = Rotation.from_matrix(d.xmat[geo.feet['right']].reshape(3,3))
            return np.r_[(d.subtree_com[geo.root]-curve(t))*1000,
                         (root_rot.inv()*current).as_rotvec()*200,
                         (d.xpos[geo.feet['right']]-right_pos)*1000,
                         (right_rot.inv()*right_current).as_rotvec()*200]

        fit = least_squares(residual, np.clip(previous, lower, upper), bounds=(lower, upper),
                            max_nfev=120, ftol=1e-10, xtol=1e-10, gtol=1e-8)
        previous = fit.x.copy(); q = pose(fit.x); error = residual(fit.x)
        poses.append(q)
        fits.append({'frame': frame+i, 'com_error_m':float(np.linalg.norm(error[:3])/1000),
                     'root_error_rad':float(np.linalg.norm(error[3:6])/200),
                     'right_position_error_m':float(np.linalg.norm(error[6:9])/1000),
                     'right_rotation_error_rad':float(np.linalg.norm(error[9:])/200)})
    poses = np.asarray(poses)
    full = np.concatenate([source[:frame], poses, np.repeat(poses[-1:,:], round(2*fps), axis=0)])
    v = differentiate_qpos(m, full, fps)
    export_qpos_motion(m, full, args.out/'webster.npz', fps, qvel=v, source=args.source)
    np.savez(args.out/'qpos.npz',qpos=full,qvel=v,joint_names=np.array(names))
    heights = []; foot_positions = []; com = []
    for q in full:
        geo.update(q); heights.append(geo.height()); com.append(geo.data.subtree_com[geo.root].copy())
        foot_positions.append([geo.foot(s)[0] for s in ['left','right']])
    heights=np.asarray(heights); foot_positions=np.asarray(foot_positions); com=np.asarray(com)
    limits=np.array([PROFILE['velocity_limit'][n] for n in names])
    lo,hi=m.jnt_range[1:].T
    wrists=[j for j,n in enumerate(names) if 'wrist' in n]
    def jitter(q):
        return {'max_step_rad':float(abs(np.diff(q[:,7+np.array(wrists)],axis=0)).max()),
                'rms_second_difference_rad':float(np.sqrt(np.mean(np.diff(q[:,7+np.array(wrists)],n=2,axis=0)**2)))}
    report={'configuration':{k:str(x) if isinstance(x,Path) else x for k,x in vars(args).items()},
            'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
            'standing_sha256':hashlib.sha256(args.standing.read_bytes()).hexdigest(),
            'output_sha256':hashlib.sha256((args.out/'webster.npz').read_bytes()).hexdigest(),
            'frames':len(full),'fps':fps,'transition_frame':frame,'transition_seconds':duration,
            'wrist_before':jitter(original),'wrist_after':jitter(source),
            'prefix_com_preservation_max_error_m':max(com_error),
            'hard_violation_rad':float(np.maximum(np.maximum(lo-full[:,7:],full[:,7:]-hi),0).max()),
            'maximum_joint_speed_ratio':float((abs(v[:,6:])/limits).max()),
            'maximum_adjacent_joint_speed_ratio':float((abs(np.diff(full[:,7:],axis=0))*fps/limits).max()),
            'minimum_heights_m':heights.min(0).tolist(),
            'transition_minimum_heights_m':heights[frame:].min(0).tolist(),
            'transition_maximum_lower_foot_height_m':float(heights[frame:,:2].min(axis=1).max()),
            'root_net_horizontal_displacement_m':(poses[-1,:2]-poses[0,:2]).tolist(),
            'root_horizontal_path_m':float(np.linalg.norm(np.diff(poses[:,:2],axis=0),axis=1).sum()),
            'endpoint_pose_difference_max':float(abs(poses[-1]-end).max()),
            'fit_max_com_error_m':max(x['com_error_m'] for x in fits),
            'fit_max_right_position_error_m':max(x['right_position_error_m'] for x in fits),
            'ik':fits,'approved_for_training':False,'dynamic_tracking_verified':False,
            'limitations':'Kinematic correction. Contact-force, friction and continuous native dynamics require separate checks.'}
    (args.out/'result.json').write_text(json.dumps(report,indent=2))
    (args.out/'contacts.json').write_bytes((args.source.parent/'contacts.json').read_bytes())
    (args.out/'manifest_filtered.csv').write_text(f'name,frames,keep,reason\nwebster,{len(full)},1,reference repair candidate; dynamic validation pending\n')
    print(json.dumps({k:x for k,x in report.items() if k!='ik'}),flush=True)


if __name__=='__main__':
    main()
