"""Diagnostic alignment and native collision audit of a retargeted box45 CSV.

Fits horizontal box placement to the final foot footprint, never changes joint
angles or rescales height, and never approves the reference for training.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--csv", type=Path)
    inputs.add_argument("--aligned", type=Path)
    parser.add_argument("--input-fps", type=float)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    if args.csv and (args.input_fps is None or args.input_fps <= 0):
        parser.error("CSV requires positive input-fps")
    from supermimic.robots.paths import select_robot
    select_robot("x2_ultra", check_assets=True)
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation
    from supermimic.robots.assets import get_spec
    from supermimic.data.motion_convert import load_csv, resample_motion
    from supermimic.envs.box_scene import FixedBox
    from supermimic.robots.dimensions import PROFILE

    if args.aligned:
        aligned = np.load(args.aligned)
        q = aligned['qpos']
        motion = resample_motion(np.c_[q[:,:3],q[:,[4,5,6,3]],q[:,7:]],float(aligned['fps']),50.0)
    else:
        motion = resample_motion(load_csv(args.csv), args.input_fps, 50.0)
    spec = get_spec()
    model = spec.compile()
    data = mujoco.MjData(model)
    feet = [[g for g in range(model.ngeom) if model.geom(g).name.startswith('foot_')
             and side in model.body(int(model.geom_bodyid[g])).name] for side in ('left', 'right')]
    if not all(feet):
        raise ValueError('Missing left/right native foot geometry')
    foot_names = [[model.geom(g).name for g in ids] for ids in feet]
    vertices = {}
    for g in sum(feet, []):
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[g]
            first, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            vertices[g] = model.mesh_vert[first:first+count].copy()
        elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            vertices[g] = np.array([(x,y,z) for x in (-1,1) for y in (-1,1) for z in (-1,1)]) * model.geom_size[g]
        else:
            raise ValueError(f'Unsupported foot geometry {model.geom(g).name}')

    def foot_points(m, d):
        return [np.concatenate([vertices[g] @ d.geom_xmat[g].reshape(3,3).T + d.geom_xpos[g] for g in ids]) for ids in feet]

    k = min(25, motion.num_frames // 4)
    if k < 2:
        raise ValueError('Reference too short')
    travel = motion.root_pos[-k:, :2].mean(0) - motion.root_pos[:k, :2].mean(0)
    if not args.aligned and np.linalg.norm(travel) < .1:
        raise ValueError('Insufficient recovered travel to define approach; needs camera/box calibration')
    yaw = 0.0 if args.aligned else -float(np.arctan2(travel[1], travel[0]))
    align = Rotation.from_euler('z', yaw)
    origin = np.zeros(3) if args.aligned else np.r_[motion.root_pos[:k,:2].mean(0), 0]
    root = align.apply(motion.root_pos - origin)
    quat = (align * Rotation.from_quat(motion.root_quat[:,[1,2,3,0]])).as_quat()[:,[3,0,1,2]]
    qpos = np.c_[root, quat, motion.joint_pos]
    bottoms = []
    for row in qpos[:k]:
        data.qpos[:] = row
        mujoco.mj_kinematics(model, data)
        bottoms.append([p[:,2].min() for p in foot_points(model,data)])
    offset = 0.0 if args.aligned else -float(np.median(np.min(bottoms,axis=1)))
    qpos[:,2] += offset
    terminal_points = []
    for row in qpos[-k:]:
        data.qpos[:] = row
        mujoco.mj_kinematics(model,data)
        terminal_points.append(np.concatenate(foot_points(model,data)))
    points = np.concatenate(terminal_points)
    center = (points[:,:2].min(0) + points[:,:2].max(0))/2
    box = FixedBox(front_x=float(center[0]-.66/2),center_y=float(center[1]))
    if args.aligned:
        center, half = aligned['box_center'], aligned['box_half_size']
        box = FixedBox(front_x=float(center[0]-half[0]),center_y=float(center[1]),
                       depth=float(half[0]*2),width=float(half[1]*2),height=float(half[2]*2))
    box.add_to_spec(spec)
    if not any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in spec.geoms):
        spec.worldbody.add_geom(name='audit_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,
                               size=(0,0,.05),contype=1,conaffinity=1)
    model = spec.compile()
    # World geoms are compiled before robot geoms. Resolve names again after
    # adding the box/floor instead of reusing the old numeric geom indices.
    old_vertices = {name: vertices[g] for names, ids in zip(foot_names, feet) for name,g in zip(names,ids)}
    feet = [[model.geom(name).id for name in names] for names in foot_names]
    vertices = {model.geom(name).id: local for name,local in old_vertices.items()}
    for side, ids in zip(('left','right'), feet):
        assert all(side in model.body(int(model.geom_bodyid[g])).name for g in ids)
    data = mujoco.MjData(model)
    box_id = model.geom('box45_collision').id
    bounds = model.jnt_range[1:]
    excess = np.maximum(bounds[:,0]-motion.joint_pos,0) + np.maximum(motion.joint_pos-bounds[:,1],0)
    rows=[];worst_box=None;minimum_contact=0.0;floor_minimum=0.0
    for i,row in enumerate(qpos):
        data.qpos[:]=row
        mujoco.mj_forward(model,data)
        fp=foot_points(model,data)
        contacts=[float(c.dist) for c in data.contact if box_id in (int(c.geom1),int(c.geom2))]
        for contact in data.contact:
            if any(model.geom_type[int(g)] == mujoco.mjtGeom.mjGEOM_PLANE for g in (contact.geom1,contact.geom2)):
                floor_minimum=min(floor_minimum,float(contact.dist))
            if box_id in (int(contact.geom1),int(contact.geom2)) and contact.dist < minimum_contact:
                minimum_contact=float(contact.dist)
                worst_box={'frame':i,'distance_m':minimum_contact,'geoms':[model.geom(int(g)).name for g in (contact.geom1,contact.geom2)]}
        rows.append([i/50, *[float(p[:,2].min()) for p in fp],
                     *[float(all(box.top_contains(float(x),float(y)) for x,y in p[:,:2])) for p in fp],
                     min(contacts,default=0),float(excess[i].max())])
    rows=np.array(rows)
    if not args.aligned:
        np.testing.assert_allclose(np.median(rows[:k,1:3].min(1)),0.0,atol=1e-6)
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    np.savez_compressed(args.out/'aligned-diagnostic.npz',qpos=qpos,fps=50,box_center=box.center,box_half_size=box.half_size)
    np.savetxt(args.out/'per-frame.csv',rows,delimiter=',',header='time_s,left_bottom_m,right_bottom_m,left_inside_top,right_inside_top,box_contact_min_distance_m,joint_excess_rad',comments='')
    source_path=args.aligned or args.csv
    velocity_limits=np.array([PROFILE['velocity_limit'][name] for name in PROFILE['joint_names']])
    gates={
        'box_penetration_at_most_1mm':bool(rows[:,5].min() >= -.001),
        'floor_penetration_at_most_5mm':bool(floor_minimum >= -.005),
        'hard_joint_limits':bool(excess.max() <= 1e-4),
        'joint_speed_limits':bool((np.abs(motion.joint_vel)/velocity_limits).max() <= 1),
        'terminal_feet_on_top':bool(np.max(np.abs(rows[-k:,1:3]-box.height)) <= .01),
        'terminal_foot_hulls_inside_top':bool(rows[-k:,3:5].min() == 1),
    }
    report={'source_reference':str(source_path.resolve()),'source_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
            'frames':len(qpos),'fps':50,'yaw_alignment_rad':yaw,'floor_z_offset_m':offset,
            'box':asdict(box),'box_placement_method':'preserved from aligned input' if args.aligned else 'fit terminal foot footprint; diagnostic estimate, not video camera calibration',
            'terminal_foot_bottom_m':np.median(rows[-k:,1:3],axis=0).tolist(),
            'terminal_foot_inside_fraction':rows[-k:,3:5].mean(0).tolist(),
            'box_max_penetration_m':float(max(0,-rows[:,5].min())),
            'worst_box_contact':worst_box,
            'floor_max_penetration_m':-floor_minimum,
            'geometry_gates':gates,'geometry_gates_passed':all(gates.values()),
            'hard_joint_excess_rad':float(excess.max()),
            'peak_joint_velocity_rad_s':float(np.abs(motion.joint_vel).max()),
            'peak_joint_velocity_to_limit':float((np.abs(motion.joint_vel)/velocity_limits).max()),
            'training_approved':False,
            'status':'diagnostic only; requires calibrated contact correction, dynamics and 3s terminal standing validation'}
    (args.out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
