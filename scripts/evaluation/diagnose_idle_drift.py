"""Reconstruct stationary targets and separate late settling from target motion.

Uses the recorded anchor transform and the same kinematic profile to recover
the reference root. Official foot geometry is a kinematic audit, not force data.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def reconstruct(rows, profile):
    spec = json.loads(profile.read_text())
    names = spec['joint_names']
    n = len(names)
    xml = Path(spec['xml_path'])
    model = mujoco.MjModel.from_xml_path(str(xml if xml.is_absolute() else profile.parent/xml))
    actual, reference = mujoco.MjData(model), mujoco.MjData(model)
    adr = [model.jnt_qposadr[model.joint(k).id] for k in names]
    anchor = model.body(spec['anchor_body']).id
    default = np.array([spec['default_joint_pos'][k] for k in names])
    targets, quats, residuals = [], [], []
    for row in rows:
        if abs(row['stand_gain']-1.) > 1e-8:
            raise ValueError('Reconstruction requires original unscaled anchor inputs')
        g = np.asarray(row['goal_current'])
        actual.qpos[:7] = np.r_[row['root'], row['q']]
        actual.qpos[adr] = row['joint_pos']
        mujoco.mj_kinematics(model, actual)
        grav = g[2*n:2*n+3]
        base = Rotation.from_euler('xyz', [np.arctan2(-grav[1], -grav[2]),
                                           np.arcsin(np.clip(grav[0], -1., 1.)), 0.])
        reference.qpos[:7] = np.r_[0., 0., g[-1], base.as_quat()[[3,0,1,2]]]
        reference.qpos[adr] = g[:n]+default
        mujoco.mj_kinematics(model, reference)
        ra = Rotation.from_quat(actual.xquat[anchor][[1,2,3,0]]).as_matrix()
        rr = Rotation.from_quat(reference.xquat[anchor][[1,2,3,0]]).as_matrix()
        value = np.array(row['anchor'])
        if 'raw_anchor_translation' in row:
            value[:3] = row['raw_anchor_translation']
        columns = value[3:].reshape(3,2)
        relative = np.column_stack((columns, np.cross(columns[:,0], columns[:,1])))
        yaw_matrix = ra@relative@rr.T
        yaw = np.arctan2(yaw_matrix[1,0], yaw_matrix[0,0])
        rz = Rotation.from_euler('z', yaw)
        residuals.append(np.max(np.abs(yaw_matrix-rz.as_matrix())))
        target = actual.xpos[anchor]+ra@value[:3]-rz.apply(reference.xpos[anchor])
        # Reference FK above contains root height, so add it back after the
        # anchor offset subtraction. XY itself already is the root position.
        target[2] += g[-1]
        targets.append(target)
        quats.append((rz*base).as_quat()[[3,0,1,2]])
    return np.asarray(targets), np.asarray(quats), float(max(residuals)), spec


def feet_geometry(model, positions, quats, joints, names):
    data = mujoco.MjData(model)
    adr = [model.jnt_qposadr[model.joint(k).id] for k in names]
    feet = [model.body(s+'_ankle_roll_link').id for s in ('left', 'right')]
    spheres = [[i for i in range(model.ngeom) if model.geom_bodyid[i] == b
                and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE and model.geom_contype[i]] for b in feet]
    if any(not g for g in spheres):
        raise ValueError('Expected official foot collision spheres')
    ankle, gap = [], []
    for pos, quat, q in zip(positions, quats, joints):
        data.qpos[:7] = np.r_[pos, quat]
        data.qpos[adr] = q
        mujoco.mj_kinematics(model, data)
        ankle.append(data.xpos[feet].copy())
        gap.append([float(np.min(data.geom_xpos[g,2]-model.geom_size[g,0])) for g in spheres])
    return np.asarray(ankle), np.asarray(gap)


def analyze(folder, profile, scene):
    rows = [json.loads(line) for line in (folder/'policy/frames.jsonl').open()]
    result = json.loads((folder/'policy/result.json').read_text())
    t = np.array([r['elapsed'] for r in rows])
    if np.any(np.diff(t) <= 0):
        raise ValueError('Nonmonotonic trace')
    pos = np.array([r['root'] for r in rows])
    quat = np.array([r['q'] for r in rows])
    q = np.array([r['joint_pos'] for r in rows])
    v = np.array([r['joint_vel'] for r in rows])
    g = np.array([r['goal_current'] for r in rows])
    n = q.shape[1]
    mode = np.array([r['mode'] for r in rows])
    reference, rq, matrix_error, spec = reconstruct(rows, profile)
    yaw = Rotation.from_quat(rq[:,[1,2,3,0]]).as_euler('xyz')[:,2]
    predicted_step = Rotation.from_euler('z', yaw[:-1,None]).apply(g[:-1,2*n+6:2*n+9])*.02
    integration_error = np.linalg.norm(np.diff(reference[:,:2], axis=0)-predicted_step[:,:2], axis=1)
    model = mujoco.MjModel.from_xml_path(str(scene))
    actual_feet, actual_gap = feet_geometry(model, pos, quat, q, spec['joint_names'])
    default = np.array([spec['default_joint_pos'][name] for name in spec['joint_names']])
    target_feet, target_gap = feet_geometry(model, reference, rq, g[:,:n]+default, spec['joint_names'])
    dq_rms = np.sqrt(np.mean(v*v, axis=1))
    reference_error = reference[:,:2]-pos[:,:2]
    releases = np.flatnonzero((mode[:-1] == 'walk') & (mode[1:] == 'stand'))+1
    stops = []
    for start in releases:
        motion = np.flatnonzero(mode[start:] != 'stand')
        end = start+int(motion[0]) if len(motion) else len(rows)
        release = t[start]
        hold = np.flatnonzero((t >= release+5.) & (t < release+15.) & (np.arange(len(t)) < end))
        transition = np.flatnonzero((t >= release) & (t < release+1.) & (np.arange(len(t)) < end))
        static = np.flatnonzero((np.max(np.abs(g-g[-1]), axis=1) < 1e-7) & (np.arange(len(t)) >= start) & (np.arange(len(t)) < end))
        item = dict(release_s=float(release), observed_hold_after_5s=len(hold)*.02,
                    first_static_reference_after_release_s=float(t[static[0]]-release) if len(static) else None)
        if len(hold) > 1:
            dist = np.linalg.norm(pos[hold,:2]-pos[hold[0],:2], axis=1)
            peak = hold[np.argmax(dist)]
            joint_peak = hold[np.argmax(dq_rms[hold])]
            item.update(max_hold_displacement_m=float(dist.max()), net_hold_displacement_m=float(dist[-1]),
                        displacement_peak_after_release_s=float(t[peak]-release),
                        late_joint_motion_peak_rms_rad_s=float(dq_rms[joint_peak]),
                        late_joint_motion_peak_after_release_s=float(t[joint_peak]-release),
                        reference_range_m=np.ptp(reference[hold,:2], axis=0).tolist(),
                        reference_velocity_max=float(np.max(np.abs(g[hold,2*n+3:2*n+9]))),
                        root_error_at_hold_start_m=reference_error[hold[0]].tolist(),
                        root_error_at_hold_end_m=reference_error[hold[-1]].tolist(),
                        root_error_norm_mean_m=float(np.mean(np.linalg.norm(reference_error[hold], axis=1))),
                        root_error_norm_max_m=float(np.max(np.linalg.norm(reference_error[hold], axis=1))))
        if len(transition) > 1:
            z_rate = np.diff(g[transition,-1])/.02
            mismatch = z_rate-.5*(g[transition[:-1],2*n+8]+g[transition[1:],2*n+8])
            foot_rate = np.linalg.norm(np.diff(target_feet[transition,:,:2], axis=0), axis=2)/.02
            near = (target_gap[transition[:-1]] < .005) & (target_gap[transition[1:]] < .005)
            item.update(target_height_velocity_inconsistency_max_m_s=float(np.max(abs(mismatch))),
                        target_near_ground_ankle_speed_p95_m_s=[float(np.percentile(foot_rate[:,s][near[:,s]],95)) if near[:,s].any() else None for s in (0,1)],
                        target_min_foot_gaps_m=target_gap[transition].min(axis=0).tolist())
        stops.append(item)
    arrays = dict(time=t, mode=mode, position=pos, reference_position=reference, reference_error=reference_error,
                  goal=g, joint_speed_rms=dq_rms, actual_ankle=actual_feet, target_ankle=target_feet,
                  actual_foot_gap=actual_gap, target_foot_gap=target_gap)
    return dict(failure=result['failure'], control_seconds=result['control_seconds'], stops=stops,
                yaw_reconstruction_matrix_max_error=matrix_error,
                reference_integral_step_max_error_m=float(integration_error.max()),
                reference_integral_step_p99_error_m=float(np.percentile(integration_error,99)),
                source_frames_sha256=hashlib.sha256((folder/'policy/frames.jsonl').read_bytes()).hexdigest()), arrays


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'profile', 'scene', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--trials', nargs='+')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    names = a.trials or [x['planner'] for x in json.loads((a.run/'comparison.json').read_text())['trials']]
    outputs = {}
    for name in names:
        outputs[name], arrays = analyze(a.run/name, a.profile, a.scene)
        np.savez_compressed(a.out/(name+'.npz'), **arrays)
    report = dict(trials=outputs, scope='Recorded-state kinematic and timing diagnosis; no contact forces or causal acceptance from this audit alone')
    (a.out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({name:values for name,values in outputs.items()}, indent=2))


if __name__ == '__main__':
    main()
