"""Simulation-only offline IMU integration with assumed lowest-sphere support.

The default assumes the lowest sphere is planted and uses odometry only for scoring.
Optional oracle-height mode uses odometry height solely for near-floor selection;
no truth velocity or position correction is fed into the estimator. Near-floor geometry is
not contact force, and this experiment is not deployable. Each selected sphere's
instantaneous bottom point is assumed stationary; rolling, slipping, and timing
errors violate that assumption.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def read(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def stamp(row):
    return row['stamp'][0] + row['stamp'][1] * 1e-9


def interp(t, x, at):
    return np.column_stack([np.interp(at, t, x[:, i]) for i in range(x.shape[1])])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--support-gate', choices=('lowest-assumed', 'oracle-height'), default='lowest-assumed')
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('Refusing to overwrite output')
    imu = read(a.run / 'trial-record/imu.jsonl')
    odom = read(a.run / 'trial-record/odom.jsonl')
    joints = [r for r in read(a.run / 'trial-record/joints.jsonl') if r['group'] == 'leg']
    it = np.array([stamp(r) for r in imu]); jt = np.array([stamp(r) for r in joints])
    tt = np.array([stamp(r) for r in odom]); truth = np.array([r['position'] for r in odom])
    if any(np.any(np.diff(t) <= 0) for t in (it, jt, tt)):
        raise ValueError('nonmonotonic recorded timestamps')
    model = mujoco.MjModel.from_xml_path(str(a.scene)); data = mujoco.MjData(model)
    names = json.loads(a.profile.read_text())['joint_names']
    joint_ids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names}
    if min(joint_ids.values()) < 0:
        raise ValueError('missing joint')
    spheres = []; sphere_sides = []
    for side in ('left', 'right'):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, side + '_ankle_roll_link')
        selected = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body
                    and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE and model.geom_contype[i]]
        spheres += selected; sphere_sides += [side] * len(selected)
    if not spheres:
        raise ValueError('no collision spheres')
    sphere = np.array(spheres); radius = model.geom_size[sphere, 0]
    lever = np.array([.0239465, -.000228682, .04171])
    psite = np.zeros(3); vsite = np.zeros(3); started = False
    previous_t = None; next_update = -np.inf; start_lever = None; start_truth = None
    support_side = None; support_since = None
    result = []; observations = []; rejected = {'no_joint': 0, 'joint_stale': 0, 'no_near_floor': 0, 'support_hold': 0, 'feet_disagree': 0}
    for r in imu:
        t = stamp(r)
        if not tt[0]+.1 <= t <= tt[-1]-.1:
            continue
        rot = Rotation.from_quat(r['q_xyzw']); gyro = np.array(r['w'])
        accel = rot.apply(r['a']) - [0., 0., 9.81]
        if started and previous_t is not None:
            dt = t - previous_t
            if not 0 < dt <= .03:
                raise ValueError('IMU gap')
            psite += vsite*dt + .5*accel*dt*dt
            vsite += accel*dt
        previous_t = t
        if t >= next_update:
            next_update = t + .02
            ji = np.searchsorted(jt, t, side='right') - 1
            if ji < 0:
                rejected['no_joint'] += 1
            elif t-jt[ji] > .010:
                rejected['joint_stale'] += 1
            else:
                data.qpos[:] = model.qpos0; data.qvel[:] = 0
                data.qpos[:7] = np.r_[[0., 0., 0.], rot.as_quat()[[3, 0, 1, 2]]]
                data.qvel[3:6] = gyro
                for name, position, velocity, _ in joints[ji]['joints']:
                    i = joint_ids[name]
                    # Causal bounded extrapolation, as in the ROS producer.
                    data.qpos[model.jnt_qposadr[i]] = position + (t-jt[ji])*velocity
                    data.qvel[model.jnt_dofadr[i]] = velocity
                mujoco.mj_forward(model, data)
                relative_heights = data.geom_xpos[sphere, 2] - radius
                k = int(np.argmin(relative_heights)); geom = int(sphere[k])
                # Optional truth-height diagnostic only. Default support selection
                # uses no odometry and assumes the lowest sphere is planted.
                gap = None
                if a.support_gate == 'oracle-height':
                    gap = float(relative_heights[k] + np.interp(t, tt, truth[:, 2]))
                side = sphere_sides[k]
                if side != support_side:
                    support_side, support_since = side, t
                def velocity_for(index):
                    g = int(sphere[index]); point = data.geom_xpos[g].copy()
                    point[2] -= radius[index]
                    jac = np.zeros((3, model.nv)); jacr = np.zeros_like(jac)
                    mujoco.mj_jac(model, data, jac, jacr, point, int(model.geom_bodyid[g]))
                    return -jac @ data.qvel
                pelvis_v = velocity_for(k)
                other = np.array([i for i, candidate_side in enumerate(sphere_sides) if candidate_side != side])
                other_k = int(other[np.argmin(relative_heights[other])])
                # Conservative heuristic: when both feet are near the same level,
                # reject incompatible stationary-contact velocity hypotheses.
                disagree = (relative_heights[other_k] - relative_heights[k] < .01
                            and np.linalg.norm(pelvis_v - velocity_for(other_k)) > .2)
                if gap is not None and not -.01 <= gap <= .005:
                    rejected['no_near_floor'] += 1
                elif t - support_since < .04:
                    rejected['support_hold'] += 1
                elif disagree:
                    rejected['feet_disagree'] += 1
                else:
                    site_v = pelvis_v + rot.apply(np.cross(gyro, lever))
                    vsite = site_v
                    if not started:
                        started = True; psite[:] = 0.; start_lever = rot.apply(lever)
                        start_truth = interp(tt, truth, np.array([t]))[0]
                    true_v = (interp(tt, truth, np.array([t+.05]))[0] - interp(tt, truth, np.array([t-.05]))[0])/.1
                    observations.append([t, *(pelvis_v-true_v)])
        if started:
            pelvis_p = psite - (rot.apply(lever) - start_lever)
            pelvis_v = vsite - rot.apply(np.cross(gyro, lever))
            truth_p = interp(tt, truth, np.array([t]))[0]-start_truth
            truth_v = (interp(tt, truth, np.array([t+.05]))[0]-interp(tt, truth, np.array([t-.05]))[0])/.1
            result.append([t, *(pelvis_p-truth_p), *(pelvis_v-truth_v)])
    if len(result) < 2 or len(observations) < 2:
        raise ValueError('insufficient accepted observations')
    e = np.asarray(result); obs = np.asarray(observations); rmse = lambda x: float(np.sqrt(np.mean(x*x)))
    summary = {'scope': __doc__, 'duration_s': float(e[-1, 0]-e[0, 0]), 'imu_samples': len(e),
               'support_gate': a.support_gate, 'support_updates': len(obs), 'rejected': rejected,
               'horizontal_position_rmse_m': rmse(np.linalg.norm(e[:, 1:3], axis=1)),
               'vertical_position_rmse_m': rmse(e[:, 3]),
               'horizontal_velocity_rmse_m_s': rmse(np.linalg.norm(e[:, 4:6], axis=1)),
               'vertical_velocity_rmse_m_s': rmse(e[:, 6]),
               'final_position_error_m': e[-1, 1:4].tolist(),
               'max_horizontal_position_error_m': float(np.max(np.linalg.norm(e[:, 1:3], axis=1))),
               'max_vertical_position_error_m': float(np.max(np.abs(e[:, 3]))),
               'max_horizontal_velocity_error_m_s': float(np.max(np.linalg.norm(e[:, 4:6], axis=1))),
               'max_vertical_velocity_error_m_s': float(np.max(np.abs(e[:, 6]))),
               'guard_parameters': {'side_hold_s': .04, 'similar_height_m': .01, 'velocity_disagreement_m_s': .2},
               'support_observation_horizontal_error_rmse_m_s': rmse(np.linalg.norm(obs[:, 1:3], axis=1)),
               'support_observation_vertical_error_rmse_m_s': rmse(obs[:, 3])}
    a.out.write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
