"""Audit no-reset simulator stops. Geometric foot proximity is not contact force."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('trial', 'profile', 'scene', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('output exists')
    result = json.loads((args.trial / 'result.json').read_text())
    rows = [json.loads(line) for line in (args.trial / 'frames.jsonl').open()]
    if len(rows) < 2:
        raise ValueError('not enough recorded frames')
    t = np.asarray([r['elapsed'] for r in rows]); dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError('non-monotonic time')
    pos = np.asarray([r['root'] for r in rows])
    quat = np.asarray([r['q'] for r in rows])
    joints = np.asarray([r['joint_pos'] for r in rows])
    modes = np.asarray([r['mode'] for r in rows])
    rotations = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
    yaw = np.unwrap(rotations.as_euler('xyz')[:, 2])
    tilt = np.degrees(np.arccos(np.clip(rotations.as_matrix()[:, 2, 2], -1, 1)))
    model = mujoco.MjModel.from_xml_path(str(args.scene)); data = mujoco.MjData(model)
    names = json.loads(args.profile.read_text())['joint_names']
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    if any(i < 0 for i in ids):
        raise ValueError('missing joint in official model')
    qaddr = [model.jnt_qposadr[i] for i in ids]
    geoms = []
    for side in ('left', 'right'):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, side + '_ankle_roll_link')
        spheres = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body
                   and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE and model.geom_contype[i]]
        if not spheres:
            raise ValueError('missing foot collision spheres')
        geoms.append(spheres)
    gap = np.empty((len(rows), 2)); toe_delta = np.empty(len(rows))
    for i in range(len(rows)):
        data.qpos[:7] = np.r_[pos[i], quat[i]]; data.qpos[qaddr] = joints[i]
        mujoco.mj_kinematics(model, data)
        forward = np.array([np.cos(yaw[i]), np.sin(yaw[i])])
        toes = []
        for j, sphere in enumerate(geoms):
            gap[i, j] = np.min(data.geom_xpos[sphere, 2] - model.geom_size[sphere, 0])
            toes.append(np.max(data.geom_xpos[sphere, :2] @ forward + model.geom_size[sphere, 0]))
        toe_delta[i] = toes[0] - toes[1]
    changes = np.flatnonzero((modes[:-1] != 'stand') & (modes[1:] == 'stand')) + 1
    events = [e for e in result['phase_events'] if e.get('to') == 'stand' and e.get('from') != 'stand']
    if len(changes) != len(events):
        raise ValueError('release event count mismatch')
    stops = []
    for idx, event in zip(changes, events):
        end = int(np.searchsorted(t, t[idx] + 5., side='right'))
        segment = slice(idx, end)
        dist = np.linalg.norm(pos[segment, :2] - pos[idx, :2], axis=1)
        item = {'skill': str(modes[idx-1]), 'release_time_s': float(t[idx]),
                'selected_reference_frame': event['next_frame_at_selection'],
                'observed_after_release_s': float(t[end-1] - t[idx]),
                'max_excursion_5s_m': float(dist.max()),
                'min_root_height_5s_m': float(pos[segment, 2].min()),
                'max_tilt_5s_deg': float(tilt[segment].max()),
                'min_foot_gaps_5s_m': gap[segment].min(axis=0).tolist()}
        if t[idx] + 4 <= t[-1]:
            at4 = min(int(np.searchsorted(t, t[idx] + 4)), len(t)-1)
            item['displacement_4s_m'] = float(np.linalg.norm(pos[at4, :2] - pos[idx, :2]))
            after = (t >= t[idx] + 4) & (t < t[idx] + 5)
            if np.any(after):
                item['toe_offset_abs_p95_4_to_5s_m'] = float(np.percentile(abs(toe_delta[after]), 95))
        stops.append(item)
    stand = None
    if len(changes):
        start = int(np.searchsorted(t, t[changes[-1]] + 5.))
        if start < len(t)-1:
            distance = np.linalg.norm(pos[start:, :2] - pos[start, :2], axis=1)
            stand = {'seconds_after_settling': float(t[-1] - t[start]),
                     'net_displacement_m': float(distance[-1]), 'max_displacement_m': float(distance.max()),
                     'yaw_change_deg': float(np.degrees(yaw[-1] - yaw[start])),
                     'max_tilt_deg': float(tilt[start:].max()),
                     'minimum_root_height_m': float(pos[start:, 2].min()),
                     'toe_offset_abs_p95_m': float(np.percentile(abs(toe_delta[start:]), 95)),
                     'both_feet_near_ground_fraction': float(np.mean(np.all(gap[start:] < .005, axis=1)))}
    report = {'failure': result['failure'], 'resets_after_start': result['resets_after_start'],
              'seconds': float(t[-1] - t[0]), 'frames': len(rows), 'stops': stops, 'stand': stand,
              'max_future_overlap_error': result['max_future_overlap_error'],
              'dt_p99_s': float(np.percentile(dt, 99)), 'dt_max_s': float(dt.max()),
              'scope': 'single no-reset rollout; foot proximity is kinematic and not measured contact force'}
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
