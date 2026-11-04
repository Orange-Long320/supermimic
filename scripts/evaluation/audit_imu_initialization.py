"""Frozen-input sensor-only ablations; oracle files are loaded only for scoring."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from supermimic.deploy.imu_dead_reckoning import ImuDeadReckoning


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists(): raise ValueError('New output required')
    raw = [json.loads(line) for line in (a.run/'imu.jsonl').open()]
    calibration = json.loads(a.calibration.read_text())
    if any(s['resets'] for s in raw): raise ValueError('Use a no-reset trace')
    stamp = np.array([s['time_s'] for s in raw])
    acc = np.array([s['acceleration'] for s in raw])
    variants = [('legacy', False, False, True, True),
                ('bias_only', False, True, True, True),
                ('tilt_only', True, False, True, True),
                ('static_full', True, True, True, True),
                ('static_no_leveling', True, True, False, True),
                ('static_no_zupt', True, True, True, False),
                ('static_no_leveling_no_zupt', True, True, False, False)]
    offset = np.array([.0239465, -.000228682, .04171])
    replay = {}
    # This whole phase has no access to oracle measurements.
    for name, tilt, bias, leveling, zupt in variants:
        f = ImuDeadReckoning(zupt=zupt, gravity_leveling=leveling, require_initialization=False)
        if tilt or bias:
            c = dict(calibration)
            if not tilt: c['quat_wxyz'] = [1., 0., 0., 0.]
            if not bias: c['gyro_bias'] = [0., 0., 0.]
            f.initialize(c)
        positions, quats, world, updates, velocity_jumps = [], [], [], [], []
        for sample in raw:
            before = f.x[2:].copy()
            f.update(sample['time_s'], sample['gyro'], sample['acceleration'])
            position = np.r_[f.x[:2], 0.]-f.rotation.apply(offset)+f.initial_rotation.apply(offset)
            positions.append(position[:2]); quats.append(f.rotation.as_quat())
            world.append(f.last_world_acceleration.copy()); updates.append(f.last_zupt)
            velocity_jumps.append(f.x[2:]-before)
        replay[name] = {'position': np.array(positions), 'quat': np.array(quats),
                        'world': np.array(world), 'zupt': np.array(updates),
                        'velocity_change': np.array(velocity_jumps)}
    # Only now load ground truth, for an independent evaluator. It cannot
    # change a replayed state or choose an estimator correction.
    truth = [json.loads(line) for line in (a.run/'truth-imu.jsonl').open()]
    tt = np.array([s['time_s'] for s in truth])
    if not np.array_equal(stamp, tt): raise ValueError('Unmatched sensor/scoring acquisitions')
    position = np.array([s['position'] for s in truth])
    rotation = Rotation.from_quat(np.array([s['quat_wxyz'] for s in truth])[:, [1, 2, 3, 0]])
    velocity = np.gradient(position, stamp, axis=0)
    speed = np.linalg.norm(velocity[:, :2], axis=1)
    true_world = rotation.apply(acc)-[0., 0., 9.81]
    report = {'scope': 'same raw IMU input; truth loaded after all estimates, scoring only',
              'run': str(a.run), 'calibration': str(a.calibration), 'seconds': float(stamp[-1]-stamp[0]),
              'samples': len(raw), 'calibration_gyro_bias': calibration['gyro_bias'], 'variants': {}}
    arrays = {'time': stamp, 'truth_position': position, 'truth_speed': speed}
    for name, result in replay.items():
        error = result['position']-position[:, :2]
        angles = np.degrees((rotation.inv()*Rotation.from_quat(result['quat'])).magnitude())
        projection_error = result['world']-true_world
        use = result['zupt']; moving = speed > .05
        report['variants'][name] = {
            'final_xy_error_m': float(np.linalg.norm(error[-1])),
            'max_xy_error_m': float(np.linalg.norm(error, axis=1).max()),
            'final_xy_error_vector_m': error[-1].tolist(),
            'attitude_error_initial_deg': float(angles[0]),
            'attitude_error_final_deg': float(angles[-1]),
            'gravity_projection_xy_rmse_m_s2': float(np.sqrt(np.mean(projection_error[1:, :2]**2))),
            'zupt_updates': int(use.sum()), 'zupt_while_truth_speed_above_005': int(np.sum(use & moving)),
            'truth_speed_at_zupt_max_m_s': float(speed[use].max()) if use.any() else None,
            'early_zupt_time_range_s': [float(stamp[use][0]), float(stamp[use][-1])] if use.any() else None}
        for key, value in result.items(): arrays[name+'_'+key] = value
    a.out.mkdir()
    (a.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(a.out/'replays.npz', **arrays)
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
