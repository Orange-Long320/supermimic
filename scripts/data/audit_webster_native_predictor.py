"""Calibrate short MuJoCo CPU predictions against recorded native Warp physics.

Uses recorded servo commands and true states at each control-window boundary.
This does not change the running controller or certify a new motion.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from supermimic.data.design_metrics import BatchedDesignMetrics
from supermimic.robots.dimensions import PROFILE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    args = parser.parse_args()
    model = mujoco.MjModel.from_binary_path(str(args.trace/'native-model.mjb'))
    physics = np.load(args.trace/'baseline-physics.npz')
    controls = np.load(args.trace/'baseline-trajectory.npz')
    ratio = round(float(physics['fps'][0])/float(controls['fps'][0]))
    assert len(physics['qpos']) == len(controls['actions'])*ratio
    assert abs(model.opt.timestep*float(physics['fps'][0])-1) < 1e-8
    names = PROFILE['joint_names']
    assert [model.joint(int(j)).name.rsplit('/', 1)[-1] for j in model.actuator_trnid[:, 0]] == list(names)
    scale = PROFILE['action_scale']
    scale = np.array([scale[n] for n in names]) if isinstance(scale, dict) else scale
    q0 = np.array([PROFILE['default_joint_pos'][n] for n in names])
    metrics = BatchedDesignMetrics(model, 'cpu')
    vertices = metrics.vertices.numpy(); geoms = metrics.geom_ids.numpy()
    groups = [g.numpy() for g in metrics.groups[:2]]
    predicted = mujoco.MjData(model); actual = mujoco.MjData(model)

    def height(data):
        matrix = data.geom_xmat.reshape(-1, 3, 3)
        z = (matrix[geoms, 2]*vertices).sum(-1)+data.geom_xpos[geoms, 2]
        return np.array([z[g].min() for g in groups])

    errors = []; predictions = []
    for step, action in enumerate(controls['actions']):
        mujoco.mj_resetData(model, predicted)
        previous = step*ratio-1
        predicted.qpos[:] = physics['qpos'][previous] if step else physics['initial_qpos']
        predicted.qvel[:] = physics['qvel'][previous] if step else physics['initial_qvel']
        predicted.ctrl[:] = q0+action*scale
        mujoco.mj_forward(model, predicted)
        for substep in range(ratio):
            index = step*ratio+substep
            mujoco.mj_step(model, predicted)
            mujoco.mj_forward(model, predicted)
            actual.qpos[:] = physics['qpos'][index]; actual.qvel[:] = physics['qvel'][index]
            mujoco.mj_forward(model, actual)
            errors.append([substep+1, np.abs(predicted.qpos-actual.qpos).max(),
                np.abs(predicted.qvel-actual.qvel).max(), np.abs(height(predicted)-height(actual)).max()])
            predictions.append(np.r_[predicted.qpos, predicted.qvel])
    errors = np.asarray(errors)
    report = {'scope': 'Teacher-forced control-window boundaries, recorded actions, zero prediction warm start; CPU MuJoCo versus native Warp',
        'control_windows': len(controls['actions']), 'physics_dt_s': model.opt.timestep,
        'maximum_errors_by_horizon': [{'horizon_s': h*model.opt.timestep,
            'qpos': float(errors[errors[:, 0] == h, 1].max()),
            'qvel': float(errors[errors[:, 0] == h, 2].max()),
            'foot_height_m': float(errors[errors[:, 0] == h, 3].max())} for h in range(1, ratio+1)],
        'input_sha256': {name: hashlib.sha256((args.trace/name).read_bytes()).hexdigest()
            for name in ['native-model.mjb', 'baseline-physics.npz', 'baseline-trajectory.npz']},
        'approved_for_controller_use': False}
    with (args.trace/'native-predictor-audit.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    np.savez(args.trace/'native-predictor-audit.npz', errors=errors, predicted_states=predictions)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
