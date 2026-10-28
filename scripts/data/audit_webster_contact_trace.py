"""Locate contact-height peaks in recorded native states without re-simulation."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from supermimic.data.design_metrics import BatchedDesignMetrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    args = parser.parse_args()
    model_path = args.trace/'native-model.mjb'
    physics_path = args.trace/'baseline-physics.npz'
    model = mujoco.MjModel.from_binary_path(str(model_path))
    data = mujoco.MjData(model)
    trace = np.load(physics_path)
    metrics = BatchedDesignMetrics(model, 'cpu')
    vertices = metrics.vertices.numpy(); geometry = metrics.geom_ids.numpy()
    groups = [g.numpy() for g in metrics.groups]
    heights = []; minimum_points = []
    for q, v in zip(trace['qpos'], trace['qvel']):
        data.qpos[:] = q; data.qvel[:] = v
        mujoco.mj_forward(model, data)
        matrix = data.geom_xmat.reshape(-1, 3, 3)
        z = (matrix[geometry, 2]*vertices).sum(-1)+data.geom_xpos[geometry, 2]
        ids = [int(g[np.argmin(z[g])]) for g in groups]
        minimum_points.append(ids); heights.append(z[ids])
    heights = np.asarray(heights); fps = float(trace['fps'][0])
    names = [model.joint(i).name.rsplit('/', 1)[-1] for i in range(1, model.njnt)]

    def event(step, side):
        data.qpos[:] = trace['qpos'][step]; data.qvel[:] = trace['qvel'][step]
        mujoco.mj_forward(model, data)
        point_id = minimum_points[step][side]; geom = int(geometry[point_id])
        point = data.geom_xpos[geom]+data.geom_xmat[geom].reshape(3, 3)@vertices[point_id]
        jacobian = np.zeros((3, model.nv)); rotation_jacobian = np.zeros_like(jacobian)
        mujoco.mj_jac(model, data, jacobian, rotation_jacobian, point, int(model.geom_bodyid[geom]))
        return {'time_s': (step+1)/fps, 'physics_step': step, 'side': ['left', 'right', 'nonfoot'][side],
            'height_m': float(heights[step, side]), 'both_foot_heights_m': heights[step, :2].tolist(),
            'geometry': model.geom(geom).name, 'point_world_m': point.tolist(),
            'root_velocity': trace['qvel'][step, :6].tolist(),
            'joints': {n: {'position_rad': float(trace['qpos'][step, 7+j]),
                'velocity_rad_s': float(trace['qvel'][step, 6+j]),
                'recorded_motor_torque_Nm': float(trace['actuator_force'][step, j]),
                'kinematic_height_jacobian_m_per_rad': float(jacobian[2, 6+j])} for j, n in enumerate(names)},
            'jacobian_scope': 'Instantaneous geometry sensitivity only; no dynamic feasibility claim'}

    peaks = [event(int(np.argmin(heights[:, side])), side) for side in range(3)]
    report = {'scope': 'Native recorded states, collision mesh hull vertices, no new dynamics or edited states',
        'physics_samples': len(heights), 'fps': fps, 'minimum_heights_m': heights.min(0).tolist(),
        'events': peaks, 'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [model_path, physics_path]}}
    with (args.trace/'contact-height-audit.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    np.savez(args.trace/'contact-height-series.npz', heights_m=heights, fps=np.array([fps]))
    print(json.dumps({k: v for k, v in report.items() if k != 'events'}))
    print(json.dumps([{k: e[k] for k in ['time_s', 'side', 'height_m', 'geometry']} for e in peaks]))


if __name__ == '__main__':
    main()
