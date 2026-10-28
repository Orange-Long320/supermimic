"""Private native inverse projection of frozen-policy position commands.

This is a diagnostic controller combination, not the learned policy alone.
Only commands change; the simulator, unactuated base, contacts and motor limits
stay unchanged. Independent actual-trajectory audits remain authoritative.
"""
import importlib.util
from pathlib import Path

import mujoco
import numpy as np


def load_snapshot(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CoupledProjection:
    def __init__(self, model, initial_data, effort, speed, source_dir):
        source_dir = Path(source_dir)
        native = load_snapshot('omni_private_native_inverse', source_dir/'native_inverse.py')
        prediction = load_snapshot('omni_private_joint_prediction', source_dir/'joint_prediction.py')
        native.position_acceleration_bounds = prediction.position_acceleration_bounds
        native.replanned_acceleration_bounds = prediction.replanned_acceleration_bounds
        root = int(model.jnt_bodyid[0])
        supports = []
        for contact in initial_data.contact:
            if contact.dist > 0:
                continue
            bodies = [int(model.geom_bodyid[g]) for g in contact.geom]
            if sum(model.body_rootid[b] == root for b in bodies) == 1:
                supports.extend(b for b in bodies if model.body_rootid[b] == root)
        if not supports:
            raise ValueError('Native projection requires an actual initial terrain contact')
        self.model = model
        self.preview = mujoco.MjData(model)
        self.controller = native.NativeContactController(model, initial_data.qpos.copy(), effort, speed, supports[0])
        self.controller.integration_aware = True
        self.controller.stabilize_support = False
        self.controller.full_acceleration_weight = np.ones(model.nv)
        self.acceleration_map = native.implicitfast_acceleration_map
        self.prediction = prediction

    def project(self, data, proposal):
        m = self.model
        d = self.preview
        d.qpos[:] = data.qpos
        d.qvel[:] = data.qvel
        d.qacc_warmstart[:] = data.qacc_warmstart
        d.ctrl[:] = proposal
        mujoco.mj_forward(m, d)
        desired = np.linalg.solve(self.acceleration_map(m, d), d.qacc.copy())
        lo, hi = m.jnt_range[1:].T
        minimum, maximum, horizon = self.prediction.replanned_acceleration_bounds(
            d.qpos[7:], d.qvel[6:], lo+self.controller.margin, hi-self.controller.margin,
            self.controller.velocity_limits, float(m.opt.timestep))
        needs_projection = bool(np.any(desired[6:] < minimum) or np.any(desired[6:] > maximum))
        if not needs_projection:
            return proposal, {'projected': False, 'prediction_horizon_s': horizon}
        self.controller.servo_seed = proposal.copy()
        self.controller.full_acceleration_reference = desired
        torque, report = self.controller.solve(data.qpos.copy(), data.qvel.copy())
        target = data.qpos[7:] + (torque-m.actuator_biasprm[:,2]*data.qvel[6:])/m.actuator_gainprm[:,0]
        report.update(projected=True, maximum_command_change_rad=float(abs(target-proposal).max()))
        return target, report
