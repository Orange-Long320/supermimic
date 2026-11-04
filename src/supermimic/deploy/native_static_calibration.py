"""Independent stationary IMU fixture for explicitly authorized native simulation.

The pelvis is physically fixed in a separate model only during calibration.
The estimator receives raw sensors/encoders, never the configured fixture pose.
The free-motion model is not changed or constrained by this helper.
"""
from collections import deque
import json

import mujoco
import numpy as np

from supermimic.deploy.imu_dead_reckoning import fit_static_initialization


def collect_native_static_calibration(scene, profile, position, quat, joints, out):
    spec = mujoco.MjSpec.from_file(str(scene))
    pelvis = spec.body('pelvis')
    free = [j for j in pelvis.joints if j.type == mujoco.mjtJoint.mjJNT_FREE]
    if len(free) != 1:
        raise ValueError('Expected one pelvis free joint for the calibration fixture')
    pelvis.pos = position; pelvis.quat = quat
    # Keep a dynamic body so its accelerometer includes the support reaction.
    # A body compiled directly into the world reports zero in this model.
    weld = spec.add_equality()
    weld.name = 'startup_stationary_fixture'
    weld.type = mujoco.mjtEq.mjEQ_WELD
    weld.objtype = mujoco.mjtObj.mjOBJ_BODY
    weld.name1 = 'pelvis'; weld.name2 = 'world'
    weld.solref[:] = [.005, 1.]
    model = spec.compile(); data = mujoco.MjData(model)
    names = profile['joint_names']
    ids = [model.joint(name).id for name in names]
    qadr, vadr = model.jnt_qposadr[ids], model.jnt_dofadr[ids]
    motor_by_joint = {int(j): i for i, j in enumerate(model.actuator_trnid[:, 0])}
    motors = np.array([motor_by_joint[j] for j in ids])
    kp = np.array([profile['stiffness'][n] for n in names])
    kd = np.array([profile['damping'][n] for n in names])
    lo, hi = model.actuator_ctrlrange[motors].T
    data.qpos[qadr] = joints
    gyro_adr = int(model.sensor('body-angular-velocity').adr[0])
    acc_adr = int(model.sensor('body-linear-acceleration').adr[0])
    sample_steps = round(.002/model.opt.timestep)
    window = deque(maxlen=1501)
    last_error = 'No complete static window'
    with (out/'calibration-imu.jsonl').open('x') as log:
        for step in range(round(10./model.opt.timestep)):
            data.ctrl[motors] = np.clip(kp*(joints-data.qpos[qadr])-kd*data.qvel[vadr], lo, hi)
            mujoco.mj_step(model, data)
            if (step+1) % sample_steps:
                continue
            mujoco.mj_forward(model, data)
            sample = {'time_s': float(data.time), 'gyro': data.sensordata[gyro_adr:gyro_adr+3].tolist(),
                      'acceleration': data.sensordata[acc_adr:acc_adr+3].tolist(),
                      'joint_pos': data.qpos[qadr].tolist(), 'joint_vel': data.qvel[vadr].tolist()}
            log.write(json.dumps(sample)+'\n'); window.append(sample)
            if data.time < 5. or (step+1) % 100:
                continue
            try:
                calibration = fit_static_initialization(list(window), declared_stationary=True)
            except ValueError as exc:
                last_error = str(exc); continue
            calibration['fixture'] = 'independent fixed-pelvis model; free-motion model unchanged'
            calibration['collection_end_s'] = float(data.time)
            calibration['free_motion_constraints'] = 0
            (out/'calibration.json').write_text(json.dumps(calibration, indent=2)+'\n')
            # Physical release state is separate from the sensor calibration.
            # It initializes the free simulator, never the estimator. Starting
            # the free body at an unrelated reference pose would invalidate
            # the just-measured gravity direction.
            np.savez(out/'calibration-release-state.npz', qpos=data.qpos.copy(), qvel=data.qvel.copy())
            return calibration, data.qpos.copy(), data.qvel.copy()
    (out/'calibration-failure.json').write_text(json.dumps({'reason': last_error})+'\n')
    raise ValueError('Static initialization refused: '+last_error)
