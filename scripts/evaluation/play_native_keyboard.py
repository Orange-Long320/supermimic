"""Local native MuJoCo keyboard policy test, with no ROS or hardware transport.

Local execution is allowed only when explicitly requested by the user. This
entry uses the official scene and profile PD law, not the official ROS plugin.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import struct
import time
import threading
import traceback
import zlib

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from supermimic.deploy.numpy_policy import NumpyPolicy
from supermimic.deploy.skill_goal import SkillGoal
from supermimic.deploy.imu_dead_reckoning import EncoderImuState


POLICY_ARCHIVE_SHA = 'd59809cb544e70c775bf0180174f5f57b3e5572a6639fe0821697639cf798063'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=3600.)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--state-source', choices=('imu', 'truth-diagnostic'), default='imu',
                   help='Sensor-only IMU by default; truth is an explicit diagnostic comparison')
    p.add_argument('--imu-initialization', choices=('static', 'legacy-diagnostic'), default='static')
    p.add_argument('--imu-no-zupt', action='store_true')
    p.add_argument('--imu-no-gravity-leveling', action='store_true')
    p.add_argument('--schedule', type=Path, help='Headless-only JSON [seconds, mode] pairs')
    a = p.parse_args()
    if a.out.exists() or a.seconds <= 0 or (a.schedule and not a.headless):
        p.error('Require a new output, positive duration, and headless scripted tests')
    assets = a.assets.resolve()
    manifest = json.loads((assets/'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if hashlib.sha256((assets/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Asset digest mismatch: '+name)
    profile = assets/'fk-only/profile.json'
    spec = json.loads(profile.read_text())
    names = spec['joint_names']; n = len(names)
    if n != 31 or spec['robot'] != 'x2_ultra':
        raise ValueError('Expected X2 Ultra profile')
    model = mujoco.MjModel.from_xml_path(str(assets/'model_info/scene.xml'))
    substeps = round(.02/model.opt.timestep)
    if model.nu != n or abs(substeps*model.opt.timestep-.02) > 1e-9:
        raise ValueError('Unexpected official scene actuator count/timestep')
    ids = [model.joint(name).id for name in names]
    qadr = model.jnt_qposadr[ids]; vadr = model.jnt_dofadr[ids]
    motor_by_joint = {int(j): i for i, j in enumerate(model.actuator_trnid[:, 0])}
    if set(motor_by_joint) != set(ids):
        raise ValueError('Actuator and profile joint sets differ')
    motors = np.array([motor_by_joint[j] for j in ids])
    limits = model.actuator_ctrlrange[motors]
    if not np.all(model.actuator_ctrllimited[motors]):
        raise ValueError('Expected official torque-limited motors')
    default = np.array([spec['default_joint_pos'][name] for name in names])
    scale = np.array([spec['action_scale'][name] for name in names])
    kp = np.array([spec['stiffness'][name] for name in names])
    kd = np.array([spec['damping'][name] for name in names])
    # The official bridge uses pelvis IMU, with pelvis odometry for the goal.
    imu_body = model.body('pelvis').id
    gyro_sensor = model.sensor('body-angular-velocity').id
    accel_sensor = model.sensor('body-linear-acceleration').id
    gyro_adr = int(model.sensor_adr[gyro_sensor]); accel_adr = int(model.sensor_adr[accel_sensor])
    site = model.site('imu_0').id
    if (model.site_bodyid[site] != imu_body or
            not np.allclose(model.site_quat[site], [1., 0., 0., 0.]) or
            model.sensor_objid[gyro_sensor] != site or model.sensor_objid[accel_sensor] != site):
        raise ValueError('Expected the declared fixed pelvis IMU mount')
    if abs(round(.002/model.opt.timestep)*model.opt.timestep-.002) > 1e-9:
        raise ValueError('Scene timestep must represent 500 Hz IMU acquisitions')
    imu_substeps = round(.002/model.opt.timestep)
    with np.load(assets/'dataset/walk_stop.npz', allow_pickle=False) as z:
        if list(z['joint_names']) != names:
            raise ValueError('Reference joint order mismatch')
        root = list(z['body_names']).index('pelvis')
        initial_q = z['joint_pos'][-1].copy()
        initial_pos = z['body_pos_w'][-1, root].copy(); initial_pos[:2] = 0.
        roll, pitch, _ = Rotation.from_quat(z['body_quat_w'][-1, root][[1, 2, 3, 0]]).as_euler('xyz')
        initial_rot = Rotation.from_euler('xyz', [roll, pitch, 0.]).as_quat()[[3, 0, 1, 2]]
    policy = NumpyPolicy(assets/'numpy-policy-11500.npz', expected_sha256=POLICY_ARCHIVE_SHA)
    fixture = json.loads((assets/'policy-fixture.json').read_text())
    fixture_error = float(np.max(np.abs(policy.infer(fixture['obs'], fixture['goal'])-fixture['expected'])))
    if fixture_error > 1e-4:
        raise ValueError('Policy fixture mismatch')
    a.out.mkdir(parents=True)
    imu_log = (a.out/'imu.jsonl').open('x')
    truth_log = (a.out/'truth-imu.jsonl').open('x')
    data = mujoco.MjData(model)
    history = deque(maxlen=10); previous = np.zeros(n); target = initial_q.copy()
    tracker = None; observer = None; resets = 0; failed = None; paused = False
    calibration = None
    if a.state_source == 'imu' and a.imu_initialization == 'static':
        from supermimic.deploy.native_static_calibration import collect_native_static_calibration
        calibration = collect_native_static_calibration(assets/'model_info/scene.xml', spec,
                                                         initial_pos, initial_rot, initial_q, a.out)

    def sample_imu():
        # Sensor adapter: no framequat, framepos, root velocity, contact or body
        # pose reads. mj_forward has generated these physical IMU channels.
        gyro = data.sensordata[gyro_adr:gyro_adr+3].copy()
        acc = data.sensordata[accel_adr:accel_adr+3].copy()
        observer.update(float(data.time), gyro, acc)
        imu_log.write(json.dumps({'time_s': float(data.time), 'gyro': gyro.tolist(),
                                  'acceleration': acc.tolist(), 'joint_pos': data.qpos[qadr].tolist(),
                                  'joint_vel': data.qvel[vadr].tolist(), 'resets': resets})+'\n')
        # Separate, write-only oracle recorder. Never read by the state observer
        # or controller; used after the run to score individual error sources.
        truth_log.write(json.dumps({'time_s': float(data.time), 'position': data.qpos[:3].tolist(),
                                    'quat_wxyz': data.qpos[3:7].tolist(),
                                    'velocity': data.qvel[:3].tolist(), 'resets': resets})+'\n')

    def reset():
        nonlocal tracker, observer, previous, target, failed
        if tracker is not None:
            tracker.close()
        mujoco.mj_resetData(model, data)
        data.qpos[:7] = np.r_[initial_pos, initial_rot]; data.qpos[qadr] = initial_q
        mujoco.mj_forward(model, data)
        history.clear(); previous = np.zeros(n); target = initial_q.copy(); failed = None
        if a.state_source == 'imu':
            observer = EncoderImuState(assets/'fk-only/fk-only.xml', names, model.site_pos[site],
                                      require_initialization=calibration is not None)
            if calibration is not None:
                observer.filter.initialize(calibration)
            observer.filter.zupt = not a.imu_no_zupt
            observer.filter.gravity_leveling = not a.imu_no_gravity_leveling
            sample_imu()
            estimated_pos, estimated_quat, _, _ = observer.state(data.qpos[qadr])
            tracker = SkillGoal(assets/'dataset', profile, estimated_pos, estimated_quat)
        else:
            tracker = SkillGoal(assets/'dataset', profile, initial_pos, initial_rot)

    reset()
    def control_step(current_mode, tick):
        nonlocal failed, previous, target, frames
        mode = current_mode
        q = data.qpos[qadr].copy(); velocity = data.qvel[vadr].copy()
        if observer is not None:
            position, quat, omega, gravity = observer.state(q)
        else:
            # Explicit oracle diagnostic only; never the default policy path.
            position, quat = data.qpos[:3].copy(), data.qpos[3:7].copy()
            w, x, y, z = data.xquat[imu_body]
            gravity = np.array([2*(w*y-x*z), -2*(w*x+y*z), -1+2*(x*x+y*y)])
            spatial = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, imu_body, spatial, 1)
            omega = spatial[:3].copy()
        history.append((q-default, velocity, omega, gravity, previous.copy()))
        if len(history) == 10:
            goal = tracker.goal(position, quat, q, mode)
            if abs(float(goal[2*n+9])-position[2]) > .25 or abs(float(goal[2*n+2])-gravity[2]) > .8:
                failed = {'time_s': float(data.time), 'reason': 'root height or tilt termination'}
            else:
                obs = np.concatenate([frame[term] for term in range(5) for frame in history])
                before = time.perf_counter(); action = policy.infer(obs, goal)
                inference_times.append(time.perf_counter()-before)
                target = default+scale*action; previous = action; tracker.advance()
        if failed is None:
            for substep in range(substeps):
                torque = kp*(target-data.qpos[qadr])-kd*data.qvel[vadr]
                data.ctrl[motors] = np.clip(torque, limits[:, 0], limits[:, 1])
                mujoco.mj_step(model, data)
                if observer is not None and (substep+1) % imu_substeps == 0:
                    # Recompute sensors at their acquisition timestamp so their
                    # values are not paired with the previous integration step.
                    mujoco.mj_forward(model, data)
                    sample_imu()
            mujoco.mj_forward(model, data)
            estimated = observer.state(data.qpos[qadr])[0] if observer is not None else position
            log.write(json.dumps({'wall_s': tick-started, 'time_s': float(data.time), 'mode': mode,
                                  'root': data.qpos[:3].tolist(), 'reference': tracker.position.tolist(),
                                  'estimated_root': estimated.tolist(), 'state_source': a.state_source,
                                  'joint_pos': data.qpos[qadr].tolist(), 'joint_vel': data.qvel[vadr].tolist(),
                                  'truth_scope': 'rendering and offline scoring only',
                                  'resets': resets})+'\n')
            frames += 1

    snapshot_lock = threading.Lock()
    snapshot_qpos = data.qpos.copy(); snapshot_qvel = data.qvel.copy(); snapshot_time = 0.
    stop = threading.Event(); reset_requested = False
    render_data = mujoco.MjData(model)
    window = None
    if not a.headless:
        import glfw
        if not glfw.init():
            raise RuntimeError('Cannot initialize local GLFW window')
        window = glfw.create_window(1100, 820, 'X2 local simulation - preparing', None, None)
        if not window:
            glfw.terminate(); raise RuntimeError('Cannot create local simulation window')
        glfw.make_context_current(window); glfw.swap_interval(0)
        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        scene = mujoco.MjvScene(model, maxgeom=10000)
        camera = mujoco.MjvCamera(); option = mujoco.MjvOption()
        camera.azimuth = 140; camera.elevation = -15; camera.distance = 2.5
        glfw.set_scroll_callback(window, lambda w, x, y: setattr(camera, 'distance', float(np.clip(camera.distance-y*.15, .7, 8.))))

        def key_event(w, key, scancode, action, mods):
            nonlocal reset_requested, paused
            if action == glfw.PRESS:
                if key == glfw.KEY_ESCAPE: glfw.set_window_should_close(w, True)
                elif key == glfw.KEY_SPACE: paused = not paused
                elif key == glfw.KEY_R:
                    reset_requested = True
        glfw.set_key_callback(window, key_event)

    schedule = json.loads(a.schedule.read_text()) if a.schedule else [[0., 'stand']]
    inference_times = []; tick_times = deque(maxlen=100)
    started = time.monotonic(); frames = 0; mode = 'stand'; screenshot_written = False
    next_render = started
    print(json.dumps({'event': 'ready', 'local': True, 'state_source': a.state_source, 'policy_fixture_error': fixture_error,
                      'mujoco': mujoco.__version__, 'keyboard': 'W/S/A/D, Shift+W, space pause, R reset, Esc close'}), flush=True)
    log = (a.out/'frames.jsonl').open('x')
    def control_loop():
        nonlocal frames, resets, reset_requested, failed
        nonlocal snapshot_qpos, snapshot_qvel, snapshot_time
        deadline = time.monotonic()
        try:
            while not stop.is_set() and time.monotonic()-started < a.seconds:
                tick = time.monotonic()
                if reset_requested:
                    reset(); resets += 1; reset_requested = False
                current_mode = mode if window is not None else next(m for t, m in reversed(schedule) if data.time >= t)
                if not paused and failed is None:
                    control_step(current_mode, tick)
                with snapshot_lock:
                    snapshot_qpos = data.qpos.copy(); snapshot_qvel = data.qvel.copy(); snapshot_time = float(data.time)
                tick_times.append(time.monotonic()-tick)
                if frames and frames % 100 == 0: log.flush(); imu_log.flush(); truth_log.flush()
                if failed and a.headless: break
                deadline += .02
                if time.monotonic()-deadline > .5: deadline = time.monotonic()
                stop.wait(max(0., deadline-time.monotonic()))
        except Exception as exc:
            failed = {'time_s': float(data.time), 'reason': str(exc)}
            traceback.print_exc()
        finally:
            stop.set()

    worker = threading.Thread(target=control_loop, name='native-physics-policy')
    worker.start()
    try:
        if window is None:
            worker.join()
        while window is not None and not stop.is_set():
            glfw.poll_events()
            if glfw.window_should_close(window): break
            pressed = lambda key: glfw.get_key(window, key) == glfw.PRESS
            focused = glfw.get_window_attrib(window, glfw.FOCUSED)
            mode = ('turn_left_inplace' if pressed(glfw.KEY_A) else
                    'turn_right_inplace' if pressed(glfw.KEY_D) else
                    'backward' if pressed(glfw.KEY_S) else
                    ('jog' if pressed(glfw.KEY_LEFT_SHIFT) or pressed(glfw.KEY_RIGHT_SHIFT) else 'walk')
                    if pressed(glfw.KEY_W) else 'stand') if focused else 'stand'
            with snapshot_lock:
                render_data.qpos[:] = snapshot_qpos; render_data.qvel[:] = snapshot_qvel; render_data.time = snapshot_time
            mujoco.mj_forward(model, render_data)
            camera.lookat[:] = render_data.qpos[:3]+[0., 0., .05]
            width, height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, width, height)
            mujoco.mjv_updateScene(model, render_data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
            mujoco.mjr_render(viewport, scene, context)
            status = 'FAILED - R to reset' if failed else 'PAUSED' if paused else mode
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               viewport, 'LOCAL X2 '+a.state_source+' - '+status, 'W/S walk | A/D turn | Shift+W jog\nSpace pause | R reset | Esc exit', context)
            if not screenshot_written:
                pixels = np.zeros((height, width, 3), dtype=np.uint8)
                mujoco.mjr_readPixels(pixels, None, viewport, context)
                def chunk(kind, payload):
                    return (struct.pack('!I', len(payload))+kind+payload
                            +struct.pack('!I', zlib.crc32(kind+payload)))
                scanlines = b''.join(b'\0'+row.tobytes() for row in pixels[::-1])
                png = (b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR', struct.pack('!2I5B', width, height, 8, 2, 0, 0, 0))
                       +chunk(b'IDAT', zlib.compress(scanlines))+chunk(b'IEND', b''))
                (a.out/'initial-window.png').write_bytes(png)
                screenshot_written = True
                print(json.dumps({'event': 'window_rendered', 'width': width, 'height': height}), flush=True)
            glfw.swap_buffers(window)
            if frames:
                glfw.set_window_title(window, f'X2 LOCAL | {mode} | physics {np.median(tick_times)*1000:.1f} ms | sim {render_data.time:.1f}s')
            next_render = max(next_render+1/30., time.monotonic())
            stop.wait(max(0., next_render-time.monotonic()))
    finally:
        stop.set(); worker.join(timeout=5.)
        if worker.is_alive(): raise RuntimeError("Physics worker did not stop")
        log.close(); imu_log.close(); truth_log.close(); tracker.close()
        if window is not None:
            context.free(); glfw.destroy_window(window); glfw.terminate()
        result = {'scope': 'native MuJoCo CPU policy diagnostic; not official ROS pipeline or hardware validation',
                  'mujoco': mujoco.__version__, 'frames': frames, 'sim_seconds': float(data.time),
                  'state_source': a.state_source,
                  'imu_initialization': a.imu_initialization,
                  'gyro_bias': observer.filter.gyro_bias.tolist() if observer is not None else None,
                  'gravity_leveling': not a.imu_no_gravity_leveling,
                  'zupt': not a.imu_no_zupt,
                  'imu_samples': observer.filter.samples if observer is not None else 0,
                  'zupt_updates': observer.filter.zupt_updates if observer is not None else 0,
                  'wall_seconds': time.monotonic()-started, 'resets': resets, 'failure': failed,
                  'policy_fixture_max_error': fixture_error,
                  'inference_median_ms': float(np.median(inference_times)*1000) if inference_times else None,
                  'tick_median_ms': float(np.median(tick_times)*1000) if tick_times else None}
        (a.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
