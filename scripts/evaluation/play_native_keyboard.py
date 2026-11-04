"""Local native MuJoCo keyboard policy test, with no ROS or hardware transport.

Local execution is allowed only when explicitly requested by the user. This
entry uses the official scene and profile PD law, not the official ROS plugin.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from supermimic.deploy.numpy_policy import NumpyPolicy
from supermimic.deploy.skill_goal import SkillGoal


POLICY_ARCHIVE_SHA = 'd59809cb544e70c775bf0180174f5f57b3e5572a6639fe0821697639cf798063'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=3600.)
    p.add_argument('--headless', action='store_true')
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
    data = mujoco.MjData(model)
    history = deque(maxlen=10); previous = np.zeros(n); target = initial_q.copy()
    tracker = None; resets = 0; failed = None; paused = False

    def reset():
        nonlocal tracker, previous, target, failed
        if tracker is not None:
            tracker.close()
        mujoco.mj_resetData(model, data)
        data.qpos[:7] = np.r_[initial_pos, initial_rot]; data.qpos[qadr] = initial_q
        mujoco.mj_forward(model, data)
        history.clear(); previous = np.zeros(n); target = initial_q.copy(); failed = None
        tracker = SkillGoal(assets/'dataset', profile, initial_pos, initial_rot)

    reset()
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
            nonlocal resets, paused
            if action == glfw.PRESS:
                if key == glfw.KEY_ESCAPE: glfw.set_window_should_close(w, True)
                elif key == glfw.KEY_SPACE: paused = not paused
                elif key == glfw.KEY_R:
                    reset(); resets += 1
        glfw.set_key_callback(window, key_event)

    schedule = json.loads(a.schedule.read_text()) if a.schedule else [[0., 'stand']]
    inference_times = []; tick_times = deque(maxlen=100)
    started = time.monotonic(); frames = 0; mode = 'stand'
    print(json.dumps({'event': 'ready', 'local': True, 'policy_fixture_error': fixture_error,
                      'mujoco': mujoco.__version__, 'keyboard': 'W/S/A/D, Shift+W, space pause, R reset, Esc close'}), flush=True)
    log = (a.out/'frames.jsonl').open('x')
    try:
        while time.monotonic()-started < a.seconds:
            tick = time.monotonic()
            if window is not None:
                glfw.poll_events()
                if glfw.window_should_close(window): break
                pressed = lambda key: glfw.get_key(window, key) == glfw.PRESS
                focused = glfw.get_window_attrib(window, glfw.FOCUSED)
                mode = ('turn_left_inplace' if pressed(glfw.KEY_A) else
                        'turn_right_inplace' if pressed(glfw.KEY_D) else
                        'backward' if pressed(glfw.KEY_S) else
                        ('jog' if pressed(glfw.KEY_LEFT_SHIFT) or pressed(glfw.KEY_RIGHT_SHIFT) else 'walk')
                        if pressed(glfw.KEY_W) else 'stand') if focused else 'stand'
            else:
                mode = next(m for t, m in reversed(schedule) if data.time >= t)
            if not paused and failed is None:
                q = data.qpos[qadr].copy(); velocity = data.qvel[vadr].copy()
                w, x, y, z = data.xquat[imu_body]
                gravity = np.array([2*(w*y-x*z), -2*(w*x+y*z), -1+2*(x*x+y*y)])
                spatial = np.zeros(6)
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, imu_body, spatial, 1)
                history.append((q-default, velocity, spatial[:3].copy(), gravity, previous.copy()))
                if len(history) == 10:
                    goal = tracker.goal(data.qpos[:3].copy(), data.qpos[3:7].copy(), q, mode)
                    if abs(float(goal[2*n+9])-data.qpos[2]) > .25 or abs(float(goal[2*n+2])-gravity[2]) > .8:
                        failed = {'time_s': float(data.time), 'reason': 'root height or tilt termination'}
                    else:
                        obs = np.concatenate([frame[term] for term in range(5) for frame in history])
                        before = time.perf_counter(); action = policy.infer(obs, goal)
                        inference_times.append(time.perf_counter()-before)
                        target = default+scale*action; previous = action; tracker.advance()
                if failed is None:
                    for _ in range(substeps):
                        torque = kp*(target-data.qpos[qadr])-kd*data.qvel[vadr]
                        data.ctrl[motors] = np.clip(torque, limits[:, 0], limits[:, 1])
                        mujoco.mj_step(model, data)
                    mujoco.mj_forward(model, data)
                    log.write(json.dumps({'wall_s': tick-started, 'time_s': float(data.time), 'mode': mode,
                                          'root': data.qpos[:3].tolist(), 'reference': tracker.position.tolist(),
                                          'resets': resets})+'\n')
                    frames += 1
            if window is not None:
                camera.lookat[:] = data.qpos[:3]+[0., 0., .05]
                width, height = glfw.get_framebuffer_size(window)
                viewport = mujoco.MjrRect(0, 0, width, height)
                mujoco.mjv_updateScene(model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
                mujoco.mjr_render(viewport, scene, context)
                status = 'FAILED - R to reset' if failed else 'PAUSED' if paused else mode
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                   viewport, 'LOCAL X2 - '+status, 'W/S walk | A/D turn | Shift+W jog\nSpace pause | R reset | Esc exit', context)
                glfw.swap_buffers(window)
            tick_times.append(time.monotonic()-tick)
            if frames and frames % 100 == 0:
                log.flush()
                if window is not None:
                    glfw.set_window_title(window, f'X2 LOCAL | {mode} | tick {np.median(tick_times)*1000:.1f} ms | sim {data.time:.1f}s')
            if failed and a.headless: break
            remaining = .02-(time.monotonic()-tick)
            if remaining > 0: time.sleep(remaining)
    finally:
        log.close(); tracker.close()
        if window is not None:
            context.free(); scene.free(); glfw.destroy_window(window); glfw.terminate()
        result = {'scope': 'native MuJoCo CPU policy diagnostic; not official ROS pipeline or hardware validation',
                  'mujoco': mujoco.__version__, 'frames': frames, 'sim_seconds': float(data.time),
                  'wall_seconds': time.monotonic()-started, 'resets': resets, 'failure': failed,
                  'policy_fixture_max_error': fixture_error,
                  'inference_median_ms': float(np.median(inference_times)*1000) if inference_times else None,
                  'tick_median_ms': float(np.median(tick_times)*1000) if tick_times else None}
        (a.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
