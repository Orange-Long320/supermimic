"""Experimental policy loop for the isolated official simulator; never hardware."""
import argparse
from collections import deque
import gc
import json
import math
import os
from pathlib import Path
import socket
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Keyboard:
    def __init__(self, port):
        self.mode, self.when, self.ready = 'stand', 0., False
        self.lock = threading.Lock()
        mailbox = self
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                page = b'''<!doctype html><meta charset="utf-8"><title>X2 official simulation</title><h1>X2 official simulation</h1><p>Focus this page: hold W/S/A/D; Shift+W jog. Release to stand.</p><p id="mode">stand</p><script>
const keys=new Set();function current(){if(keys.has('KeyA'))return 'turn_left_inplace';if(keys.has('KeyD'))return 'turn_right_inplace';if(keys.has('KeyS'))return 'backward';if(keys.has('KeyW'))return keys.has('ShiftLeft')||keys.has('ShiftRight')?'jog':'walk';return 'stand'}
let inflight=false, pending=false;
async function send(){pending=true;if(inflight)return;inflight=true;
 while(pending){pending=false;let m=current();document.querySelector('#mode').textContent=m;
  const controller=new AbortController();const timer=setTimeout(()=>controller.abort(),300);
  try{const r=await fetch('/mode',{method:'POST',body:m,signal:controller.signal,cache:'no-store'});document.querySelector('#mode').textContent=r.status===409?'Waiting for simulation state':r.ok?m:'Connection error';}catch(_){document.querySelector('#mode').textContent='Connection error';}finally{clearTimeout(timer);}}
 inflight=false;}
onkeydown=e=>{keys.add(e.code);send()};onkeyup=e=>{keys.delete(e.code);send()};function stop(){keys.clear();send()}onblur=stop;document.addEventListener('visibilitychange',()=>{if(document.hidden)stop()});setInterval(send,100);
</script>'''
                self.send_response(200); self.send_header('Content-Length', str(len(page))); self.end_headers(); self.wfile.write(page)
            def do_POST(self):
                try:
                    if self.path != '/mode': raise ValueError()
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length < 32: raise ValueError()
                    mode = self.rfile.read(length).decode('ascii')
                    if mode not in ('stand', 'walk', 'jog', 'backward', 'turn_left_inplace', 'turn_right_inplace'): raise ValueError()
                    with mailbox.lock:
                        if not mailbox.ready:
                            self.send_error(409, 'Simulation state not ready'); return
                        mailbox.mode, mailbox.when = mode, time.monotonic()
                except (ValueError, UnicodeError): self.send_error(400); return
                self.send_response(204); self.end_headers()
            def log_message(self, *args): pass
        self.server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def mark_ready(self):
        with self.lock:
            self.ready = True

    def read(self):
        with self.lock:
            return self.mode if time.monotonic()-self.when < .5 else 'stand'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--state-port', type=int, required=True)
    p.add_argument('--command-port', type=int, required=True)
    p.add_argument('--mode', choices=('stand', 'walk', 'jog'), default='stand')
    p.add_argument('--schedule', type=Path, help='JSON array of [seconds, skill]')
    p.add_argument('--keyboard-port', type=int)
    p.add_argument('--jog-brake-fade', type=float, help='Experimental root-velocity fade duration in seconds')
    p.add_argument('--stand-anchor-gain', type=float, default=1., help='Experimental settled-stand position-error scaling')
    p.add_argument('--stand-gain-quiet-time', type=float, default=0., help='Require low root speed for this many seconds before gain ramps')
    p.add_argument('--compact-log', action='store_true', help='Log measured joints and per-step diagnostics without full network vectors')
    p.add_argument('--seconds', type=float, default=10.)
    p.add_argument('--sim-only', action='store_true', required=True)
    a = p.parse_args()
    if a.schedule and a.keyboard_port: raise ValueError('Select schedule or keyboard, not both')
    if os.environ.get('ROS_LOCALHOST_ONLY') != '1' or os.environ.get('ROS_DOMAIN_ID') != '179':
        raise RuntimeError('Requires isolated simulation ROS domain 179 and ROS_LOCALHOST_ONLY=1')
    if a.out.exists() or a.seconds <= 0:
        raise ValueError('New output and positive duration required')
    import numpy as np
    import torch
    from supermimic.deploy.skill_goal import SkillGoal
    from supermimic.deploy.torch_policy import TorchScriptPolicy
    from supermimic.deploy.core import Contract, JointSample, State, order_and_validate
    torch.set_num_threads(2)
    gc_started = {}
    def observe_gc(phase, info):
        if info['generation'] != 2:
            return
        stamp = time.monotonic()
        if phase == 'start':
            gc_started[2] = stamp
        else:
            print(json.dumps({'event': 'python_gc_gen2', 'monotonic': stamp,
                              'duration_s': stamp-gc_started.pop(2, stamp),
                              'collected': info['collected']}), flush=True)
    gc.callbacks.append(observe_gc)
    spec = json.loads(a.profile.read_text()); names = spec['joint_names']; n = len(names)
    if spec['robot'] != 'x2_ultra' or n != 31:
        raise ValueError('Expected X2 Ultra profile')
    model = TorchScriptPolicy(a.policy)
    # Limits here only reject obviously broken simulator inputs; they are not real robot limits.
    default = tuple(spec['default_joint_pos'][x] for x in names)
    scale = tuple(spec['action_scale'][x] for x in names)
    contract = Contract(tuple(names), default, scale, (-10.,)*n, (10.,)*n, (10.,)*n, .15, .10, 10, 729)
    with np.load(a.dataset/'walk_stop.npz', allow_pickle=False) as data:
        if tuple(data['joint_names']) != tuple(names): raise ValueError('reference joint order differs')
        stand = data['joint_pos'][-1].tolist()
    states = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    states.bind(('127.0.0.1', a.state_port)); states.settimeout(.1)
    commands = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    def send(target):
        commands.sendto(json.dumps({'target': target}, allow_nan=False).encode(), ('127.0.0.1', a.command_port))
    a.out.mkdir(parents=True)
    history, previous = deque(maxlen=10), (0.,)*n
    tracker = SkillGoal(a.dataset, a.profile, [0., 0., .65], [1., 0., 0., 0.], jog_brake_fade=a.jog_brake_fade, stand_anchor_gain=a.stand_anchor_gain, stand_gain_quiet_time=a.stand_gain_quiet_time)
    schedule = json.loads(a.schedule.read_text()) if a.schedule else None
    valid = ('stand', 'walk', 'jog', 'backward', 'turn_left_inplace', 'turn_right_inplace')
    if schedule and (schedule[0][0] != 0 or any(m not in valid for _, m in schedule) or any(schedule[i][0] >= schedule[i+1][0] for i in range(len(schedule)-1))):
        raise ValueError('Invalid schedule')
    keyboard = Keyboard(a.keyboard_port) if a.keyboard_port else None
    initialized = False; control_started = None; failure = None; failure_context = None
    frame_count = 0; first_elapsed = None; last_elapsed = None; max_overlap_error = None
    frame_log = (a.out/'frames.jsonl').open('x')
    # Keep the already-initialized PyTorch/MuJoCo object graph out of later
    # cyclic-GC scans. This bounded simulation process continues collecting
    # newly created cycles; it is not a real-time or hardware guarantee.
    gc.collect(); gc.freeze()
    started = time.monotonic(); next_tick = started
    previous_window = None
    print('policy ready for simulator', flush=True)
    try:
        while time.monotonic()-started < a.seconds:
            now = time.monotonic()
            if now < next_tick: time.sleep(next_tick-now)
            now = time.monotonic(); next_tick = max(next_tick+.02, now+.001)
            try:
                raw, _ = states.recvfrom(65535)
                states.setblocking(False)
                while True:
                    try: raw, _ = states.recvfrom(65535)
                    except BlockingIOError: break
                states.settimeout(.1)
                data = json.loads(raw)
                sample_now = time.monotonic()
                joints, stamps, effort_by_name = [], [], {}
                for group in ('leg', 'waist', 'head', 'arm'):
                    block = data['groups'][group]
                    if block['domain'] != 0: raise ValueError('simulator joint domain error')
                    for name, q, v, err in block['joints']:
                        joints.append(JointSample(name, q, v, err)); stamps.append(block['time'])
                    for name, effort in block.get('efforts', []):
                        if name in effort_by_name or not math.isfinite(effort):
                            raise ValueError('invalid simulator effort sample')
                        effort_by_name[name] = effort
                state = State(tuple(joints), tuple(stamps), data['imu']['time'], tuple(data['imu']['q']), tuple(data['imu']['w']))
                ordered = order_and_validate(state, contract, sample_now)
                odom = data['odom']; position, quat = odom['position'], odom['q']
                odom_age = sample_now-odom['time']
                if not all(math.isfinite(x) for x in position+quat):
                    failure_context = {'odometry_age_s': odom_age, 'odometry_finite': False}
                    raise ValueError('non-finite simulator odometry')
                if odom_age < 0 or odom_age > .15:
                    failure_context = {'odometry_age_s': odom_age, 'odometry_finite': True,
                                       'joint_group_age_s': {g: sample_now-data['groups'][g]['time']
                                                             for g in ('leg', 'waist', 'head', 'arm')},
                                       'imu_age_s': sample_now-data['imu']['time'],
                                       'packet_age_s': sample_now-data['time'],
                                       'receive_wait_s': sample_now-now}
                    raise ValueError('stale or future simulator odometry')
                q = tuple(j.position for j in ordered); qvel = tuple(j.velocity for j in ordered)
                w, x, y, z = state.orientation_wxyz
                gravity = (2*(w*y-x*z), -2*(w*x+y*z), -1+2*(x*x+y*y))
                terms = (tuple(v-d for v, d in zip(q, default)), qvel, state.angular_velocity, gravity, previous)
                history.append(terms)
                if not initialized:
                    tracker.position = np.asarray(position, dtype=float).copy()
                    tracker.yaw = tracker.rotation(quat).as_euler('xyz')[2]
                    initialized = True
                if len(history) < 10:
                    send(stand)
                    continue
                if control_started is None:
                    control_started = now
                    if keyboard: keyboard.mark_ready()
                elapsed = now-control_started
                mode = keyboard.read() if keyboard else (next(m for t, m in reversed(schedule) if elapsed >= t) if schedule else (a.mode if 3. <= elapsed < a.seconds-3. else 'stand'))
                goal = tracker.goal(position, quat, q, mode)
                window = tracker.scheduler.window().copy()
                overlap_error = float(np.max(np.abs(previous_window[1:]-window[:-1]))) if previous_window is not None else 0.
                previous_window = window
                if abs(float(goal[2*n+9])-position[2]) > .25 or abs(float(goal[2*n+2])-gravity[2]) > .8:
                    failure_context = {'position': position, 'orientation_wxyz': quat,
                                       'gravity_body': gravity, 'reference_height': float(goal[2*n+9]),
                                       'reference_gravity_z': float(goal[2*n+2]),
                                       'height_error': float(goal[2*n+9])-position[2],
                                       'gravity_z_error': float(goal[2*n+2])-gravity[2],
                                       'odometry_age_s': odom_age, 'packet_age_s': sample_now-data['time'],
                                       'joint_group_age_s': {g: sample_now-data['groups'][g]['time']
                                                             for g in ('leg', 'waist', 'head', 'arm')},
                                       'imu_age_s': sample_now-data['imu']['time'],
                                       'control_elapsed_s': elapsed}
                    raise ValueError('root height or tilt termination')
                obs = tuple(v for term in range(5) for frame in history for v in frame[term])
                action = model.infer(obs, goal)
                if len(action) != n or not all(math.isfinite(v) for v in action): raise ValueError('invalid action')
                target = [d+s*v for d, s, v in zip(default, scale, action)]
                send(target); previous = action; tracker.advance()
                row = {'elapsed': elapsed, 'mode': mode, 'root': position, 'q': quat,
                       'odom_age_s': odom_age,
                       'packet_age_s': sample_now-data['time'],
                       'loop_processing_s': time.monotonic()-now,
                       'body_angular_velocity': state.angular_velocity,
                       'joint_effort': [effort_by_name[n] for n in names] if set(effort_by_name)==set(names) else None,
                       'joint_pos': q, 'joint_vel': qvel, 'target': target,
                       'goal_current': window[0].tolist(), 'anchor': goal[-9:].tolist(),
                       'future_overlap_error': overlap_error,
                       'stand_gain': tracker.stand_gain, 'stand_quiet_steps': tracker.stand_quiet_steps,
                       'phase_source': tracker.scheduler.source, 'phase_index': tracker.scheduler.index,
                       'planned_frames': len(tracker.scheduler.plan)}
                if not a.compact_log: row.update(observation=obs, goal=goal.tolist(), action=action)
                frame_log.write(json.dumps(row)+'\n')
                frame_count += 1
                first_elapsed = elapsed if first_elapsed is None else first_elapsed
                last_elapsed = elapsed
                max_overlap_error = overlap_error if max_overlap_error is None else max(max_overlap_error, overlap_error)
                if frame_count % 50 == 0:
                    frame_log.flush()
            except socket.timeout:
                if not initialized:
                    send(stand)
                    continue
                raise ValueError('simulator state timeout')
    except Exception as exc:
        failure = str(exc)
    finally:
        if not frame_count and failure is None:
            failure = 'no policy frames received before timeout'
        frame_log.close()
        report = {'frames': frame_count, 'failure': failure, 'mode': a.mode,
                  'failure_context': failure_context,
                  'elapsed': time.monotonic()-started, 'resets_after_start': 0,
                  'control_seconds': last_elapsed-first_elapsed if frame_count > 1 else 0.,
                  'schedule': schedule, 'keyboard': bool(keyboard),
                  'jog_brake_fade': a.jog_brake_fade, 'phase_events': tracker.scheduler.events,
                  'stand_anchor_gain': a.stand_anchor_gain,
                  'stand_gain_quiet_time': a.stand_gain_quiet_time,
                  'max_future_overlap_error': max_overlap_error,
                  'scope': 'official simulator exploratory run, simulation odometry ground truth'}
        (a.out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2), flush=True)
        states.close(); commands.close()
        if keyboard: keyboard.server.shutdown(); keyboard.server.server_close()
        gc.callbacks.remove(observe_gc)
    if failure: raise SystemExit(1)


if __name__ == '__main__': main()
