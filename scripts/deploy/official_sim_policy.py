"""Experimental policy loop for the isolated official simulator; never hardware."""
import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import socket
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--state-port', type=int, required=True)
    p.add_argument('--command-port', type=int, required=True)
    p.add_argument('--mode', choices=('stand', 'walk', 'jog'), default='stand')
    p.add_argument('--seconds', type=float, default=10.)
    p.add_argument('--sim-only', action='store_true', required=True)
    a = p.parse_args()
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
    tracker = SkillGoal(a.dataset, a.profile, [0., 0., .65], [1., 0., 0., 0.])
    initialized = False; records = []; failure = None; started = time.monotonic(); next_tick = started
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
                joints, stamps = [], []
                for group in ('leg', 'waist', 'head', 'arm'):
                    block = data['groups'][group]
                    if block['domain'] != 0: raise ValueError('simulator joint domain error')
                    for name, q, v, err in block['joints']:
                        joints.append(JointSample(name, q, v, err)); stamps.append(block['time'])
                state = State(tuple(joints), tuple(stamps), data['imu']['time'], tuple(data['imu']['q']), tuple(data['imu']['w']))
                ordered = order_and_validate(state, contract, time.monotonic())
                odom = data['odom']; position, quat = odom['position'], odom['q']
                if time.monotonic()-odom['time'] > .15 or not all(math.isfinite(x) for x in position+quat):
                    raise ValueError('invalid simulator odometry')
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
                elapsed = now-started
                mode = a.mode if 3. <= elapsed < a.seconds-3. else 'stand'
                goal = tracker.goal(position, quat, q, mode)
                if abs(float(goal[2*n+9])-position[2]) > .25 or abs(float(goal[2*n+2])-gravity[2]) > .8:
                    raise ValueError('root height or tilt termination')
                obs = tuple(v for term in range(5) for frame in history for v in frame[term])
                action = model.infer(obs, goal)
                if len(action) != n or not all(math.isfinite(v) for v in action): raise ValueError('invalid action')
                target = [d+s*v for d, s, v in zip(default, scale, action)]
                send(target); previous = action; tracker.advance()
                records.append({'elapsed': elapsed, 'mode': mode, 'root': position, 'q': quat,
                                'target': target, 'observation': obs, 'goal': goal.tolist(), 'action': action})
            except socket.timeout:
                if not initialized:
                    send(stand)
                    continue
                raise ValueError('simulator state timeout')
    except Exception as exc:
        failure = str(exc)
    finally:
        (a.out/'frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
        report = {'frames': len(records), 'failure': failure, 'mode': a.mode,
                  'elapsed': time.monotonic()-started, 'resets_after_start': 0,
                  'scope': 'official simulator exploratory run, simulation odometry ground truth'}
        (a.out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2), flush=True)
        states.close(); commands.close()
    if failure: raise SystemExit(1)


if __name__ == '__main__': main()
