"""Minimal read-only 200 Hz IMU integration. No filter, no contact logic.

Deliberately the simplest possible estimator: latest torso IMU sample each 200 Hz
tick, gravity removed, double-integrated. Optional one-line static offset removal
is off by default. Never publishes; status is always control_ready=false.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=30.)
    p.add_argument('--rate', type=float, default=200.)
    p.add_argument('--gravity-sign', choices=('neg', 'pos'), default='neg',
                   help='sign applied to R(q)a before subtracting g (measured on this robot)')
    p.add_argument('--static-bias', action='store_true',
                   help='subtract the mean static acceleration from the first 3 s')
    a = p.parse_args()
    if not 1. <= a.seconds <= 600. or not 50. <= a.rate <= 1000.:
        raise ValueError('Bounded duration and rate required')
    if os.environ.get('SM_MOTOR_OUTPUT') not in (None, '', '0'):
        raise SystemExit('This shadow process must never run with motor output enabled')

    import rclpy
    from rclpy.node import Node
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import Imu
    from scipy.spatial.transform import Rotation

    a.out.mkdir(parents=True, exist_ok=False)
    frames = (a.out / 'frames.jsonl').open('x')

    class Source(Node):
        def __init__(self):
            super().__init__('supermimic_shadow_imu_integration')
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE)
            self.latest = None
            self.subs = [self.create_subscription(Imu, '/aima/hal/imu/torso/state', self.on_imu, qos)]

        def on_imu(self, msg):
            q, acc = msg.orientation, msg.linear_acceleration
            self.latest = {'recv': time.monotonic(),
                           'stamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                           'q': [q.w, q.x, q.y, q.z],
                           'a': [acc.x, acc.y, acc.z]}

    rclpy.init()
    node = Source()
    sign = -1. if a.gravity_sign == 'neg' else 1.

    # Optional static offset: only a mean subtraction over a still window.
    bias = np.zeros(3)
    if a.static_bias:
        samples = []
        end = time.monotonic() + 3.
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=.01)
            if node.latest is not None:
                world = sign * Rotation.from_quat(node.latest['q']).apply(np.array(node.latest['a']))
                samples.append(world - np.array([0., 0., 9.81]))
        if len(samples) < 100:
            raise SystemExit('Static bias window produced too few samples')
        bias = np.array(samples).mean(axis=0)
        print('static bias %s' % np.round(bias, 4), flush=True)

    position = np.zeros(3)
    velocity = np.zeros(3)
    tick = 1. / a.rate
    started = last = time.monotonic()
    counts = {'ticks': 0, 'imu_age_over_50ms': 0}
    age_max = 0.
    try:
        while time.monotonic() - started < a.seconds:
            rclpy.spin_once(node, timeout_sec=.001)
            now = time.monotonic()
            if now - last < tick or node.latest is None:
                continue
            dt = now - last
            sample = node.latest
            age = now - sample['recv']
            age_max = max(age_max, age)
            if age > .05:
                counts['imu_age_over_50ms'] += 1
                last = now
                continue
            acc_world = sign * Rotation.from_quat(sample['q']).apply(np.array(sample['a'])) \
                - np.array([0., 0., 9.81]) - bias
            velocity = velocity + acc_world * dt
            position = position + velocity * dt
            counts['ticks'] += 1
            frames.write(json.dumps({'t': now - started, 'dt': dt, 'imu_age': age,
                                     'position': position.tolist(), 'velocity': velocity.tolist(),
                                     'acc_world': acc_world.tolist()}, allow_nan=False) + '\n')
            last = now
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        frames.flush()
        frames.close()
        node.destroy_node()
        rclpy.try_shutdown()
        summary = {'seconds': time.monotonic() - started, 'rate_hz': a.rate,
                   'gravity_sign': a.gravity_sign, 'static_bias': bias.tolist(),
                   'counters': counts, 'imu_age_max_s': age_max,
                   'final_position': position.tolist(), 'final_velocity': velocity.tolist(),
                   'control_ready': False, 'motor_output_enabled': False,
                   'algorithm': 'pure IMU strapdown, no filter, no contact logic',
                   'limitations': ('No bias/scale/extrinsic model beyond an optional mean '
                                   'subtraction; no absolute reference; drift grows without bound.')}
        (a.out / 'summary.json').write_text(json.dumps(summary, allow_nan=False) + '\n')
        print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
