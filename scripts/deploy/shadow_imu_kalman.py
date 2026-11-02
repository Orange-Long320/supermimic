"""Read-only 200 Hz strapdown with a Kalman filter and zero-velocity updates.

Keeps the Kalman filter: constant-velocity propagation driven by measured
acceleration, with a world-frame acceleration-bias state. Drops everything else
from the earlier estimator: no forward kinematics, no contact Jacobian, no
support selection, no slip guard, no mount fitting.

The only measurement is a zero-velocity pseudo-measurement applied while the IMU
itself reports a still robot. State is horizontal position/velocity and bias, so
height is not estimated. Never publishes; control_ready is always false.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=30.)
    p.add_argument('--rate', type=float, default=200.)
    p.add_argument('--still-accel-tol', type=float, default=.5,
                   help='|norm(a) - g| allowed for the still test')
    p.add_argument('--still-gyro-tol', type=float, default=.08,
                   help='|omega| allowed for the still test')
    p.add_argument('--still-hold', type=float, default=.2,
                   help='seconds of continuously still IMU before a velocity update')
    p.add_argument('--timer', action='store_true',
                   help='integrate on a fixed-rate loop instead of on every IMU callback')
    p.add_argument('--init-bias-seconds', type=float, default=0.,
                   help='initialise the bias state from this many still seconds (0 = zero start)')
    p.add_argument('--velocity-sigma', type=float, default=.02,
                   help='standard deviation of the zero-velocity measurement')
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

    state = {'last': None, 'still_since': None, 'integrate': False}
    counts = {'ticks': 0, 'zupt': 0, 'imu_age_over_50ms': 0, 'dt_over_10ms': 0}
    timing = {'age_max': 0., 'dt_max': 0.}
    started = time.monotonic()

    class Source(Node):
        def __init__(self):
            super().__init__('supermimic_shadow_imu_kalman')
            qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE)
            self.latest = None
            self.subs = [self.create_subscription(Imu, '/aima/hal/imu/torso/state', self.on_imu, qos)]

        def on_imu(self, msg):
            q, w, acc = msg.orientation, msg.angular_velocity, msg.linear_acceleration
            self.latest = {'recv': time.monotonic(), 'q': [q.w, q.x, q.y, q.z],
                           'w': [w.x, w.y, w.z], 'a': [acc.x, acc.y, acc.z]}
            if not a.timer and state['integrate']:
                step(self.latest)

    rclpy.init()
    node = Source()
    x = np.zeros(6)                                  # px, py, vx, vy, bx, by
    P = np.diag([.01, .01, .1, .1, .1, .1])
    H = np.zeros((2, 6))
    H[0, 2] = H[1, 3] = 1.
    R = np.eye(2) * (a.velocity_sigma ** 2)
    Q = np.diag([1e-8, 1e-8, 4e-5, 4e-5, 1e-8, 1e-8])
    identity = np.eye(6)
    if a.init_bias_seconds > 0.:
        window = []
        end = time.monotonic() + a.init_bias_seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=.01)
            if node.latest is not None:
                window.append(-Rotation.from_quat(node.latest['q']).apply(np.array(node.latest['a']))[:2])
        if len(window) < 100:
            raise SystemExit('Bias initialisation window produced too few samples')
        x[4:6] = np.array(window).mean(axis=0)
        print('initial bias %s' % np.round(x[4:6], 4), flush=True)
    tick = 1. / a.rate
    def step(sample):
        nonlocal x, P
        now = time.monotonic()
        dt = now - state['last'] if state['last'] is not None else tick
        state['last'] = now
        age = now - sample['recv']
        timing['age_max'] = max(timing['age_max'], age)
        timing['dt_max'] = max(timing['dt_max'], dt)
        if age > .05:
            counts['imu_age_over_50ms'] += 1
            return
        if dt > .01:
            counts['dt_over_10ms'] += 1
        acc_body = np.array(sample['a'])
        gyro = np.array(sample['w'])
        world = -Rotation.from_quat(sample['q']).apply(acc_body)   # measured convention
        u = world[:2]
        F = np.eye(6)
        F[0, 2] = F[1, 3] = dt
        F[0, 4] = F[1, 5] = -.5 * dt ** 2
        F[2, 4] = F[3, 5] = -dt
        B = np.zeros((6, 2))
        B[0, 0] = B[1, 1] = .5 * dt ** 2
        B[2, 0] = B[3, 1] = dt
        x = F @ x + B @ u
        P = F @ P @ F.T + Q * (dt / .02)
        still = (abs(np.linalg.norm(acc_body) - 9.81) <= a.still_accel_tol
                 and np.linalg.norm(gyro) <= a.still_gyro_tol)
        state['still_since'] = (state['still_since'] if state['still_since'] is not None else now) if still else None
        if still and now - state['still_since'] >= a.still_hold:
            innovation = np.zeros(2) - H @ x
            S = H @ P @ H.T + R
            K = np.linalg.solve(S, (P @ H.T).T).T
            x = x + K @ innovation
            P = (identity - K @ H) @ P @ (identity - K @ H).T + K @ R @ K.T
            counts['zupt'] += 1
        counts['ticks'] += 1
        frames.write(json.dumps({'t': now - started, 'dt': dt, 'imu_age': age,
                                 'still': bool(still), 'acc_world': (world - [0., 0., 9.81]).tolist(),
                                 'position': [float(x[0]), float(x[1])],
                                 'velocity': [float(x[2]), float(x[3])],
                                 'bias': [float(x[4]), float(x[5])]}, allow_nan=False) + '\n')

    state['integrate'] = True
    try:
        while time.monotonic() - started < a.seconds:
            rclpy.spin_once(node, timeout_sec=.001)
            if a.timer:
                now = time.monotonic()
                if node.latest is None or now - (state['last'] or started) < tick:
                    continue
                step(node.latest)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        frames.flush()
        frames.close()
        node.destroy_node()
        rclpy.try_shutdown()
        elapsed = time.monotonic() - started
        summary = {'seconds': elapsed,
                   'achieved_rate_hz': counts['ticks'] / elapsed if elapsed > 0 else None,
                   'timer_rate_hz': a.rate,
                   'counters': counts, 'imu_age_max_s': timing['age_max'],
                   'max_tick_interval_s': timing['dt_max'], 'timer_mode': a.timer,
                   'still_test': {'accel_tol': a.still_accel_tol, 'gyro_tol': a.still_gyro_tol,
                                  'hold_s': a.still_hold},
                   'velocity_sigma': a.velocity_sigma,
                   'init_bias_seconds': a.init_bias_seconds,
                   'final_position_xy': [float(x[0]), float(x[1])],
                   'final_velocity_xy': [float(x[2]), float(x[3])],
                   'final_bias_xy': [float(x[4]), float(x[5])],
                   'control_ready': False, 'motor_output_enabled': False,
                   'algorithm': 'callback-rate strapdown + Kalman with bias state + still-detected zero velocity updates',
                   'limitations': ('No attitude/extrinsic calibration beyond the measured gravity sign; '
                                   'zero-velocity updates only while the IMU itself looks still, so motion '
                                   'drift is uncorrected; height not estimated; no absolute reference.')}
        (a.out / 'summary.json').write_text(json.dumps(summary, allow_nan=False) + '\n')
        print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
