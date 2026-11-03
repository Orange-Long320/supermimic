"""Read-only joint-state recorder for the preparation run.

Subscribes to the four HAL joint-state topics only. It never touches the
control target or command topics, so the worker's exclusivity check is intact.
"""
import argparse
import json
import time

import numpy as np


def summarize_rows(rows, names, target, termination):
    if not rows:
        return {'samples': 0, 'termination': termination, 'note': 'no joint state received'}
    array = np.asarray([row['q'] for row in rows])
    times = np.asarray([row['t'] for row in rows])
    error = np.abs(array - target)
    reached = np.flatnonzero(error.max(axis=1) <= .05)
    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError('Joint recording timestamps must increase')
    rate = float(np.max(np.abs(np.diff(array, axis=0)) / dt[:, None])) if len(dt) else None
    return dict(
        samples=len(rows), seconds=float(times[-1] - times[0]), termination=termination,
        first_q=array[0].tolist(), last_q=array[-1].tolist(),
        last_max_abs_error_to_startup=float(error[-1].max()),
        last_worst_joint=names[int(error[-1].argmax())],
        max_abs_error_over_run=float(error.max()),
        first_time_within_0p05_s=float(times[reached[0]] - times[0]) if len(reached) else None,
        max_abs_rate_rad_s=rate,
        per_joint_max_travel=[float(v) for v in np.abs(array - array[0]).max(axis=0)],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds', type=float, required=True)
    p.add_argument('--startup', required=True, help='JSON file with joint_names and target vector')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from aimdk_msgs.msg import JointStateArray

    with open(a.startup) as handle:
        startup = json.load(handle)
    names = startup['joint_names']
    target = np.asarray(startup['position'])

    rclpy.init()
    node = Node('supermimic_joint_recorder_readonly', enable_rosout=False,
                start_parameter_services=False)
    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                     durability=DurabilityPolicy.VOLATILE)
    latest = {}
    rows = []
    last_saved = [0.0]

    def callback(msg):
        for joint in msg.joints:
            latest[joint.name] = joint.position
        now = time.monotonic()
        if len(latest) < 31 or now - last_saved[0] < 0.02:
            return
        last_saved[0] = now
        rows.append(dict(t=now, q=[latest[name] for name in names]))

    for group in ('leg', 'waist', 'head', 'arm'):
        node.create_subscription(JointStateArray, f'/aima/hal/joint/{group}/state', callback, qos)
    execu = SingleThreadedExecutor()
    execu.add_node(node)
    started = time.monotonic()
    termination = 'deadline'
    try:
        while time.monotonic() - started < a.seconds:
            execu.spin_once(timeout_sec=.02)
    except BaseException as exc:
        termination = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        # ROS shutdown may interrupt spin before the deadline. Persist the
        # already collected data even on that path; never overwrite an old run.
        try:
            summary = summarize_rows(rows, names, target, termination)
            with open(a.out, 'x') as handle:
                json.dump(dict(summary=summary, rows=rows), handle, allow_nan=False)
            print(json.dumps({k: v for k, v in summary.items()
                              if k not in ('first_q', 'last_q', 'per_joint_max_travel')}, indent=2))
        finally:
            node.destroy_node()
            rclpy.try_shutdown()


if __name__ == '__main__':
    main()
