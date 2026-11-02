"""Read-only joint-state recorder for the preparation run.

Subscribes to the four HAL joint-state topics only. It never touches the
control target or command topics, so the worker's exclusivity check is intact.
"""
import argparse
import json
import time

import numpy as np


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

    startup = json.load(open(a.startup))
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
    while time.monotonic() - started < a.seconds:
        execu.spin_once(timeout_sec=.02)
    if not rows:
        print(json.dumps({'samples': 0, 'note': 'no joint state received'}))
        node.destroy_node()
        rclpy.try_shutdown()
        return
    array = np.asarray([row['q'] for row in rows])
    times = np.asarray([row['t'] for row in rows])
    error = np.abs(array - target)
    reached = np.flatnonzero(error.max(axis=1) <= .05)
    summary = dict(
        samples=len(rows), seconds=float(times[-1] - times[0]),
        first_q=array[0].tolist(), last_q=array[-1].tolist(),
        last_max_abs_error_to_startup=float(error[-1].max()),
        last_worst_joint=names[int(error[-1].argmax())],
        max_abs_error_over_run=float(error.max()),
        first_time_within_0p05_s=float(times[reached[0]] - times[0]) if len(reached) else None,
        max_abs_rate_rad_s=float(np.abs(np.diff(array, axis=0)).max() / .02),
        per_joint_max_travel=[float(v) for v in np.abs(array - array[0]).max(axis=0)],
    )
    with open(a.out, 'w') as handle:
        json.dump(dict(summary=summary, rows=rows), handle)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ('first_q', 'last_q', 'per_joint_max_travel')}, indent=2))
    print('largest travel:',
          [(names[i], round(float(v), 3)) for i, v in
           sorted(enumerate(summary['per_joint_max_travel']), key=lambda x: -x[1])[:5]])
    print('final worst joints:',
          [(names[i], round(float(v), 4)) for i, v in
           sorted(enumerate(error[-1]), key=lambda x: -x[1])[:5]])
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
