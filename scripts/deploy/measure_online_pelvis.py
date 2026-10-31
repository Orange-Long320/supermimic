"""Bounded independent subscription audit of the online diagnostic pelvis topic."""
import argparse
import json
from pathlib import Path
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=30.)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    if not 1 <= args.seconds <= 180: raise ValueError('Expected bounded audit duration')
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    rclpy.init(); node = Node('supermimic_pelvis_independent_audit', enable_rosout=False, start_parameter_services=False)
    rows, status = [], []
    qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
    def receive(msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        rows.append([time.monotonic(), time.time(), msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9,
                     p.x, p.y, p.z, q.x, q.y, q.z, q.w])
    def observe(msg): status.append({'received': time.monotonic(), 'status': json.loads(msg.data)})
    subs = [node.create_subscription(Odometry, '/supermimic/pelvis/odometry', receive, qos),
            node.create_subscription(String, '/supermimic/x2_online_lio/status', observe, qos)]
    started = time.monotonic()
    try:
        while time.monotonic()-started < args.seconds: rclpy.spin_once(node, timeout_sec=.03)
        a = np.asarray(rows)
        result = {'scope': 'live independent subscriber; timestamp freshness and static output change, not motion accuracy',
                  'duration_s': time.monotonic()-started, 'messages': len(rows),
                  'pose_publishers': node.count_publishers('/supermimic/pelvis/odometry'),
                  'status_messages': len(status), 'latest_status': status[-1]['status'] if status else None,
                  'motor_output_enabled': False}
        if len(rows) > 1:
            gap = np.diff(a[:, 0]); age = a[:, 1]-a[:, 2]
            displacement = np.linalg.norm(a[:, 3:6]-a[0, 3:6], axis=1)
            def stats(x): return dict(zip(('min', 'p50', 'p95', 'p99', 'max'), np.percentile(x, [0, 50, 95, 99, 100]).tolist()))
            result.update(received_rate_hz=(len(a)-1)/(a[-1, 0]-a[0, 0]),
                          receive_gap_s=stats(gap), acquisition_header_age_s=stats(age),
                          pose_height_m=stats(a[:, 5]), all_finite=bool(np.isfinite(a).all()),
                          quaternion_norm=stats(np.linalg.norm(a[:, 6:10], axis=1)),
                          header_nondecreasing=bool(np.all(np.diff(a[:, 2])>=0)),
                          distinct_acquisition_times=int(len(np.unique(a[:, 2]))),
                          max_displacement_from_first_m=float(displacement.max()),
                          net_displacement_from_first_m=float(displacement[-1]),
                          receive_gaps_over_60ms=int(np.sum(gap>.06)))
        (args.out/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        (args.out/'status.json').write_text(json.dumps(status, indent=2, allow_nan=False)+'\n')
        if rows: np.savetxt(args.out/'poses.csv', a, delimiter=',', header='receive_monotonic,receive_wall,acquisition_header,x,y,z,qx,qy,qz,qw', comments='')
        print(json.dumps({k: v for k, v in result.items() if k != 'latest_status'}, indent=2))
    finally:
        node.destroy_node(); rclpy.try_shutdown()


if __name__ == '__main__': main()
