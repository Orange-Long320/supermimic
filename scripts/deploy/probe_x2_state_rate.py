"""Read-only rate/gap probe for /supermimic/x2_online_lio/policy_state.

Subscribes only. No actuator publisher, no policy, no control client.
Reports delivery rate, inter-arrival gaps, per-packet age and field sanity.
"""
import argparse
import json
import time

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds', type=float, default=30.)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from std_msgs.msg import String

    rclpy.init()
    node = Node('supermimic_state_probe_readonly', enable_rosout=False, start_parameter_services=False)
    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                     durability=DurabilityPolicy.VOLATILE)
    topic = '/supermimic/x2_online_lio/policy_state'
    arrivals, ages, sessions, seqs, bad = [], [], set(), [], []
    stamps = []
    flags = set()
    count = [0]

    def receive(msg):
        count[0] += 1
        now = time.time()
        try:
            row = json.loads(msg.data)
            stamps.append(float(row['state_stamp']))
            flags.add(row.get('motor_output_enabled'))
            flags.add(row.get('eligible_for_control'))
            arrivals.append(time.monotonic())
            ages.append(now - float(row['state_stamp']))
            sessions.add(row.get('session_id'))
            if len(row.get('joint_position', [])) != 31 or len(row.get('joint_velocity', [])) != 31:
                bad.append('joint length')
            if len(row.get('quat_wxyz', [])) != 4 or len(row.get('angular_velocity_body', [])) != 3:
                bad.append('pose length')
        except Exception as exc:
            bad.append(repr(exc))

    node.create_subscription(String, topic, receive, qos)
    deadline = time.monotonic() + 3.
    while node.count_publishers(topic) != 1 and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=.05)
    publishers = node.count_publishers(topic)
    execu = SingleThreadedExecutor()
    execu.add_node(node)
    started = time.monotonic()
    while time.monotonic() - started < a.seconds:
        execu.spin_once(timeout_sec=.05)
    elapsed = time.monotonic() - started
    gaps = np.diff(arrivals).tolist() if len(arrivals) > 1 else []
    ds = np.diff(stamps).tolist() if len(stamps) > 1 else []
    result = dict(
        subscription_seconds=elapsed, publishers=publishers, messages=count[0],
        rate_hz=(len(arrivals) / elapsed) if elapsed > 0 else None,
        distinct_sessions=len(sessions), session_ids=[s for s in sessions if s],
        state_stamp_monotonic=(all(b > a_ for a_, b in zip(stamps, stamps[1:])) if len(stamps) > 1 else None),
        motor_output_enabled_flags=sorted({str(row_flag) for row_flag in flags}),
        arrival_gap_ms=(dict(zip(('p50', 'p95', 'p99', 'max'),
                                 np.percentile(gaps, [50, 95, 99, 100]).tolist())) if gaps else None),
        state_stamp_gap_ms=(dict(zip(('p50', 'p95', 'p99', 'max'),
                                     np.percentile(ds, [50, 95, 99, 100]).tolist())) if ds else None),
        gaps_over_30ms=sum(1 for g in gaps if g > .03),
        gaps_over_100ms=sum(1 for g in gaps if g > .1),
        age_ms=(dict(zip(('p50', 'p95', 'p99', 'max'),
                         np.percentile(ages, [50, 95, 99, 100]).tolist())) if ages else None),
        age_over_50ms=sum(1 for x in ages if x > .05),
        field_errors=bad[:10], field_error_count=len(bad))
    with open(a.out, 'w') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
