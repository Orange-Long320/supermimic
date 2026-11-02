"""Read-only trace of /supermimic/x2/control_target during the trial.

Subscribes only; no publisher, no control client. Records phase transitions,
sequence continuity and the largest target-to-measured deviation.
"""
import argparse
import json
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds', type=float, required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from std_msgs.msg import String

    rclpy.init()
    node = Node('supermimic_target_trace_readonly', enable_rosout=False, start_parameter_services=False)
    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.VOLATILE)
    topic = '/supermimic/x2/control_target'
    rows, transitions = [], []
    phase = [None]
    max_dev = [0.0]
    seq_bad = [0]

    def receive(msg):
        try:
            packet = json.loads(msg.data)
        except Exception:
            return
        now = time.time()
        position = np.asarray(packet['position'])
        measured = np.asarray(packet['measured_position'])
        dev = float(np.max(np.abs(position - measured)))
        max_dev[0] = max(max_dev[0], dev)
        rows.append(dict(wall=now, sequence=packet.get('sequence'), phase=packet.get('phase'),
                         target_measured_dev=dev))
        if packet.get('phase') != phase[0]:
            phase[0] = packet.get('phase')
            transitions.append(dict(wall=now, phase=phase[0], max_target_measured_dev=dev,
                                    stiffness_max=float(np.max(packet['stiffness'])),
                                    stiffness_min=float(np.min(packet['stiffness']))))

    node.create_subscription(String, topic, receive, qos)
    execu = SingleThreadedExecutor()
    execu.add_node(node)
    started = time.monotonic()
    while time.monotonic() - started < a.seconds:
        execu.spin_once(timeout_sec=.02)
    requested = [r for r in rows if r['sequence'] is not None]
    summary = dict(topic_seconds=a.seconds, messages=len(rows),
                   first_wall=rows[0]['wall'] if rows else None,
                   last_wall=rows[-1]['wall'] if rows else None,
                   phases_in_order=[t['phase'] for t in transitions],
                   transitions=transitions,
                   sequence_monotonic=all(b['sequence'] > c['sequence']
                                          for c, b in zip(requested, requested[1:])),
                   max_target_measured_dev=max_dev[0])
    with open(a.out, 'w') as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != 'transitions'}, indent=2))
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
