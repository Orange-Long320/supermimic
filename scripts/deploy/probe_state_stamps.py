"""Read-only diagnostic: classify why the worker rejects policy state packets.

Subscribes to the state producer topic with a matching BEST_EFFORT QoS and, for
every packet, compares the five device timestamps and the four wall-clock
receive times against the consumer's own ``time.time()``. The worker's gate is

    all(0 <= now - t <= max_age)

so both a stale stamp and a *future* stamp reject the packet. This probe counts
each category separately instead of guessing.
"""
import argparse
import json
import statistics
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


def pct(values, q):
    if not values:
        return float('nan')
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topic', default='/supermimic/x2_online_lio/policy_state')
    ap.add_argument('--seconds', type=float, default=20.0)
    ap.add_argument('--max-age', type=float, default=0.05)
    ap.add_argument('--max-skew', type=float, default=0.03)
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    rclpy.init()
    node = Node('state_stamp_probe')
    qos = QoSProfile(depth=20)
    qos.reliability = ReliabilityPolicy.BEST_EFFORT
    qos.history = HistoryPolicy.KEEP_LAST

    ages_state, ages_joint, ages_pub, ages_recv, skews = [], [], [], [], []
    gaps, fails = [], []
    counts = {'msg': 0, 'future_state': 0, 'future_joint': 0,
              'future_published': 0, 'future_receive': 0,
              'stale_state': 0, 'stale_joint': 0, 'skew': 0,
              'would_reject': 0, 'parse': 0}
    last_recv = {'t': None}

    def on_msg(msg):
        now = time.time()
        counts['msg'] += 1
        if last_recv['t'] is not None:
            gaps.append(now - last_recv['t'])
        last_recv['t'] = now
        try:
            p = json.loads(msg.data)
        except Exception:
            counts['parse'] += 1
            return
        state = p['state_stamp']
        joints = list(p['joint_stamps'])
        pub = p['published_wall']
        recv = list(p['joint_received_wall'])
        ages_state.append(now - state)
        ages_pub.append(now - pub)
        ages_joint.extend(now - t for t in joints)
        ages_recv.extend(now - t for t in recv)
        skew = max([state, *joints]) - min([state, *joints])
        skews.append(skew)

        reasons = []
        if state > now:
            counts['future_state'] += 1
            reasons.append('future_state')
        if any(t > now for t in joints):
            counts['future_joint'] += 1
            reasons.append('future_joint')
        if pub > now:
            counts['future_published'] += 1
            reasons.append('future_published')
        if any(t > now for t in recv):
            counts['future_receive'] += 1
            reasons.append('future_receive')
        if now - state > a.max_age:
            counts['stale_state'] += 1
            reasons.append('stale_state')
        if any(now - t > a.max_age for t in joints):
            counts['stale_joint'] += 1
            reasons.append('stale_joint')
        if skew > a.max_skew:
            counts['skew'] += 1
            reasons.append('skew')
        if reasons:
            counts['would_reject'] += 1
            fails.append({'t': now, 'reasons': reasons, 'age_state': now - state,
                          'age_joints': [now - t for t in joints],
                          'age_pub': now - pub, 'age_recv': [now - t for t in recv],
                          'skew': skew})

    from std_msgs.msg import String
    node.create_subscription(String, a.topic, on_msg, qos)

    deadline = time.time() + a.seconds
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

    def report(name, values):
        if not values:
            return {'count': 0}
        return {'count': len(values),
                'p50_ms': 1e3 * pct(values, .5),
                'p95_ms': 1e3 * pct(values, .95),
                'p99_ms': 1e3 * pct(values, .99),
                'min_ms': 1e3 * min(values), 'max_ms': 1e3 * max(values)}

    summary = {
        'counts': counts,
        'age_state': report('age_state', ages_state),
        'age_joint': report('age_joint', ages_joint),
        'age_published': report('age_published', ages_pub),
        'age_receive': report('age_receive', ages_recv),
        'skew': report('skew', skews),
        'gap': report('gap', gaps),
        'fails_sample': fails[:12],
    }
    text = json.dumps(summary, indent=1, allow_nan=True)
    print(text, flush=True)
    if a.out:
        with open(a.out, 'w') as fh:
            json.dump(summary, fh, indent=1, allow_nan=True)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
