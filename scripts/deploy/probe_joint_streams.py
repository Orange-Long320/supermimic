"""Read-only: measure arrival rate and gaps of the five device streams.

The state producer publishes on the arrival of one joint group (default
``head``) and skips the cycle when the four group stamps disagree by more than
25 ms. A long publish gap therefore points at a joint stream hiccup, not at the
host. This probe reports each stream separately.

Usage:
    python3 probe_joint_streams.py --seconds 15
"""
import argparse
import json
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from aimdk_msgs.msg import JointStateArray
from sensor_msgs.msg import Imu

GROUPS = ('leg', 'waist', 'head', 'arm')


def stats(values):
    if not values:
        return {'count': 0}
    v = sorted(values)
    def q(p):
        return v[min(len(v) - 1, max(0, int(round(p * (len(v) - 1)))))]
    return {'count': len(v), 'rate_hz': None,
            'p50_ms': 1e3 * q(.5), 'p95_ms': 1e3 * q(.95),
            'p99_ms': 1e3 * q(.99), 'max_ms': 1e3 * v[-1],
            'over_25ms': sum(1 for x in v if x > .025),
            'over_100ms': sum(1 for x in v if x > .1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=15.)
    ap.add_argument('--imu-topic', default='/aima/hal/imu/torso/state')
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    rclpy.init()
    node = Node('joint_stream_probe')
    qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT,
                     durability=DurabilityPolicy.VOLATILE)

    names = list(GROUPS) + ['imu']
    rec = {n: {'last': None, 'gaps': [], 'ages': [], 'count': 0, 'first': None} for n in names}
    latest = {g: None for g in GROUPS}
    skews = []
    t0 = time.time()

    def handle(name, stamp_ns):
        now = time.time()
        r = rec[name]
        r['count'] += 1
        if r['first'] is None:
            r['first'] = now
        if r['last'] is not None:
            r['gaps'].append(now - r['last'])
        r['last'] = now
        if stamp_ns:
            r['ages'].append(now - stamp_ns * 1e-9)

    def make_joint(group):
        def cb(msg):
            stamp_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            latest[group] = stamp_ns
            handle(group, stamp_ns)
            if all(v is not None for v in latest.values()):
                values = list(latest.values())
                skews.append((max(values) - min(values)) * 1e-9)
        return cb

    def imu_cb(msg):
        handle('imu', msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec)

    subs = [node.create_subscription(Imu, a.imu_topic, imu_cb, qos)]
    for g in GROUPS:
        subs.append(node.create_subscription(
            JointStateArray, f'/aima/hal/joint/{g}/state', make_joint(g), qos))

    deadline = time.time() + a.seconds
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    elapsed = time.time() - t0

    report = {'seconds': elapsed}
    for n in names:
        r = rec[n]
        s = stats(r['gaps'])
        s['rate_hz'] = r['count'] / elapsed if elapsed else None
        age = stats(r['ages'])
        s['age_p50_ms'] = age.get('p50_ms')
        s['age_p99_ms'] = age.get('p99_ms')
        s['age_max_ms'] = age.get('max_ms')
        report[n] = s
    s = stats(skews)
    s['over_25ms'] = sum(1 for x in skews if x > .025)
    s['over_30ms'] = sum(1 for x in skews if x > .03)
    s['fraction_over_25ms'] = (s['over_25ms'] / len(skews)) if skews else None
    report['group_skew'] = s
    print(json.dumps(report, indent=1), flush=True)
    if a.out:
        with open(a.out, 'w') as fh:
            json.dump(report, fh, indent=1)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
