"""Bounded read-only atomic state audit, with separate training/device bounds.

Reports raw measurements without clipping or zero offsets. No command publisher,
power service or robot mode operation. A clean window is not hardware clearance.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from supermimic.deploy.policy_packet import validate_policy_packet


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=180.)
    a = p.parse_args()
    if not 3 <= a.seconds <= 600 or a.out.exists():
        raise ValueError('New output and bounded duration 3..600 seconds required')
    c = json.loads((a.release/'startup-candidate.json').read_text())
    names = c['joint_names']
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    rclpy.init()
    node = Node('supermimic_readonly_state_audit', enable_rosout=False, start_parameter_services=False)
    from std_msgs.msg import String
    count = 0
    valid = 0
    reject = Counter()
    violations = {key: Counter() for key in ('model', 'pc2_urdf')}
    ranges = {n: [float('inf'), float('-inf')] for n in names}
    last = None
    maxima = dict(source_gap_s=0., publication_gap_s=0., receipt_gap_s=0., age_s=0.)
    gaps = Counter()
    sessions = set()

    def receive(msg):
        nonlocal count, valid, last
        now = time.time()
        receipt = time.monotonic()
        count += 1
        row = None
        try:
            row = json.loads(msg.data)
            validate_policy_packet(row, names, now)
            valid += 1
        except (ValueError, KeyError, TypeError) as exc:
            reject[str(exc)] += 1
            # Still inspect timing of stale but structurally complete packets.
            if str(exc) != 'policy packet stale or future':
                return
        try:
            session = row['session_id']
            sessions.add(session)
            stamp = row['state_stamp_ns']
            publication = row['published_wall']
            maxima['age_s'] = max(maxima['age_s'], now-stamp*1e-9)
            if last is not None and last[0] == session:
                values = ((stamp-last[1])*1e-9, publication-last[2], receipt-last[3])
                for key, value in zip(('source_gap_s', 'publication_gap_s', 'receipt_gap_s'), values):
                    maxima[key] = max(maxima[key], value)
                    if value > .03:
                        gaps[key+' over 30ms'] += 1
                if values[0] <= 0:
                    gaps['nonincreasing source time'] += 1
            last = session, stamp, publication, receipt
            for i, (name, q) in enumerate(zip(names, row['joint_position'])):
                ranges[name][0] = min(ranges[name][0], q)
                ranges[name][1] = max(ranges[name][1], q)
                for key in violations:
                    if not c[key+'_lower'][i] <= q <= c[key+'_upper'][i]:
                        violations[key][name] += 1
        except (KeyError, TypeError, IndexError) as exc:
            reject['unreadable packet: '+str(exc)] += 1

    node.create_subscription(String, '/supermimic/x2_online_lio/policy_state', receive,
        QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
    start = time.monotonic()
    try:
        while time.monotonic()-start < a.seconds:
            rclpy.spin_once(node, timeout_sec=.02)
    finally:
        report = dict(scope=__doc__, seconds=time.monotonic()-start, received=count,
            valid=valid, rejected=dict(reject), sessions=sorted(sessions), maxima=maxima,
            gaps=dict(gaps), trailing_silence_s=None if last is None else time.monotonic()-last[3],
            measurements_rad={n: r for n, r in ranges.items() if r[0] != float('inf')},
            violation_frames={k: dict(v) for k, v in violations.items()},
            compared_bounds={k: dict(zip(names, zip(c[k+'_lower'], c[k+'_upper']))) for k in violations},
            motor_commands_sent=False, ready_for_hardware_trial=False)
        a.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(report, indent=2, allow_nan=False))
        node.destroy_node()
        rclpy.try_shutdown()
    if not valid:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
