"""Bounded PC2 sensor/readiness report. No actuator publishers or control clients.

Run inside the project's lxh container. This does not initialize a pelvis
estimator, alter joint zeroes, prepare a pose, or transfer robot control.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('policy', 'startup-candidate', 'release-manifest', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=6.)
    args = parser.parse_args()
    if args.out.exists() or not 1 <= args.seconds <= 30:
        raise ValueError('New output and a bounded observation duration required')
    policy_hash = hashlib.sha256(args.policy.read_bytes()).hexdigest()
    if policy_hash != '4424f231fc1896da098b2fca366e8522df6791596638b2b9257ba682520bbe4c':
        raise ValueError('Expected selected 11500 model')
    candidate = json.loads(args.startup_candidate.read_text())
    release = json.loads(args.release_manifest.read_text())
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from aimdk_msgs.msg import JointStateArray
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu

    class Observer(Node):
        def __init__(self):
            super().__init__('supermimic_pc2_readiness')
            self.groups, self.counts, self.odom, self.imu = {}, {}, {}, None
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE)
            self.subs = [self.create_subscription(JointStateArray, '/aima/hal/joint/'+g+'/state',
                         lambda m, g=g: self.joints(m, g), qos) for g in ('leg', 'waist', 'head', 'arm')]
            self.topics = ('/supermimic/pelvis/odometry', '/aima/mc/leg_odometry',
                           '/pnc/estimate_odom', '/slam/localization/odometry')
            self.subs += [self.create_subscription(Odometry, t, lambda m, t=t: self.pose(m, t), qos)
                          for t in self.topics]
            self.subs.append(self.create_subscription(Imu, '/aima/hal/imu/torso/state', self.on_imu, qos))

        def joints(self, msg, group):
            self.counts[group] = self.counts.get(group, 0)+1
            self.groups[group] = {'received': time.monotonic(),
                'header_age_at_receive_s': time.time()-msg.header.stamp.sec-msg.header.stamp.nanosec*1e-9,
                'joints': [{'name': j.name, 'position': j.position, 'velocity': j.velocity,
                            'error_code': j.error_code} for j in msg.joints]}

        def pose(self, msg, topic):
            self.counts[topic] = self.counts.get(topic, 0)+1
            self.odom[topic] = {'frame_id': msg.header.frame_id, 'child_frame_id': msg.child_frame_id,
                               'received': time.monotonic()}

        def on_imu(self, msg):
            self.counts['imu'] = self.counts.get('imu', 0)+1
            self.imu = {'received': time.monotonic(), 'frame_id': msg.header.frame_id}

    rclpy.init(); node = Observer(); started = time.monotonic()
    try:
        while time.monotonic()-started < args.seconds:
            rclpy.spin_once(node, timeout_sec=.02)
        now = time.monotonic()
        measured = [j for g in node.groups.values() for j in g['joints']]
        names = candidate['joint_names']; by_name = {j['name']: j for j in measured}
        matched = len(by_name) == len(measured) == len(names) and set(by_name) == set(names)
        outside, errors, faults = {}, {}, []
        if matched:
            for i, name in enumerate(names):
                row = by_name[name]; q = row['position']
                errors[name] = q-candidate['startup']['position'][i]
                lo, hi = candidate['measured_lower_candidate'][i], candidate['measured_upper_candidate'][i]
                if not lo <= q <= hi:
                    outside[name] = {'measured_rad': q, 'file_intersection_rad': [lo, hi]}
                if row['error_code']:
                    faults.append(name)
        topics = {t: {'publishers': node.count_publishers(t), 'messages': node.counts.get(t, 0),
                       'last': node.odom.get(t)} for t in node.topics}
        report = {'scope': __doc__, 'selected_checkpoint': 11500, 'policy_sha256': policy_hash,
                  'release_source_snapshot': release['source_snapshot'],
                  'user_deployment_approval': release['user_deployment_approval'],
                  'duration_s': now-started, 'counts': node.counts, 'joint_name_set_matched': matched,
                  'joint_faults': faults, 'measured_outside_file_intersection': outside,
                  'startup_position_error_rad': errors,
                  'group_receive_age_s': {g: now-r['received'] for g, r in node.groups.items()},
                  'group_header_age_at_receive_s': {g: r['header_age_at_receive_s'] for g, r in node.groups.items()},
                  'imu': node.imu, 'odometry_topics': topics,
                  'motor_output_enabled': False, 'control_transferred': False,
                  'blocking_conditions': ['Live pelvis pose is not validated',
                      'Startup motion/handover has not passed dynamics validation',
                      'Encoder calibration, hardware command contract and independent watchdog remain unvalidated']}
        if not matched: report['blocking_conditions'].append('Missing or duplicate joint states')
        if outside: report['blocking_conditions'].append('Measured joints disagree with model/device file envelope')
        if faults: report['blocking_conditions'].append('Joint error codes present')
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(report, indent=2, allow_nan=False))
    finally:
        node.destroy_node(); rclpy.try_shutdown()


if __name__ == '__main__':
    main()
