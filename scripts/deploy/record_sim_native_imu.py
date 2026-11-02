"""Record native-rate simulator IMU/odometry/joints to JSONL; never publish ROS.

Purpose: the deployed state chain keeps only the latest IMU message and forwards
it at the 50 Hz control tick, so the strapdown integrator never sees the
high-frequency content of foot-strike impacts. This recorder stores every native
sample with both the message header stamp and the local receive time, so the
same session can be integrated at native rate and at 50 Hz snapshots for a
direct comparison. Read-only: it subscribes and writes files only.
"""
import argparse
import json
import os
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, help='new output directory')
    p.add_argument('--seconds', type=float, required=True)
    p.add_argument('--ros-domain', type=int, default=179)
    p.add_argument('--sim-only', action='store_true', required=True)
    a = p.parse_args()
    if not 128 <= a.ros_domain <= 232 or os.environ.get('ROS_LOCALHOST_ONLY') != '1' \
            or os.environ.get('ROS_DOMAIN_ID') != str(a.ros_domain):
        raise RuntimeError('Requires the explicitly selected isolated simulation ROS domain and localhost only')
    if not 1. <= a.seconds <= 600.:
        raise ValueError('Bounded duration required')

    import rclpy
    from rclpy.node import Node
    from rclpy.executors import ExternalShutdownException
    from rclpy.impl.implementation_singleton import rclpy_implementation
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from aimdk_msgs.msg import JointStateArray
    from sensor_msgs.msg import Imu
    from nav_msgs.msg import Odometry

    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=False)
    imu_file = open(os.path.join(out, 'imu.jsonl'), 'x')
    chest_file = open(os.path.join(out, 'imu-chest.jsonl'), 'x')
    odom_file = open(os.path.join(out, 'odom.jsonl'), 'x')
    joint_file = open(os.path.join(out, 'joints.jsonl'), 'x')
    counts = {'imu': 0, 'chest': 0, 'odom': 0, 'joints': 0}

    def emit(handle, key, record):
        handle.write(json.dumps(record, allow_nan=False) + '\n')
        counts[key] += 1
        if counts[key] % 200 == 0:
            handle.flush()

    class Recorder(Node):
        def __init__(self):
            super().__init__('supermimic_native_imu_recorder')
            # Deep best-effort queue: a recorder that drops samples would bias
            # the very comparison it is collecting.
            qos = QoSProfile(depth=400, reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE)
            self.subs = [
                self.create_subscription(Imu, '/aima/hal/imu/torso/state', self.on_imu, qos),
                self.create_subscription(Imu, '/aima/hal/imu/chest/state', self.on_chest, qos),
                self.create_subscription(Odometry, '/aima/hal/odom/state', self.on_odom, qos),
            ]
            self.subs += [self.create_subscription(JointStateArray, f'/aima/hal/joint/{g}/state',
                                                   lambda msg, group=g: self.on_joint(group, msg), qos)
                          for g in ('leg', 'waist', 'head', 'arm')]

        def on_imu(self, msg):
            self.record_imu(msg, imu_file, 'imu')

        def on_chest(self, msg):
            self.record_imu(msg, chest_file, 'chest')

        def record_imu(self, msg, handle, key):
            q, w, acc = msg.orientation, msg.angular_velocity, msg.linear_acceleration
            emit(handle, key, {
                'stamp': [msg.header.stamp.sec, msg.header.stamp.nanosec],
                'recv': time.monotonic(),
                'frame_id': msg.header.frame_id,
                'q_xyzw': [q.x, q.y, q.z, q.w],
                'w': [w.x, w.y, w.z],
                'a': [acc.x, acc.y, acc.z],
            })

        def on_odom(self, msg):
            pose = msg.pose.pose
            p, q = pose.position, pose.orientation
            tw = msg.twist.twist
            emit(odom_file, 'odom', {
                'stamp': [msg.header.stamp.sec, msg.header.stamp.nanosec],
                'recv': time.monotonic(),
                'position': [p.x, p.y, p.z],
                'q_xyzw': [q.x, q.y, q.z, q.w],
                'linear': [tw.linear.x, tw.linear.y, tw.linear.z],
                'angular': [tw.angular.x, tw.angular.y, tw.angular.z],
                'frame_id': msg.header.frame_id, 'child_frame_id': msg.child_frame_id,
            })

        def on_joint(self, group, msg):
            emit(joint_file, 'joints', {
                'stamp': [msg.header.stamp.sec, msg.header.stamp.nanosec],
                'recv': time.monotonic(),
                'group': group,
                'joints': [[j.name, j.position, j.velocity, j.effort] for j in msg.joints],
            })

    rclpy.init()
    node = Recorder()
    started = time.monotonic()
    try:
        while time.monotonic() - started < a.seconds:
            rclpy.spin_once(node, timeout_sec=.2)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except rclpy_implementation.RCLError:
        if rclpy.ok():
            raise
    finally:
        for handle in (imu_file, chest_file, odom_file, joint_file):
            handle.flush()
            handle.close()
        node.destroy_node()
        rclpy.try_shutdown()
    summary = {'counts': counts, 'seconds': float(time.monotonic() - started)}
    with open(os.path.join(out, 'summary.json'), 'x') as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
