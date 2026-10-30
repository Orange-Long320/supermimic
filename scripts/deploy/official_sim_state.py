"""Forward official simulator ROS state to a local UDP consumer; never publish ROS."""
import argparse
import json
import os
import socket
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--sim-only', action='store_true', required=True)
    a = p.parse_args()
    if os.environ.get('ROS_LOCALHOST_ONLY') != '1' or os.environ.get('ROS_DOMAIN_ID') != '179':
        raise RuntimeError('Requires isolated simulation ROS domain 179 and ROS_LOCALHOST_ONLY=1')
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from aimdk_msgs.msg import JointStateArray
    from sensor_msgs.msg import Imu
    from nav_msgs.msg import Odometry

    class SimState(Node):
        def __init__(self):
            super().__init__('supermimic_sim_state')
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.groups, self.imu, self.odom = {}, None, None
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE)
            self.subs = [self.create_subscription(JointStateArray, f'/aima/hal/joint/{g}/state',
                         lambda msg, group=g: self.joint(group, msg), qos)
                         for g in ('leg', 'waist', 'head', 'arm')]
            self.subs += [self.create_subscription(Imu, '/aima/hal/imu/torso/state', self.on_imu, qos),
                          self.create_subscription(Odometry, '/aima/hal/odom/state', self.on_odom, qos)]
            self.timer = self.create_timer(.02, self.send)

        def joint(self, group, msg):
            self.groups[group] = {'time': time.monotonic(), 'domain': msg.state.value,
                                  'joints': [[j.name, j.position, j.velocity, j.error_code] for j in msg.joints]}

        def on_imu(self, msg):
            q, w = msg.orientation, msg.angular_velocity
            self.imu = {'time': time.monotonic(), 'q': [q.w, q.x, q.y, q.z], 'w': [w.x, w.y, w.z]}

        def on_odom(self, msg):
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            self.odom = {'time': time.monotonic(), 'position': [p.x, p.y, p.z], 'q': [q.w, q.x, q.y, q.z]}

        def send(self):
            if len(self.groups) == 4 and self.imu and self.odom:
                message = {'time': time.monotonic(), 'groups': self.groups, 'imu': self.imu, 'odom': self.odom}
                self.sock.sendto(json.dumps(message, allow_nan=False).encode(), ('127.0.0.1', a.port))

    rclpy.init()
    node = SimState()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        node.sock.close()
        rclpy.try_shutdown()


if __name__ == '__main__': main()
