"""Read-only AimDK 1.1 HAL topic inspector. Never publishes commands."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imu', choices=('torso', 'chest'), default='torso')
    args = parser.parse_args()
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from aimdk_msgs.msg import JointStateArray
    from sensor_msgs.msg import Imu

    class Observer(Node):
        def __init__(self):
            super().__init__('supermimic_readonly_observer')
            joint_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                                   durability=DurabilityPolicy.TRANSIENT_LOCAL)
            imu_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                                 durability=DurabilityPolicy.VOLATILE)
            self.seen = set()
            self.subs = []
            for group in ('arm', 'waist', 'leg', 'head'):
                self.subs.append(self.create_subscription(JointStateArray,
                    f'/aima/hal/joint/{group}/state',
                    lambda msg, group=group: self.joints(group, msg), joint_qos))
            self.subs.append(self.create_subscription(Imu, f'/aima/hal/imu/{args.imu}/state',
                                                      self.imu, imu_qos))

        def joints(self, group, msg):
            if group not in self.seen:
                self.seen.add(group)
                self.get_logger().info(f'{group}: {[(j.name, j.error_code) for j in msg.joints]}')

        def imu(self, msg):
            if 'imu' not in self.seen:
                self.seen.add('imu')
                self.get_logger().info(f'IMU frame_id={msg.header.frame_id!r}; verify mounting and orientation before use')

    rclpy.init()
    node = Observer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
