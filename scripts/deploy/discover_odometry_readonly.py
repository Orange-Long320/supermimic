"""Observe discovered Odometry topics in a single bounded read-only ROS node."""
import json
import time


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from nav_msgs.msg import Odometry

    class Inspector(Node):
        def __init__(self):
            super().__init__('supermimic_readonly_discover_odom')
            self.qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                                  durability=DurabilityPolicy.VOLATILE)
            self.subs = {}; self.data = {}

        def discover(self):
            for topic, types in self.get_topic_names_and_types():
                if 'nav_msgs/msg/Odometry' not in types:
                    continue
                item = self.data.setdefault(topic, {'count':0, 'max_publishers':0, 'frames':set(), 'children':set()})
                item['max_publishers'] = max(item['max_publishers'], self.count_publishers(topic))
                if topic not in self.subs:
                    self.subs[topic] = self.create_subscription(Odometry, topic, lambda m,t=topic:self.record(t,m), self.qos)

        def record(self, topic, msg):
            row = self.data[topic]; row['count'] += 1
            row['frames'].add(msg.header.frame_id); row['children'].add(msg.child_frame_id)

    rclpy.init(); node = Inspector(); start = time.monotonic(); next_scan = start
    try:
        while time.monotonic()-start < 20.:
            rclpy.spin_once(node, timeout_sec=.1)
            if time.monotonic() >= next_scan:
                node.discover(); next_scan = time.monotonic()+2.
        for item in node.data.values():
            item['frames'] = sorted(item['frames']); item['children'] = sorted(item['children'])
        print(json.dumps({'topics':node.data, 'seconds':20,
                          'scope':'read-only nav_msgs/Odometry inventory; no received data is not proof a node is disabled'}, indent=2))
    finally:
        node.destroy_node(); rclpy.try_shutdown()


if __name__ == '__main__':
    main()
