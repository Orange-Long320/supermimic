"""Bounded read-only ROS odometry inventory; never commands a robot."""
import json
import time


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from nav_msgs.msg import Odometry

    class Observer(Node):
        def __init__(self):
            super().__init__('supermimic_readonly_odometry_inventory')
            self.topics = ('/aima/mc/leg_odometry', '/pnc/estimate_odom', '/slam/localization/odometry')
            self.records = {topic: [] for topic in self.topics}
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE)
            self.subs = [self.create_subscription(Odometry, topic,
                         lambda msg, t=topic: self.record(t, msg), qos) for topic in self.topics]

        def record(self, topic, msg):
            self.records[topic].append({'received':time.monotonic(), 'frame_id':msg.header.frame_id,
                                        'child_frame_id':msg.child_frame_id,
                                        'stamp_s':msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9})

    rclpy.init(); node = Observer(); start = time.monotonic()
    try:
        while time.monotonic()-start < 6.:
            rclpy.spin_once(node, timeout_sec=.1)
        summary = {}
        for topic, rows in node.records.items():
            times = [x['received'] for x in rows]
            summary[topic] = {'messages':len(rows),
                              'frame_ids':sorted(set(x['frame_id'] for x in rows)),
                              'child_frame_ids':sorted(set(x['child_frame_id'] for x in rows)),
                              'header_stamp_nonzero':all(x['stamp_s'] > 0 for x in rows) if rows else None,
                              'max_interval_s':max((b-a for a,b in zip(times,times[1:])), default=None)}
        print(json.dumps({'topics':summary, 'scope':'read-only inventory; not a validated pelvis estimate'}, indent=2))
    finally:
        node.destroy_node(); rclpy.try_shutdown()


if __name__ == '__main__':
    main()
