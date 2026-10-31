"""Read-only, bounded inventory of potential map-free state sources."""
import json
import time


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu
    from tf2_msgs.msg import TFMessage
    from aimdk_msgs.msg import McCommonState

    class Inspector(Node):
        def __init__(self):
            super().__init__('supermimic_mapfree_readonly_inventory')
            self.items = {}
            self.topics = {'/aima/mc/common/state': McCommonState,
                           '/aima/hal/imu/torso/state': Imu,
                           '/aima/mc/leg_odometry': Odometry,
                           '/pnc/estimate_odom': Odometry,
                           '/ranger_base_node/odom': Odometry,
                           '/tf': TFMessage}
            self.qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                                  durability=DurabilityPolicy.VOLATILE)
            self.subscriptions_ = [self.create_subscription(msg, topic, lambda m, t=topic:self.record(t,m),self.qos)
                                   for topic,msg in self.topics.items()]
            self.publisher_snapshots = []

        def record(self, topic, msg):
            item = self.items.setdefault(topic, {'count':0, 'first':time.monotonic(), 'last':0., 'max_gap_s':0.})
            now = time.monotonic()
            if item['last']:
                item['max_gap_s'] = max(item['max_gap_s'], now-item['last'])
            item['last'] = now; item['count'] += 1
            if isinstance(msg, Odometry):
                item['frame_id'] = msg.header.frame_id; item['child_frame_id'] = msg.child_frame_id
                item['stamp_nonzero'] = (msg.header.stamp.sec != 0 or msg.header.stamp.nanosec != 0)
            elif isinstance(msg, Imu):
                item['frame_id'] = msg.header.frame_id
            elif isinstance(msg, TFMessage):
                pairs = item.setdefault('frame_pairs', set())
                pairs.update((x.header.frame_id,x.child_frame_id) for x in msg.transforms)
            elif isinstance(msg, McCommonState):
                item['message_fields'] = list(msg.get_fields_and_field_types())

    rclpy.init(); node = Inspector(); start = time.monotonic(); next_sample = start
    try:
        while time.monotonic()-start < 20.:
            rclpy.spin_once(node, timeout_sec=.1)
            if time.monotonic() >= next_sample:
                node.publisher_snapshots.append({t:node.count_publishers(t) for t in node.topics})
                next_sample = time.monotonic()+1.
        for item in node.items.values():
            item['observed_span_s'] = item.pop('last')-item.pop('first')
            if 'frame_pairs' in item:item['frame_pairs'] = sorted(item['frame_pairs'])
        print(json.dumps({'topics':node.items,'publisher_counts':node.publisher_snapshots,
                          'scope':'read-only discovery and topic inventory; not state-estimator validation'},indent=2))
    finally:
        node.destroy_node(); rclpy.try_shutdown()


if __name__ == '__main__':
    main()
