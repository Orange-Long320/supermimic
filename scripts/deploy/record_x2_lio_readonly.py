"""Bounded sensor-only rosbag capture on X2 PC2; never publishes commands."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=20.)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 60:
        parser.error('--seconds must be between 1 and 60')
    args.out.mkdir(parents=True, exist_ok=False)

    import rclpy
    import rosbag2_py
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import Imu, PointCloud2
    from tf2_msgs.msg import TFMessage

    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(args.out / 'bag'), storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    topics = {
        '/aima/hal/sensor/lidar_chest_front/lidar_pointcloud':
            (PointCloud2, 'sensor_msgs/msg/PointCloud2'),
        '/aima/hal/sensor/lidar_chest_front/imu': (Imu, 'sensor_msgs/msg/Imu'),
        '/tf_static': (TFMessage, 'tf2_msgs/msg/TFMessage'),
    }
    for topic, (_, msg_type) in topics.items():
        writer.create_topic(rosbag2_py.TopicMetadata(
            name=topic, type=msg_type, serialization_format='cdr'))

    rclpy.init()
    node = Node('supermimic_lio_sensor_capture', enable_rosout=False,
                start_parameter_services=False)
    sensor_qos = QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT,
                            durability=DurabilityPolicy.VOLATILE)
    static_qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE,
                            durability=DurabilityPolicy.TRANSIENT_LOCAL)
    counts = {topic: 0 for topic in topics}
    bounds = {}

    def record(topic, msg):
        received_ns = time.time_ns()
        writer.write(topic, serialize_message(msg), received_ns)
        counts[topic] += 1
        if hasattr(msg, 'header'):
            stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            row = bounds.setdefault(topic, {'first_header_ns': stamp})
            row['last_header_ns'] = stamp

    subscriptions = [node.create_subscription(
        msg_type, topic, lambda msg, t=topic: record(t, msg),
        static_qos if topic == '/tf_static' else sensor_qos)
        for topic, (msg_type, _) in topics.items()]
    del subscriptions  # Node retains subscriptions; no publishers or service clients.
    start = time.monotonic()
    error = None
    try:
        while time.monotonic() - start < args.seconds:
            rclpy.spin_once(node, timeout_sec=.1)
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        report = {'wall_time': time.strftime('%Y-%m-%d %H:%M:%S %z'),
                  'duration_s': time.monotonic() - start, 'counts': counts,
                  'header_bounds': bounds, 'error': error,
                  'scope': 'sensor subscriptions and local bag writes only; '
                           'no motor commands, service calls, mode or application changes'}
        node.destroy_node()
        rclpy.try_shutdown()
        del writer
        (args.out / 'capture.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    if not all(counts[t] for t in topics):
        raise RuntimeError('Capture is incomplete: inspect capture.json before use')


if __name__ == '__main__':
    main()
