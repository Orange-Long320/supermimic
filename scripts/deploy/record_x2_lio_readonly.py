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
    parser.add_argument('--include-raw', action='store_true',
                        help='also capture existing raw packet topic for timestamp auditing')
    parser.add_argument('--include-robot-state', action='store_true',
                        help='passively capture named joints and torso/chest IMUs; export JSONL too')
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
    if args.include_raw:
        from aimdk_msgs.msg import LidarRawData
        topics['/aima/hal/sensor/lidar_chest_front/lidar_raw_data'] = (
            LidarRawData, 'aimdk_msgs/msg/LidarRawData')
    state_log = None
    if args.include_robot_state:
        from aimdk_msgs.msg import JointStateArray
        for group in ('leg', 'waist', 'head', 'arm'):
            topics[f'/aima/hal/joint/{group}/state'] = (JointStateArray, 'aimdk_msgs/msg/JointStateArray')
        for part in ('torso', 'chest'):
            topics[f'/aima/hal/imu/{part}/state'] = (Imu, 'sensor_msgs/msg/Imu')
        state_log = (args.out / 'robot_state.jsonl').open('x')
    for topic, (_, msg_type) in topics.items():
        writer.create_topic(rosbag2_py.TopicMetadata(
            name=topic, type=msg_type, serialization_format='cdr'))

    rclpy.init()
    node = Node('supermimic_lio_sensor_capture', enable_rosout=False,
                start_parameter_services=False)
    sensor_qos = QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT,
                            durability=DurabilityPolicy.VOLATILE)
    # These high-rate topics feed a latest-state snapshot, not an IMU integrator.
    # Do not let bag/JSON serialization accumulate a stale FIFO behind the robot.
    state_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
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
        if state_log is not None and ('/joint/' in topic or '/hal/imu/' in topic):
            row = {'topic': topic, 'received_ns': received_ns, 'header_ns': stamp,
                   'frame': msg.header.frame_id}
            if hasattr(msg, 'joints'):
                row['joints'] = [{'name': j.name, 'position': j.position, 'velocity': j.velocity,
                                  'effort': j.effort, 'error_code': j.error_code} for j in msg.joints]
                row['meas_ns'] = msg.header.meas_stamp.sec * 10**9 + msg.header.meas_stamp.nanosec
            else:
                q, w, a = msg.orientation, msg.angular_velocity, msg.linear_acceleration
                row.update(orientation_wxyz=[q.w, q.x, q.y, q.z], omega=[w.x, w.y, w.z],
                           acceleration=[a.x, a.y, a.z])
            state_log.write(json.dumps(row, allow_nan=False) + '\n')

    subscriptions = [node.create_subscription(
        msg_type, topic, lambda msg, t=topic: record(t, msg),
        static_qos if topic == '/tf_static' else
        state_qos if '/joint/' in topic or '/hal/imu/' in topic else sensor_qos)
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
        if state_log is not None:
            state_log.close()
        (args.out / 'capture.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    if not all(counts[t] for t in topics):
        raise RuntimeError('Capture is incomplete: inspect capture.json before use')


if __name__ == '__main__':
    main()
