"""Convert a sensor-only X2 bag for isolated, exploratory FAST-LIO replay.

Point time scale is explicit: the X2 driver's time semantics still require
independent verification. This script never connects to the live robot.
"""
from __future__ import annotations

import argparse
from array import array
import hashlib
import json
from pathlib import Path

import numpy as np

LIDAR = '/aima/hal/sensor/lidar_chest_front/lidar_pointcloud'
IMU = '/aima/hal/sensor/lidar_chest_front/imu'
CLOUD_OUT = '/supermimic/x2_lio_r1/cloud'
IMU_OUT = '/supermimic/x2_lio_r1/imu'


def lidar_pose_in_imu(translation, xyzw):
    """Invert TF's lidar->imu transform to FAST-LIO's IMU<-lidar convention."""
    q = np.asarray(xyzw, dtype=float)
    if not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 1e-3:
        raise ValueError('Invalid static TF quaternion')
    x, y, z, w = q / np.linalg.norm(q)
    r = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    return (-r.T @ np.asarray(translation)).tolist(), r.T.reshape(-1).tolist()


def convert_cloud(msg, time_scale):
    from sensor_msgs.msg import PointCloud2, PointField

    kinds = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}
    fields = {f.name: f for f in msg.fields}
    required = ('x', 'y', 'z', 'intensity', 'ring', 'timestamp')
    if any(name not in fields or fields[name].count != 1 for name in required):
        raise ValueError('X2 cloud is missing required scalar fields')
    if len(msg.data) != msg.row_step * msg.height or msg.row_step < msg.point_step * msg.width:
        raise ValueError('Invalid cloud byte layout')
    data = {}
    for name in required:
        f = fields[name]
        dtype = np.dtype(('>' if msg.is_bigendian else '<') + kinds[f.datatype])
        data[name] = np.ndarray((msg.height, msg.width), dtype=dtype, buffer=msg.data,
                                offset=f.offset, strides=(msg.row_step, msg.point_step)).reshape(-1)
    if not all(np.isfinite(a).all() for a in data.values()):
        raise ValueError('Nonfinite cloud field')
    offsets = data['timestamp'].astype(np.float64) * time_scale
    if not len(offsets) or offsets.min() < 0 or not 0.05 <= offsets.max() <= .15:
        raise ValueError('Point time does not fit a 10 Hz scan under the supplied scale')
    if data['ring'].min() < 0 or data['ring'].max() >= 96:
        raise ValueError('Unexpected E1R ring values')
    # FAST-LIO uses the final point time as the scan end. Never infer timing
    # from spinning-lidar azimuth: E1R already supplies per-point offsets.
    order = np.argsort(offsets, kind='stable')
    dtype = np.dtype({'names': ['x', 'y', 'z', 'intensity', 'time', 'ring'],
                      'formats': ['<f4'] * 5 + ['<u2'],
                      'offsets': [0, 4, 8, 12, 16, 20], 'itemsize': 24})
    points = np.zeros(len(order), dtype=dtype)
    for name in ('x', 'y', 'z', 'intensity', 'ring'):
        points[name] = data[name][order]
    points['time'] = offsets[order]
    out = PointCloud2()
    out.header = msg.header
    out.height, out.width = 1, len(points)
    out.fields = [PointField(name=name, offset=dtype.fields[name][1],
                            datatype=PointField.UINT16 if name == 'ring' else PointField.FLOAT32,
                            count=1) for name in dtype.names]
    out.is_bigendian, out.is_dense = False, True
    out.point_step, out.row_step = 24, 24 * len(points)
    # ROS's generated setter validates a bytes sequence element by element.
    # A typed byte array takes its supported bulk-assignment path instead.
    out.data = array('B', points.tobytes())
    return out, [float(offsets.min()), float(offsets.max())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--point-time-scale', type=float, required=True,
                        help='explicit candidate conversion to seconds; e.g. 1e-9 for ns')
    parser.add_argument('--timing-audit', type=Path,
                        help='packet audit directory; binds this bag and selects paired clouds')
    parser.add_argument('--cloud-stamp-mode', choices=['host', 'raw_sensor'], default='host')
    args = parser.parse_args()
    if not np.isfinite(args.point_time_scale) or args.point_time_scale <= 0:
        parser.error('point time scale must be positive and finite')
    if args.cloud_stamp_mode != 'host' and args.timing_audit is None:
        parser.error('raw sensor cloud stamps require a matching packet timing audit')
    raw_anchors = None
    if args.timing_audit:
        audit = json.loads((args.timing_audit / 'report.json').read_text())
        if (not audit['paired_raw_cloud_frames'] or audit['mismatched_frames'] or
                audit['exact_point_time_and_ring_matches'] != audit['paired_raw_cloud_frames'] or
                audit.get('point_time_unit') != 'nanosecond' or args.point_time_scale != 1e-9):
            raise ValueError('Packet audit does not verify the point time interpretation')
        hashes = {}
        for path in sorted(args.bag.glob('*.db3')):
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(block)
            hashes[path.name] = digest.hexdigest()
        if not hashes or hashes != audit.get('source_db_sha256'):
            raise ValueError('Packet audit belongs to a different bag')
        raw_anchors = {row['cloud_header_ns']: row['sensor_scan_base_ns']
                       for row in json.loads((args.timing_audit / 'frames.json').read_text())
                       if not row.get('unpaired')}
    args.out.mkdir(parents=True, exist_ok=False)
    import rosbag2_py
    import yaml
    from rclpy.serialization import deserialize_message, serialize_message
    from sensor_msgs.msg import Imu, PointCloud2
    from tf2_msgs.msg import TFMessage

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(args.bag), storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(args.out / 'bag'), storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    for name, typ in [(CLOUD_OUT, 'sensor_msgs/msg/PointCloud2'), (IMU_OUT, 'sensor_msgs/msg/Imu')]:
        writer.create_topic(rosbag2_py.TopicMetadata(name=name, type=typ, serialization_format='cdr'))
    counts = {'cloud': 0, 'imu': 0}
    times = []
    imu_bounds = []
    frames = set()
    extrinsic = None
    skipped_unpaired_clouds = 0
    while reader.has_next():
        topic, raw, received = reader.read_next()
        if topic == LIDAR:
            msg = deserialize_message(raw, PointCloud2)
            if msg.header.frame_id != 'lidar_chest_front':
                raise ValueError('Unexpected lidar frame')
            host_stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            if raw_anchors is not None and host_stamp not in raw_anchors:
                skipped_unpaired_clouds += 1
                continue
            msg, interval = convert_cloud(msg, args.point_time_scale)
            if args.cloud_stamp_mode == 'raw_sensor':
                msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(raw_anchors[host_stamp], 10**9)
            times.append(interval)
            writer.write(CLOUD_OUT, serialize_message(msg), received)
            counts['cloud'] += 1
        elif topic == IMU:
            msg = deserialize_message(raw, Imu)
            if msg.header.frame_id != 'lidar_imu_chest_front':
                raise ValueError('Unexpected IMU frame')
            values = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
                      msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite IMU values')
            imu_bounds.append(float(np.linalg.norm(values[:3])))
            writer.write(IMU_OUT, raw, received)
            counts['imu'] += 1
        elif topic == '/tf_static':
            for tf in deserialize_message(raw, TFMessage).transforms:
                frames.add((tf.header.frame_id, tf.child_frame_id))
                if (tf.header.frame_id, tf.child_frame_id) == ('lidar_chest_front', 'lidar_imu_chest_front'):
                    t, q = tf.transform.translation, tf.transform.rotation
                    extrinsic = lidar_pose_in_imu([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])
    del writer
    if not all(counts.values()) or extrinsic is None:
        raise ValueError('Capture lacks lidar, IMU or their static extrinsic')
    params = {
        'feature_extract_enable': False, 'point_filter_num': 2, 'max_iteration': 3,
        'filter_size_surf': .3, 'filter_size_map': .3, 'cube_side_length': 100.,
        'runtime_pos_log_enable': False,
        'common': {'lid_topic': CLOUD_OUT, 'imu_topic': IMU_OUT, 'time_sync_en': False,
                   'time_offset_lidar_to_imu': 0.},
        'preprocess': {'lidar_type': 2, 'scan_line': 96, 'scan_rate': 10,
                       'timestamp_unit': 0, 'blind': .3},
        'mapping': {'acc_cov': .1, 'gyr_cov': .1, 'b_acc_cov': .0001, 'b_gyr_cov': .0001,
                    'fov_degree': 120., 'det_range': 30., 'extrinsic_est_en': False,
                    'extrinsic_T': extrinsic[0], 'extrinsic_R': extrinsic[1]},
        'publish': {'path_en': False, 'scan_publish_en': False, 'dense_publish_en': False,
                    'scan_bodyframe_pub_en': False},
        'pcd_save': {'pcd_save_en': False, 'interval': -1},
    }
    (args.out / 'fastlio.yaml').write_text(yaml.safe_dump({'/**': {'ros__parameters': params}}))
    report = {'source_bag': str(args.bag), 'counts': counts,
              'point_offset_seconds_minmax': [np.min(times).item(), np.max(times).item()],
              'imu_acceleration_norm_minmax': [min(imu_bounds), max(imu_bounds)],
              'static_frame_pairs': sorted(frames), 'extrinsic_T': extrinsic[0],
              'extrinsic_R': extrinsic[1], 'candidate_point_time_scale': args.point_time_scale,
              'point_time_semantics_verified': raw_anchors is not None,
              'cloud_stamp_mode': args.cloud_stamp_mode,
              'timing_audit': str(args.timing_audit) if args.timing_audit else None,
              'skipped_unpaired_clouds': skipped_unpaired_clouds,
              'imu_stamp_mode': 'native_host_header_unchanged',
              'time_semantics_verified': False, 'motion_accuracy_validated': False,
              'scope': 'offline exploratory configuration; no prior map or leg odometry input; '
                       'pose is IMU frame, not pelvis; no robot control'}
    (args.out / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
