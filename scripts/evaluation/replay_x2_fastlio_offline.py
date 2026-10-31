"""Replay prepared sensors into a child FAST-LIO process on localhost only."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rate', type=float, default=.5)
    parser.add_argument('--reliable-inputs', action='store_true',
                        help='use the instrumented binary with lossless finite-bag input queues')
    args = parser.parse_args()
    if os.environ.get('ROS_LOCALHOST_ONLY') != '1' or not os.environ.get('ROS_DOMAIN_ID'):
        parser.error('An explicit isolated ROS_DOMAIN_ID and ROS_LOCALHOST_ONLY=1 are required')
    if not 0 < args.rate <= 1:
        parser.error('--rate must be in (0, 1]')
    args.out.mkdir(parents=True, exist_ok=False)
    import rclpy
    import rosbag2_py
    from rclpy.node import Node
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Imu, PointCloud2
    from nav_msgs.msg import Odometry
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

    cloud_topic, imu_topic = '/supermimic/x2_lio_r1/cloud', '/supermimic/x2_lio_r1/imu'
    odom_topic = '/supermimic/x2_lio_r1/odometry'
    rclpy.init()
    node = Node('supermimic_offline_lio_replay', enable_rosout=False,
                start_parameter_services=False)
    qos = QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.VOLATILE)
    pub_qos = (QoSProfile(history=HistoryPolicy.KEEP_ALL, reliability=ReliabilityPolicy.RELIABLE,
                          durability=DurabilityPolicy.VOLATILE) if args.reliable_inputs else qos)
    pubs = {cloud_topic: node.create_publisher(PointCloud2, cloud_topic, pub_qos),
            imu_topic: node.create_publisher(Imu, imu_topic, pub_qos)}
    types = {cloud_topic: PointCloud2, imu_topic: Imu}
    trajectory = []
    availability = []
    replay_clock = {'origin_ns': None, 'start': None}

    def observe(msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        trajectory.append([msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                           p.x, p.y, p.z, q.x, q.y, q.z, q.w])
        if replay_clock['origin_ns'] is None:
            raise RuntimeError('Odometry arrived before replay input')
        availability.append(int(replay_clock['origin_ns'] +
            (time.monotonic()-replay_clock['start']) * args.rate * 1e9))

    odom_sub = node.create_subscription(Odometry, odom_topic, observe, qos)
    del odom_sub
    command = [str(args.binary.resolve()), '--ros-args', '--params-file',
               str((args.prepared / 'fastlio.yaml').resolve()),
               '-r', '/Odometry:=' + odom_topic, '-r', '/tf:=/supermimic/x2_lio_r1/tf',
               '-r', '__node:=supermimic_fastlio_offline']
    if args.reliable_inputs:
        command += ['-p', 'sm_offline_reliable_inputs:=true']
    (args.out / 'command.json').write_text(json.dumps({'argv': command,
        'ROS_DOMAIN_ID': os.environ['ROS_DOMAIN_ID'], 'ROS_LOCALHOST_ONLY': '1'}, indent=2))
    log = (args.out / 'fastlio.log').open('w')
    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    error = None
    sent = {topic: 0 for topic in pubs}
    last_cloud_end = None
    try:
        deadline = time.monotonic() + 15
        while not all(p.get_subscription_count() for p in pubs.values()):
            if proc.poll() is not None:
                raise RuntimeError('FAST-LIO exited before subscriptions were ready')
            if time.monotonic() >= deadline:
                raise TimeoutError('FAST-LIO input subscriptions did not appear')
            rclpy.spin_once(node, timeout_sec=.05)
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=str(args.prepared / 'bag'), storage_id='sqlite3'),
                    rosbag2_py.ConverterOptions('', ''))
        first_received = None
        start = time.monotonic()
        while reader.has_next():
            topic, raw, received = reader.read_next()
            if topic not in pubs:
                raise ValueError('Prepared bag contains a topic outside the sensor whitelist')
            if first_received is None:
                first_received = received
                replay_clock.update(origin_ns=received, start=start)
            due = start + (received - first_received) * 1e-9 / args.rate
            while time.monotonic() < due:
                if proc.poll() is not None:
                    raise RuntimeError('FAST-LIO exited during replay')
                rclpy.spin_once(node, timeout_sec=min(.01, max(0, due-time.monotonic())))
            msg = deserialize_message(raw, types[topic])
            if topic == cloud_topic:
                fields = {f.name: f for f in msg.fields}
                offsets = np.ndarray(msg.width * msg.height, dtype='<f4', buffer=msg.data,
                                     offset=fields['time'].offset, strides=(msg.point_step,))
                last_cloud_end = (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                                  + float(np.max(offsets)))
            pubs[topic].publish(msg)
            sent[topic] += 1
            rclpy.spin_once(node, timeout_sec=0)
        drain = time.monotonic() + 15
        while time.monotonic() < drain:
            rclpy.spin_once(node, timeout_sec=.05)
            if trajectory and last_cloud_end is not None and trajectory[-1][0] >= last_cloud_end - .001:
                break
        if len(trajectory) < 2:
            raise RuntimeError('No usable FAST-LIO trajectory was received')
        if trajectory[-1][0] < last_cloud_end - .001:
            raise RuntimeError('FAST-LIO did not cover the final input scan')
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        log.close()
        node.destroy_node()
        rclpy.try_shutdown()
        np.savetxt(args.out / 'trajectory.csv', np.asarray(trajectory).reshape(-1, 8),
                   delimiter=',', header='stamp,x,y,z,qx,qy,qz,qw', comments='')
        (args.out / 'availability.json').write_text(json.dumps({
            'capture_clock_available_ns': availability,
            'meaning': 'offline replay callback time projected onto original bag receive clock; '
                       'includes this server replay computation, not measured PC2 online latency',
            'replay_rate': args.rate}, indent=2)+'\n')
        report = {'samples': len(trajectory), 'sent': sent, 'error': error,
                  'child_return_code': proc.returncode, 'replay_rate': args.rate,
                  'reliable_offline_inputs': args.reliable_inputs,
                  'last_input_cloud_end_stamp': last_cloud_end,
                  'time_semantics_verified': False, 'motion_accuracy_validated': False,
                  'scope': 'server localhost-only offline sensor replay; no prior map, '
                           'no leg odometry input, no live robot connection or motor outputs'}
        for line in (args.out / 'fastlio.log').read_text().splitlines():
            if line.startswith('SM_LIO_COUNTS '):
                counts = json.loads(line[len('SM_LIO_COUNTS '):])
                report['native_counts'] = counts
                report['native_cloud_callbacks_match_sent'] = counts['cloud_callbacks'] == sent[cloud_topic]
                report['native_imu_callbacks_match_sent'] = counts['imu_callbacks'] == sent[imu_topic]
                report['odom_received_matches_native_publications'] = counts['odom_publications'] == len(trajectory)
        report['input_delivery_complete'] = bool(report.get('native_cloud_callbacks_match_sent') and
                                                  report.get('native_imu_callbacks_match_sent'))
        if args.reliable_inputs and not report['input_delivery_complete'] and report['error'] is None:
            report['error'] = 'Offline input callback counts differ from published counts'
        if trajectory:
            a = np.asarray(trajectory)
            report.update({'finite': bool(np.isfinite(a).all()),
                'sensor_span_s': float(a[-1, 0]-a[0, 0]),
                'final_scan_lag_s': float(last_cloud_end-a[-1, 0]) if last_cloud_end is not None else None,
                'output_gap_max_s': float(np.max(np.diff(a[:, 0]))) if len(a)>1 else None,
                'net_translation_m': float(np.linalg.norm(a[-1, 1:4]-a[0, 1:4])),
                'max_translation_from_first_m': float(np.max(np.linalg.norm(a[:, 1:4]-a[0, 1:4], axis=1))),
                'quaternion_norm_minmax': [float(np.linalg.norm(a[:, 4:], axis=1).min()),
                                            float(np.linalg.norm(a[:, 4:], axis=1).max())]})
        (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    if args.reliable_inputs and report['error']:
        raise RuntimeError(report['error'])


if __name__ == '__main__':
    main()
