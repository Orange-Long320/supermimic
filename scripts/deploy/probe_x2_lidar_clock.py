"""Ten-second passive E1R DIFOP clock/PTP check; sparse snapshots only.

Does not recover or invent full-rate IMU timestamps, or modify driver settings.
"""
from collections import Counter
import argparse
import json
from pathlib import Path
import struct
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('New output required')
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from aimdk_msgs.msg import LidarRawData
    from sensor_msgs.msg import Imu, PointCloud2
    rclpy.init()
    node = Node('supermimic_readonly_lidar_clock', enable_rosout=False, start_parameter_services=False)
    samples = []
    invalid = Counter()
    imu = []
    cloud_headers = set()

    def receive_imu(msg):
        a, w = msg.linear_acceleration, msg.angular_velocity
        imu.append((msg.header.stamp.sec*10**9+msg.header.stamp.nanosec, time.time_ns(),
                    struct.pack('>6f', a.x,a.y,a.z,w.x,w.y,w.z).hex()))

    def receive(msg):
        now = time.time_ns()
        raw = bytes(msg.difop_data)
        if len(raw) != 256 or raw[:8] != bytes.fromhex('a5ff005a11115555'):
            invalid['unsupported DIFOP format'] += 1
            return
        micros = int.from_bytes(raw[106:110], 'big')
        if micros >= 1_000_000:
            invalid['invalid fractional seconds'] += 1
            return
        sensor_ns = (int.from_bytes(raw[100:106], 'big')*1_000_000+micros)*1000
        msop = bytes(msg.msop_data[:20])
        cloud_ns = msg.header.meas_stamp.sec*10**9+msg.header.meas_stamp.nanosec
        msop_ns = ((int.from_bytes(msop[10:16], 'big')*1_000_000+int.from_bytes(msop[16:20], 'big'))*1000
                   if len(msop)==20 and msop[:4]==bytes.fromhex('55aa5aa5') else None)
        samples.append(dict(received_ns=now, sensor_ns=sensor_ns,
                            ptp_mode=raw[98], ptp_status=raw[99], phy=raw[110], payload_hex=raw[180:204].hex(),
                            native_cloud_stamp_ns=cloud_ns, first_msop_sensor_ns=msop_ns))

    node.create_subscription(LidarRawData, '/aima/hal/sensor/lidar_chest_front/lidar_raw_data', receive,
        QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE))
    node.create_subscription(Imu, '/aima/hal/sensor/lidar_chest_front/imu', receive_imu,
        QoSProfile(depth=400, reliability=ReliabilityPolicy.RELIABLE))
    node.create_subscription(PointCloud2, '/aima/hal/sensor/lidar_chest_front/lidar_pointcloud',
        lambda msg: cloud_headers.add(msg.header.stamp.sec*10**9+msg.header.stamp.nanosec),
        QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
    start = time.monotonic()
    try:
        while time.monotonic()-start < 10.:
            rclpy.spin_once(node, timeout_sec=.02)
    finally:
        modes = Counter((r['ptp_mode'], r['ptp_status'], r['phy']) for r in samples)
        ages = [(r['received_ns']-r['sensor_ns'])/1e6 for r in samples]
        offsets = []
        cloud_offsets = []
        for r in samples:
            candidates = [v for v in imu if v[2] == r['payload_hex'] and abs(v[1]-r['received_ns']) < 300_000_000]
            if candidates:
                match = min(candidates, key=lambda v: abs(v[1]-r['received_ns']))
                r['matched_imu_header_ns'] = match[0]
                offsets.append((match[0]-r['sensor_ns'])/1e6)
            if r['first_msop_sensor_ns'] is not None and r['native_cloud_stamp_ns'] in cloud_headers:
                cloud_offsets.append((r['native_cloud_stamp_ns']-r['first_msop_sensor_ns'])/1e6)
        report = dict(scope=__doc__, seconds=time.monotonic()-start, samples=samples,
            invalid=dict(invalid), mode_status_phy_counts=[dict(mode=k[0], status=k[1], phy=k[2], count=v)
                                                        for k,v in modes.items()],
            received_minus_sensor_ms_min_max=[min(ages), max(ages)] if ages else None,
            matched_snapshot_count=len(offsets), imu_header_minus_sensor_ms_min_max=[min(offsets),max(offsets)] if offsets else None,
            matched_cloud_count=len(cloud_offsets), cloud_header_minus_first_msop_ms_min_max=
                [min(cloud_offsets),max(cloud_offsets)] if cloud_offsets else None,
            motor_commands_sent=False, driver_configuration_changed=False)
        a.out.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k != 'samples'}, indent=2))
        node.destroy_node()
        rclpy.try_shutdown()
    if not samples or invalid:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
