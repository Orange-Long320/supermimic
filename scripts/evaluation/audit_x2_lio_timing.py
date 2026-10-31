"""Audit recorded X2 E1R packet times against cloud fields and IMU headers.

Native driver reference: SHA256
834074532c6df0879fb5ca01ed863156f285fd54245a5284a80a210d0b53ff3e.
No live subscriptions, publishers, clock changes, or offset optimization.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


def parse_raw_cdr(data):
    """Decode the installed MessageHeader + LidarRawData schema, CDR1 only."""
    if data[:2] not in (b'\0\0', b'\0\1'):
        raise ValueError('Unsupported CDR encapsulation')
    endian = '<' if data[1] == 1 else '>'
    pos = 4

    def u32():
        nonlocal pos
        pos = (pos + 3) & ~3
        value = struct.unpack_from(endian + 'I', data, pos)[0]
        pos += 4
        return value

    def seq():
        nonlocal pos
        n = u32()
        if pos + n > len(data):
            raise ValueError('Truncated CDR sequence')
        value = data[pos:pos+n]
        pos += n
        return value

    stamp = u32() * 10**9 + u32()
    frame = seq()
    if not frame or frame[-1] != 0:
        raise ValueError('Invalid CDR string')
    sequence = u32()
    meas_stamp = u32() * 10**9 + u32()
    msop, difop = seq(), seq()
    if pos != len(data):
        raise ValueError('Unexpected bytes after LidarRawData')
    return {'stamp': stamp, 'meas_stamp': meas_stamp, 'sequence': sequence,
            'frame': frame[:-1].decode(), 'msop': msop, 'difop': difop}


def sensor_us(packet, start):
    return int.from_bytes(packet[start:start+6], 'big') * 10**6 + int.from_bytes(packet[start+6:start+10], 'big')


def reconstruct_points(msop):
    """Reproduce the installed E1R parser's point-time arithmetic exactly."""
    if not msop or len(msop) % 1200:
        raise ValueError('Unexpected MSOP packet size')
    packets = [msop[i:i+1200] for i in range(0, len(msop), 1200)]
    first = sensor_us(packets[0], 10)
    values = []
    seqs = []
    for packet in packets:
        if packet[:4] != bytes.fromhex('55aa5aa5'):
            raise ValueError('Unexpected MSOP magic')
        seqs.append(int.from_bytes(packet[4:6], 'big'))
        dt = sensor_us(packet, 10) - first
        dtype = np.dtype({'names': ['offset', 'distance'], 'formats': ['>u2', '>u2'],
                          'offsets': [0, 2], 'itemsize': 12})
        points = np.frombuffer(packet, dtype=dtype, count=96, offset=32)
        distance = (points['distance'].astype(np.float32) * np.float32(.005)).astype(np.float64)
        keep = (distance >= .1) & (distance <= 150.)
        # The native parser converts microsecond differences to float32 first,
        # multiplies by 1000 in float32, then stores a float64 nanosecond field.
        ns = (np.asarray(dt + points['offset'].astype(np.int64), dtype=np.float32)
              * np.float32(1000.)).astype(np.float64)
        values.extend(zip(ns[keep].astype(np.int64).tolist(), np.flatnonzero(keep).tolist()))
    return first * 1000, values, seqs


def stats(values):
    a = np.asarray(values, dtype=float)
    return {'count': len(a), 'min': float(a.min()), 'p50': float(np.percentile(a, 50)),
            'p95': float(np.percentile(a, 95)), 'p99': float(np.percentile(a, 99)),
            'max': float(a.max())} if len(a) else {'count': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Imu, PointCloud2
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(args.bag), storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    clouds = {}
    raws = []
    imus = []
    imu_values = collections.defaultdict(list)
    receive_ages = collections.defaultdict(list)
    stamps = collections.defaultdict(list)
    offset_bounds = []
    while reader.has_next():
        topic, data, received = reader.read_next()
        if topic.endswith('/lidar_raw_data'):
            raw = parse_raw_cdr(data)
            raw['received'] = received
            raws.append(raw)
        elif topic.endswith('/lidar_pointcloud'):
            msg = deserialize_message(data, PointCloud2)
            stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            fields = {f.name: f for f in msg.fields}
            times = np.ndarray((msg.height, msg.width), dtype='<f8', buffer=msg.data,
                               offset=fields['timestamp'].offset,
                               strides=(msg.row_step, msg.point_step)).reshape(-1).copy()
            ring = np.ndarray((msg.height, msg.width), dtype='<u2', buffer=msg.data,
                              offset=fields['ring'].offset,
                              strides=(msg.row_step, msg.point_step)).reshape(-1).copy()
            clouds[stamp] = (times, ring)
            stamps['cloud'].append(stamp)
            receive_ages['cloud'].append((received - stamp) / 1e6)
            offset_bounds.append([times.min(), times.max()])
        elif topic.endswith('/lidar_chest_front/imu'):
            msg = deserialize_message(data, Imu)
            stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            values = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
                      msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
            key = struct.pack('>6f', *values)
            imu_values[key].append(stamp)
            imus.append(stamp)
            stamps['imu'].append(stamp)
            receive_ages['imu'].append((received - stamp) / 1e6)
    comparisons = []
    cloud_clock_offsets = []
    imu_clock_offsets = []
    relative_offsets = []
    modes = collections.Counter()
    matched_imu = 0
    raw_difop_sizes = collections.Counter()
    for raw in raws:
        base_ns, expected, sequences = reconstruct_points(raw['msop'])
        stamp = raw['meas_stamp']
        cloud_offset = stamp - base_ns
        cloud_clock_offsets.append(cloud_offset / 1e6)
        row = {'raw_sequence': raw['sequence'], 'cloud_header_ns': stamp,
               'sensor_scan_base_ns': base_ns, 'first_packet_sequence': sequences[0],
               'last_packet_sequence': sequences[-1], 'packets': len(sequences)}
        if stamp in clouds:
            times, rings = clouds[stamp]
            actual = collections.Counter(zip(times.astype(np.int64).tolist(), rings.tolist()))
            exp = collections.Counter(expected)
            row.update({'cloud_points': len(times), 'reconstructed_points': len(expected),
                        'missing_from_cloud': sum((exp-actual).values()),
                        'unexplained_cloud_points': sum((actual-exp).values())})
        else:
            row['unpaired'] = True
        difop = raw['difop']
        raw_difop_sizes[len(difop)] += 1
        if len(difop) == 256 and difop[:8] == bytes.fromhex('a5ff005a11115555'):
            modes[(difop[98], difop[99], difop[110])] += 1
            # processNewVersionPacket passes (packet+180, packet+100)
            # as (IMU vector payload, sensor timestamp) to publishImuData.
            imu_sensor_ns = sensor_us(difop, 100) * 1000
            key = difop[180:204]
            candidates = imu_values.get(key, [])
            if candidates:
                observed = min(candidates, key=lambda t: abs(raw['stamp']-t))
                if abs(raw['stamp'] - observed) < 200_000_000:
                    matched_imu += 1
                    imu_offset = observed - imu_sensor_ns
                    imu_clock_offsets.append(imu_offset / 1e6)
                    relative_offsets.append((cloud_offset-imu_offset) / 1e6)
                    row.update({'matched_imu_header_ns': observed, 'imu_sensor_ns': imu_sensor_ns})
        comparisons.append(row)
    paired = [r for r in comparisons if not r.get('unpaired')]
    mismatched = [r for r in paired if r['missing_from_cloud'] or r['unexplained_cloud_points']]
    db_hashes = {}
    for path in sorted(args.bag.glob('*.db3')):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
        db_hashes[path.name] = digest.hexdigest()
    report = {'source_bag': str(args.bag.resolve()), 'source_db_sha256': db_hashes,
              'driver_sha256': '834074532c6df0879fb5ca01ed863156f285fd54245a5284a80a210d0b53ff3e',
              'point_time_unit': 'nanosecond',
              'point_time_origin': 'first_accepted_msop_packet_sensor_time',
              'raw_frames': len(raws), 'cloud_frames': len(clouds),
              'imu_samples': len(imus), 'paired_raw_cloud_frames': len(paired),
              'exact_point_time_and_ring_matches': len(paired)-len(mismatched),
              'mismatched_frames': mismatched[:5], 'matched_difop_imu_samples': matched_imu,
              'difop_sizes': dict(raw_difop_sizes),
              'ptp_mode_status_phy_counts': [{'mode': k[0], 'status': k[1], 'phy': k[2], 'count': v}
                                           for k, v in modes.items()],
              'point_field_ns_range': [float(np.min(offset_bounds)), float(np.max(offset_bounds))],
              'cloud_host_minus_sensor_ms': stats(cloud_clock_offsets),
              'matched_imu_host_minus_sensor_ms': stats(imu_clock_offsets),
              'cloud_minus_matched_imu_clock_offsets_ms': stats(relative_offsets),
              'receipt_minus_header_ms': {k: stats(v) for k, v in receive_ages.items()},
              'header_gaps_ms': {k: stats(np.diff(v)/1e6) for k, v in stamps.items()},
              'header_backwards': {k: int(np.sum(np.diff(v)<0)) for k, v in stamps.items()},
              'scope': 'packet/field time audit only; host stamp is not acquisition-time proof; '
                       'sparse DIFOP snapshots do not reconstruct every 200 Hz IMU sensor stamp'}
    (args.out/'frames.json').write_text(json.dumps(comparisons, indent=2)+'\n')
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if not paired or mismatched:
        raise RuntimeError('Point time reconstruction did not match every paired cloud')
    if not matched_imu:
        raise RuntimeError('No DIFOP IMU payload could be matched to a received IMU')


if __name__ == '__main__':
    main()
