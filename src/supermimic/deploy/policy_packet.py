"""Validate an atomic, diagnostic pelvis/joint packet from the online bridge."""
from math import isfinite, sqrt


def validate_policy_packet(packet, names, now_wall, *, max_age=.05, max_skew=.03):
    if packet['schema'] not in (2, 3) or tuple(packet['joint_names']) != tuple(names):
        raise ValueError('policy packet schema or joint order')
    if not isinstance(packet['session_id'], str) or not packet['session_id']:
        raise ValueError('policy packet session missing')
    if packet['motor_output_enabled'] is not False:
        raise ValueError('expected diagnostic producer')
    vectors = {'position': 3, 'quat_wxyz': 4, 'angular_velocity_body': 3,
               'joint_position': len(names), 'joint_velocity': len(names),
               'joint_stamps': 4, 'joint_received_wall': 4}
    for key, size in vectors.items():
        v = packet[key]
        if len(v) != size or not all(isfinite(x) for x in v):
            raise ValueError('invalid policy packet '+key)
    if abs(sqrt(sum(x*x for x in packet['quat_wxyz']))-1.) > .01:
        raise ValueError('policy packet quaternion')
    if len(packet['joint_domains']) != 4 or any(x != 0 for x in packet['joint_domains']):
        raise ValueError('policy packet joint domain')
    stamps = [packet['state_stamp'], *packet['joint_stamps']]
    receives = [packet['published_wall'], *packet['joint_received_wall']]
    if not all(isfinite(x) for x in [now_wall, *stamps, *receives]):
        raise ValueError('invalid policy packet timestamps')
    integer_stamps = [packet['state_stamp_ns'], *packet['joint_stamps_ns']]
    if len(integer_stamps) != 5 or any(type(t) is not int or t < 0 for t in integer_stamps):
        raise ValueError('invalid policy packet integer timestamps')
    if packet['schema'] == 3:
        support = packet['imu_support_stamps_ns']
        if (packet['state_time_kind'] != 'resampled_imu_grid' or len(support) != 2
                or any(type(t) is not int or t < 0 for t in support)
                or not support[0] <= packet['state_stamp_ns'] <= support[1]
                or support[1]-support[0] > 30_000_000
                or support[1]*1e-9 > packet['published_wall']):
            raise ValueError('invalid policy packet IMU interpolation support')
    if any(t > packet['state_stamp_ns'] for t in packet['joint_stamps_ns']):
        raise ValueError('policy packet future joint acquisition')
    canonical_stamps = [t*1e-9 for t in integer_stamps]
    if any(abs(a-b)>5e-7 for a,b in zip(stamps, canonical_stamps)):
        raise ValueError('policy packet timestamp representations disagree')
    stamps = canonical_stamps
    if any(not 0 <= now_wall-t <= max_age for t in [*stamps, *receives]):
        raise ValueError('policy packet stale or future')
    if max(stamps)-min(stamps) > max_skew:
        raise ValueError('policy packet acquisition skew')
    return packet
