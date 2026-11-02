"""Capture one coherent, passive joint/IMU snapshot; never publishes commands.

The IMU frame and raw orientation are retained. No pelvis pose, calibration,
contact force or support condition is inferred from this snapshot.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('New output required')
    names = json.loads(a.candidate.read_text())['joint_names']
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from rclpy.serialization import deserialize_message
    from aimdk_msgs.msg import JointStateArray
    from sensor_msgs.msg import Imu
    rclpy.init()
    node = Node('supermimic_entry_snapshot', enable_rosout=False, start_parameter_services=False)
    groups, imu = {}, {}
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    def joint(raw, group):
        msg = deserialize_message(raw, JointStateArray)
        groups[group] = dict(header_ns=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec,
            received_ns=time.time_ns(), domain=int(msg.state.value),
            joints=[dict(name=j.name, position=j.position, velocity=j.velocity, error_code=j.error_code)
                    for j in msg.joints])
    def inertial(msg):
        w,v,q = msg.angular_velocity, msg.linear_acceleration, msg.orientation
        imu.update(header_ns=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec,
            received_ns=time.time_ns(), frame_id=msg.header.frame_id,
            angular_velocity=[w.x,w.y,w.z], linear_acceleration=[v.x,v.y,v.z],
            orientation_wxyz=[q.w,q.x,q.y,q.z])
    subscriptions = [node.create_subscription(JointStateArray, f'/aima/hal/joint/{g}/state',
                     lambda raw,g=g:joint(raw,g), qos, raw=True) for g in ('leg','waist','head','arm')]
    subscriptions.append(node.create_subscription(Imu, '/aima/hal/imu/torso/state', inertial, qos))
    started = time.monotonic()
    try:
        while time.monotonic()-started < 10:
            rclpy.spin_once(node, timeout_sec=.02)
            if len(groups)!=4 or not imu:
                continue
            stamps=[r['header_ns'] for r in groups.values()]+[imu['header_ns']]
            now=time.time_ns()
            joints=[j for row in groups.values() for j in row['joints']]
            by_name={j['name']:j for j in joints}
            if (max(stamps)-min(stamps)>30_000_000 or not all(0<=now-s<=50_000_000 for s in stamps)
                    or len(joints)!=len(names) or set(by_name)!=set(names)
                    or any(j['error_code'] for j in joints)):
                continue
            result=dict(scope=__doc__, captured_wall_ns=now,
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                candidate_sha256=hashlib.sha256(a.candidate.read_bytes()).hexdigest(),
                names=names, q=[by_name[n]['position'] for n in names],
                qd=[by_name[n]['velocity'] for n in names], groups=groups, imu=imu,
                max_header_skew_ns=max(stamps)-min(stamps), motor_commands_sent=0,
                pelvis_pose_validated=False, physical_calibration_validated=False)
            a.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
            print(json.dumps(dict(captured=True,joints=len(names),max_header_skew_ns=result['max_header_skew_ns'])))
            return
        raise RuntimeError('No coherent fresh complete snapshot within 10 seconds')
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
