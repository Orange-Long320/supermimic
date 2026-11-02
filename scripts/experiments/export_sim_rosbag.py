"""Offline export of simulator ROS bag into native IMU/odom/joint JSONL."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bag', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=a.bag, storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    types = {x.name: get_message(x.type) for x in reader.get_all_topics_and_types()}
    a.out.mkdir(parents=True, exist_ok=False)
    handles = {name: (a.out/(name+'.jsonl')).open('x') for name in ('imu', 'imu-chest', 'odom', 'joints')}
    counts = {name: 0 for name in handles}
    try:
        while reader.has_next():
            topic, raw, recv_ns = reader.read_next()
            msg = deserialize_message(raw, types[topic])
            row = {'stamp': [msg.header.stamp.sec, msg.header.stamp.nanosec], 'recv': recv_ns*1e-9}
            if topic in ('/aima/hal/imu/torso/state', '/aima/hal/imu/chest/state'):
                key = 'imu' if '/torso/' in topic else 'imu-chest'
                q, w, acc = msg.orientation, msg.angular_velocity, msg.linear_acceleration
                row.update(frame_id=msg.header.frame_id, q_xyzw=[q.x,q.y,q.z,q.w],
                           w=[w.x,w.y,w.z], a=[acc.x,acc.y,acc.z])
            elif topic == '/aima/hal/odom/state':
                key = 'odom'; pose, tw = msg.pose.pose, msg.twist.twist
                pos, q = pose.position, pose.orientation
                row.update(position=[pos.x,pos.y,pos.z],q_xyzw=[q.x,q.y,q.z,q.w],
                           linear=[tw.linear.x,tw.linear.y,tw.linear.z],
                           angular=[tw.angular.x,tw.angular.y,tw.angular.z],
                           frame_id=msg.header.frame_id,child_frame_id=msg.child_frame_id)
            elif topic.startswith('/aima/hal/joint/') and topic.endswith('/state'):
                key = 'joints'; group = topic.split('/')[-2]
                row.update(group=group,joints=[[j.name,j.position,j.velocity,j.effort] for j in msg.joints])
            else:
                continue
            handles[key].write(json.dumps(row,allow_nan=False)+'\n'); counts[key] += 1
    finally:
        for handle in handles.values():
            handle.close()
    (a.out/'summary.json').write_text(json.dumps({'counts':counts,'recv_clock':'rosbag receive timestamp, seconds'},indent=2)+'\n')
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
