"""Read a finite bag into CPU-only shadow diagnostics; no ROS node is created."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bag',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Imu,PointCloud2
    from tf2_msgs.msg import TFMessage
    reader=rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(a.bag),storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('',''))
    imus=[]; clouds=[]; cloud_stamps=[]; transforms={}; ci=0
    while reader.has_next():
        topic,raw,received=reader.read_next()
        if topic.endswith('/lidar_chest_front/imu'):
            m=deserialize_message(raw,Imu); w=m.angular_velocity; acc=m.linear_acceleration
            imus.append([m.header.stamp.sec*10**9+m.header.stamp.nanosec,received,w.x,w.y,w.z,acc.x,acc.y,acc.z])
        elif topic.endswith('/lidar_pointcloud'):
            m=deserialize_message(raw,PointCloud2); ci+=1
            if ci%10: continue
            if m.height!=1 or m.is_bigendian: raise ValueError('Unexpected cloud layout')
            fs={f.name:f for f in m.fields}
            xyz=np.column_stack([np.ndarray(m.width,dtype='<f4',buffer=m.data,
                    offset=fs[k].offset,strides=(m.point_step,)) for k in ('x','y','z')])[::16].copy()
            clouds.append(xyz); cloud_stamps.extend([m.header.stamp.sec+m.header.stamp.nanosec*1e-9]*len(xyz))
        elif topic=='/tf_static':
            for t in deserialize_message(raw,TFMessage).transforms:
                v,q=t.transform.translation,t.transform.rotation
                transforms[t.header.frame_id+' -> '+t.child_frame_id]={'xyz':[v.x,v.y,v.z],'xyzw':[q.x,q.y,q.z,q.w]}
    np.savez_compressed(a.out/'sensors.npz',imu=np.asarray(imus),points=np.concatenate(clouds),
                        point_stamps=np.asarray(cloud_stamps))
    (a.out/'tf.json').write_text(json.dumps(transforms,indent=2)+'\n')


if __name__=='__main__': main()
