"""Standalone synthetic 50 Hz state source for isolated DDS testing only."""
import argparse
import json
import os
from pathlib import Path
import signal
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--release',type=Path,required=True)
    p.add_argument('--prefix',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if os.environ.get('ROS_DOMAIN_ID')!='197' or not a.prefix.startswith('/supermimic/isolated_'):
        raise ValueError('Requires isolated domain and topics')
    if a.out.exists():raise ValueError('Output already exists')
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
    from std_msgs.msg import String
    c=json.loads((a.release/'startup-candidate.json').read_text());names=c['joint_names'];n=len(names)
    q=c['startup']['position'];quat=c['startup']['root_quat_wxyz'];position=c['startup']['root_position']
    rclpy.init();node=Node('supermimic_synthetic_source',enable_rosout=False,start_parameter_services=False)
    qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.VOLATILE)
    publisher=node.create_publisher(String,a.prefix+'/policy_state',qos);stamps=[];running=True
    def stop(*_):
        nonlocal running
        running=False
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        deadline=time.monotonic()
        while running:
            delay=deadline-time.monotonic()
            if delay>0:time.sleep(delay)
            if not running:break
            t=time.time_ns();ts=t*1e-9;stamps.append(t)
            row={'schema':2,'session_id':'synthetic-only','motor_output_enabled':False,'joint_names':names,
                 'state_stamp_ns':t,'state_stamp':ts,'published_wall':ts,'joint_stamps_ns':[t]*4,
                 'joint_stamps':[ts]*4,'joint_received_wall':[ts]*4,'joint_domains':[0]*4,
                 'joint_position':q,'joint_velocity':[0.]*n,'quat_wxyz':quat,'position':position,'angular_velocity_body':[0.]*3}
            msg=String();msg.data=json.dumps(row);publisher.publish(msg)
            deadline+=.02
            if deadline<time.monotonic()-.02:deadline=time.monotonic()+.02
    finally:
        gaps=[(b-a)*1e-9 for a,b in zip(stamps,stamps[1:])]
        a.out.write_text(json.dumps({'scope':__doc__,'count':len(stamps),'max_gap_s':max(gaps,default=None),
            'over_30ms':sum(x>.03 for x in gaps),'published_stamps_ns':stamps},indent=2)+'\n')
        node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
