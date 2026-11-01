"""ROS integration probe for isolated test topics; no robot commands."""
import argparse
import json
import os
from pathlib import Path
import time

from supermimic.deploy.x2_command_groups import JOINT_GROUPS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',type=Path,required=True);p.add_argument('--prefix',required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if os.environ.get('ROS_DOMAIN_ID')!='197' or not a.prefix.startswith('/supermimic/isolated_'):
        raise ValueError('Requires isolated test domain and topics')
    if a.out.exists():raise ValueError('Output exists')
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
    from std_msgs.msg import String
    from aimdk_msgs.msg import JointCommandArray
    names=json.loads(a.profile.read_text())['joint_names'];rclpy.init()
    node=Node('supermimic_isolated_command_probe',enable_rosout=False,start_parameter_services=False)
    qos=QoSProfile(depth=10,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.VOLATILE)
    pub=node.create_publisher(String,a.prefix+'/target',qos);rows={g:[] for g in JOINT_GROUPS}
    subs=[]
    for group in JOINT_GROUPS:
        def receive(msg,g=group):
            rows[g].append((time.monotonic(),[(j.name,j.position,j.stiffness,j.damping) for j in msg.joints]))
        subs.append(node.create_subscription(JointCommandArray,a.prefix+'/hal/'+group+'/command',receive,qos))
    try:
        deadline=time.monotonic()+8
        while time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.02)
            if node.count_subscribers(a.prefix+'/target')==1 and all(
                    node.count_publishers(a.prefix+'/hal/'+g+'/command')==1 for g in JOINT_GROUPS):break
        else:raise ValueError('DDS discovery not complete')
        last_sent=None
        for i in range(4):
            now=time.monotonic();msg=String();msg.data=json.dumps({'names':names,'sequence':i,
                'generated':now,'state_sampled':now,'position':[.1+j*.001 for j in range(31)],
                'velocity':[0.]*31,'effort':[0.]*31,'stiffness':[20.]*31,'damping':[2.]*31})
            pub.publish(msg);last_sent=now;end=time.monotonic()+.025
            while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.002)
        end=time.monotonic()+.4
        while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.005)
        latency={}
        for g,order in JOINT_GROUPS.items():
            active=[row for _,row in rows[g] if row and all(j[2]==20. for j in row)]
            fallback=[(t,row) for t,row in rows[g] if row and all(j[2]==0. and j[3]==3. for j in row)]
            if not active or not fallback:raise ValueError('Missing active or timeout fallback: '+g)
            if [j[0] for j in active[0]]!=list(order):raise ValueError('Wrong joint order: '+g)
            if any(abs(j[1]-(.1+names.index(j[0])*.001))>1e-12 for j in active[0]):
                raise ValueError('Wrong position mapping: '+g)
            latency[g]=fallback[0][0]-last_sent
            if not 0<=latency[g]<=.25:raise ValueError('Fallback not timely: '+g)
        result={'scope':__doc__,'domain':197,'prefix':a.prefix,
                'counts':{g:len(v) for g,v in rows.items()},'all_groups_active_then_fallback':True,
                'fallback_after_last_sent_s':latency,
                'hardware_output':False}
        a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    finally:node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
