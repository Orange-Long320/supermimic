"""Isolated-domain X2 command transport/watchdog prototype; NEVER robot topics.

This program deliberately requires a dedicated DDS domain and publishes only
under a unique /supermimic/isolated_* prefix, not /aima. A live watchdog cannot
protect against its own process, host, transport, power or HAL failure.
"""
import argparse
import json
import os
from pathlib import Path
import re
import time

from supermimic.deploy.supervision import TargetLeaseSupervisor
from supermimic.deploy.x2_command_groups import group_command


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--prefix',required=True)
    p.add_argument('--fallback-damping',type=float,required=True)
    p.add_argument('--max-state-age',type=float,default=.05)
    p.add_argument('--max-target-age',type=float,default=.05)
    p.add_argument('--hz',type=float,default=100.)
    a=p.parse_args()
    if not re.fullmatch('/supermimic/isolated_[a-z0-9_]+',a.prefix):raise ValueError('Isolated prefix required')
    if os.environ.get('ROS_DOMAIN_ID')!='197':raise ValueError('Dedicated test DDS domain 197 required')
    if not 20<=a.hz<=200:raise ValueError('Invalid prototype frequency')
    names=json.loads(a.profile.read_text())['joint_names']
    dummy={'names':names,**{k:[0.]*len(names) for k in ('position','velocity','effort','stiffness','damping')}}
    group_command(dummy)
    supervisor=TargetLeaseSupervisor(names,[a.fallback_damping]*len(names),
                                     max_state_age=a.max_state_age,max_target_age=a.max_target_age)
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
    from std_msgs.msg import String
    from aimdk_msgs.msg import JointCommand,JointCommandArray
    rclpy.init();node=Node('supermimic_isolated_command_publisher',enable_rosout=False,start_parameter_services=False)
    qos=QoSProfile(depth=10,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.VOLATILE)
    pubs={group:node.create_publisher(JointCommandArray,a.prefix+'/hal/'+group+'/command',qos)
          for group in ('leg','waist','arm','head')}
    def receive(msg):
        try:
            packet=json.loads(msg.data)
            if not isinstance(packet,dict):raise ValueError('Target must be a JSON object')
            supervisor.accept(packet)
        except (TypeError,ValueError) as exc:supervisor.fault_reason='Malformed target: '+str(exc)
    node.create_subscription(String,a.prefix+'/target',receive,qos)
    last_fault=None
    def tick():
        nonlocal last_fault
        packet=supervisor.tick(time.monotonic())
        if packet is None:return
        grouped=group_command(packet)
        for group,values in grouped.items():
            msg=JointCommandArray()
            for row in values:
                joint=JointCommand();joint.name=row['name']
                for field in ('position','velocity','effort','stiffness','damping'):setattr(joint,field,float(row[field]))
                msg.joints.append(joint)
            pubs[group].publish(msg)
        if packet['fallback'] and packet['reason']!=last_fault:
            last_fault=packet['reason'];print('isolated watchdog fallback: '+str(last_fault),flush=True)
    node.create_timer(1./a.hz,tick)
    try:rclpy.spin(node)
    finally:node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
