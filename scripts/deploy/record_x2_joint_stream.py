"""Continuous read-only X2 joint recording using a bounded asynchronous queue."""
import argparse
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--startup',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('Recording output already exists')
    names=json.loads(a.startup.read_text())['joint_names']
    if len(names)!=31 or len(set(names))!=31:raise ValueError('Expected 31 named joints')
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
    from rclpy.impl.implementation_singleton import rclpy_implementation
    from aimdk_msgs.msg import JointStateArray
    from supermimic.deploy.async_sim_diagnostic_log import AsyncDiagnosticLog

    a.out.mkdir(parents=True,exist_ok=False)
    rclpy.init();node=Node('supermimic_joint_stream_readonly',enable_rosout=False,start_parameter_services=False)
    logger=AsyncDiagnosticLog(a.out);latest={};group_info={};last=[0.];count=[0]
    qos=QoSProfile(depth=10,reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.VOLATILE)
    def receive(msg,group):
        now=time.monotonic();wall=time.time()
        group_info[group]={'header_ns':int(msg.header.stamp.sec)*10**9+int(msg.header.stamp.nanosec),
                           'received_wall':wall}
        for joint in msg.joints:latest[joint.name]=(float(joint.position),float(joint.velocity))
        if not all(name in latest for name in names) or now-last[0]<.02:return
        last[0]=now
        logger.emit('packets',{'monotonic':now,'received_wall':wall,
            'q':[latest[name][0] for name in names], 'qd':[latest[name][1] for name in names],
            'groups':group_info.copy()})
        count[0]+=1
    subscriptions=[node.create_subscription(JointStateArray,'/aima/hal/joint/'+g+'/state',
        lambda msg,group=g:receive(msg,group),qos) for g in ('leg','waist','head','arm')]
    reason='stopped'
    try:
        while rclpy.ok():rclpy.spin_once(node,timeout_sec=.02)
    except (KeyboardInterrupt,ExternalShutdownException):reason='interrupted'
    except rclpy_implementation.RCLError:
        if rclpy.ok():raise
        reason='ROS shutdown'
    finally:
        status=logger.close();node.destroy_node();rclpy.try_shutdown()
        summary={'samples_enqueued':count[0],'joint_names':names,'termination':reason,
                 'diagnostic_logger':status,'motor_commands_sent':0,
                 'limitation':'Groups are latest values, not a synchronized joint snapshot'}
        (a.out/'summary.json').write_text(json.dumps(summary,allow_nan=False)+'\n')
        print(json.dumps(summary,allow_nan=False),flush=True)
        if not status['success']:raise RuntimeError('Joint recording incomplete')


if __name__=='__main__':main()
