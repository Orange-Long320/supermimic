"""Bounded read-only native HAL readiness/DCU telemetry. Run inside lxh.

Uses a private copy of the robot's current native interfaces when the public SDK
lacks these Get services. No Set clients, actuator publishers or enable paths.
Unspecified firmware status fields are recorded as raw values, not interpreted.
"""
import argparse
import json
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise ValueError('New output required')
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile,ReliabilityPolicy
    from aimdk_msgs.msg import JointNoRealTimeStateArray
    from aimdk_msgs.srv import GetAllJointIsReady,GetDcuInfo,GetSystemState,GetModulePowerState,GetPmuInfoList
    rclpy.init();node=Node('supermimic_readonly_hardware_status',enable_rosout=False,start_parameter_services=False)
    executor=SingleThreadedExecutor();executor.add_node(node)
    counts={};latest={}
    def receive(msg,group):
        counts[group]=counts.get(group,0)+1
        latest[group]=dict(received_wall=time.time(),joints=[dict(name=j.name,
            motor_current_state_raw=int(j.motor_current_state),target_effort_raw=int(j.target_effort),
            vbus_raw=int(j.vbus),ibus_raw=int(j.ibus),motor_temp_raw=int(j.motor_temp)) for j in msg.joints])
    qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT)
    subscriptions=[node.create_subscription(JointNoRealTimeStateArray,
        '/aima/hal/joint/norealtime/'+g+'_joint/state',lambda msg,g=g:receive(msg,g),qos) for g in ('udcu','ddcu')]
    types=(GetAllJointIsReady,GetDcuInfo,GetSystemState,GetModulePowerState,GetPmuInfoList)
    calls={};services={};clients=[]
    try:
        for cls in types:
            name=cls.__name__;client=node.create_client(cls,'/aimdk_5Fmsgs/srv/'+name);clients.append(client)
            if not client.wait_for_service(timeout_sec=.5):
                services[name]=dict(available=False);continue
            req=cls.Request();header=getattr(req,'request',getattr(req,'header',None))
            header.header.stamp=node.get_clock().now().to_msg()
            calls[name]=client.call_async(req)
        end=time.monotonic()+5.
        while time.monotonic()<end:executor.spin_once(timeout_sec=.02)
        for name,future in calls.items():
            if not future.done() or future.result() is None:
                services[name]=dict(available=True,response_received=False);continue
            result=future.result();common=getattr(result,'reponse',getattr(result,'header',None))
            row=dict(available=True,response_received=True,status=int(common.status.value))
            if name=='GetAllJointIsReady':row['all_joints_initialized']=bool(result.state)
            elif name=='GetDcuInfo':
                row['versions']={k:getattr(result,k) for k in result.get_fields_and_field_types()
                                 if k.endswith(('_fw_ver','_hw_ver','_build_time'))}
            elif name=='GetSystemState':row.update(cur_state=result.cur_state,status_value=int(result.curr_status.value))
            elif name=='GetModulePowerState':
                row.update(result_code=int(result.result),module_states=[dict(module=int(s.module),power_state=int(s.power_state))
                                                                         for s in result.module_states])
            elif name=='GetPmuInfoList':
                row['pmu']=[{k:getattr(s,k) for k in ('pmu_bool_status','bus_48v_current','output_48v_current',
                    'bus_48v_voltage','output_48v_voltage','battery_voltage','battery_current')} for s in result.pmu_infos.pmu]
            services[name]=row
        report=dict(scope=__doc__,services=services,norealtime_counts=counts,norealtime_latest=latest,
                    probe_publishes_motor_commands=False,power_or_mode_command_sent=False)
        args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps(report,indent=2,allow_nan=False))
    finally:
        executor.shutdown(timeout_sec=1.);node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
