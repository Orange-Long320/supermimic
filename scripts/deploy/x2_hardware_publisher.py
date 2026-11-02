"""Gated X2 hardware command publisher. Never run without verified readiness.

This is a separate lease-supervised ROS process, not a host/network watchdog.
An explicit plan, evidence and fresh user motor-power confirmation are required.
Its fallback requires a live process, communications and HAL; it is NOT a
freestanding balance controller or an independent hardware safety mechanism.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from supermimic.deploy.supervision import MotorPowerConsent,TargetLeaseSupervisor
from supermimic.deploy.x2_command_groups import JOINT_GROUPS,group_command
from supermimic.deploy.failsafe_evidence import require_independent_failsafe

POLICY_SHA='4424f231fc1896da098b2fca366e8522df6791596638b2b9257ba682520bbe4c'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def validated_contract(plan_path,consent_path,now,*,isolated=False):
    plan=json.loads(plan_path.read_text());consent=json.loads(consent_path.read_text())
    MotorPowerConsent(consent.get('user_confirmed_motor_power') is True,consent.get('plan_sha256',''),
                      consent.get('confirmed_wall',float('nan'))).require(digest(plan_path),now)
    if plan.get('schema')!=1 or plan.get('policy_sha256')!=POLICY_SHA:
        raise ValueError('Wrong hardware plan or policy')
    evidence_path=Path(plan['readiness_path']); evidence=json.loads(evidence_path.read_text())
    if bool(evidence.get('test_only',False)) != bool(isolated):raise ValueError('Evidence scope does not match transport')
    if digest(evidence_path)!=plan.get('readiness_sha256') or evidence.get('policy_sha256')!=POLICY_SHA:
        raise ValueError('Readiness evidence mismatch')
    safety_mode=plan.get('safety_mode','independent_hardware')
    if safety_mode not in ('independent_hardware','manual_supervised_trial'):
        raise ValueError('Unknown explicit safety mode')
    required=('ready_for_hardware_trial','preparation_and_handover_validated','command_limits_validated',
              'state_pipeline_validated','native_mc_stopped_verified')
    if any(evidence.get(k) is not True for k in required):raise ValueError('Hardware readiness not passed')
    if safety_mode=='independent_hardware':
        if evidence.get('independent_hardware_failsafe_validated') is not True:
            raise ValueError('Hardware readiness not passed: independent failsafe')
    else:
        # A distinct, explicitly disclosed trial mode; never relabel a human
        # emergency stop as automatic protection against a dead host/network.
        supervision=plan.get('manual_supervision',{})
        seconds=supervision.get('max_trial_seconds')
        if (type(seconds) not in (int,float) or not math.isfinite(seconds) or not 0<seconds<=30
                or supervision.get('support_condition')!='feet_loaded_fall_arrest'
                or supervision.get('automatic_host_loss_protection') is not False):
            raise ValueError('Invalid bounded manual-supervision plan')
        if (evidence.get('safety_mode')!='manual_supervised_trial'
                or evidence.get('manual_supervised_trial_validated') is not True
                or evidence.get('independent_hardware_failsafe_validated') is not False):
            raise ValueError('Manual trial readiness must be explicit and separate from hardware failsafe')
        for key in ('user_confirmed_manual_supervision','operator_at_estop',
                    'manual_estop_function_checked','acknowledges_no_automatic_host_loss_protection'):
            if consent.get(key) is not True:raise ValueError('Fresh manual trial confirmation missing: '+key)
    if not 0<=now-evidence.get('verified_wall',float('nan'))<=120:raise ValueError('Readiness evidence stale')
    names=plan['joint_names']; n=len(names)
    if n!=31 or set(names)!={name for group in JOINT_GROUPS.values() for name in group} or len(set(names))!=n:
        raise ValueError('Invalid joint order')
    vectors=('command_lower','command_upper','measured_lower','measured_upper','max_target_step',
             'max_stiffness','max_damping','fallback_damping')
    for key in vectors:
        if len(plan[key])!=n or not all(type(x) in (int,float) and math.isfinite(x) for x in plan[key]):
            raise ValueError('Invalid plan '+key)
    for lo,hi in (('command_lower','command_upper'),('measured_lower','measured_upper')):
        if any(a>=b for a,b in zip(plan[lo],plan[hi])):raise ValueError('Invalid plan limits')
    for key in ('max_target_step','max_stiffness','max_damping'):
        if any(x<=0 for x in plan[key]):raise ValueError('Invalid plan '+key)
    if any(x<0 or x>m for x,m in zip(plan['fallback_damping'],plan['max_damping'])):
        raise ValueError('Invalid fallback damping')
    if not all(type(plan.get(k)) in (int,float) and math.isfinite(plan[k]) and 0<plan[k]<=.05
               for k in ('max_state_age','max_target_age')):raise ValueError('Invalid timeout')
    if plan.get('output_hz') not in (100,200):raise ValueError('Unsupported command frequency')
    if not isolated and safety_mode=='independent_hardware':require_independent_failsafe(plan,now)
    return plan


def validate_target(packet,plan,previous):
    names=plan['joint_names'];n=len(names)
    if tuple(packet['names'])!=tuple(names):raise ValueError('Target joint order')
    for key in ('position','velocity','effort','stiffness','damping','measured_position'):
        if len(packet[key])!=n or not all(type(v) in (int,float) and math.isfinite(v) for v in packet[key]):
            raise ValueError('Invalid target '+key)
    if any(not lo<=x<=hi for x,lo,hi in zip(packet['measured_position'],plan['measured_lower'],plan['measured_upper'])):
        raise ValueError('Measured position outside contract')
    if any(not lo<=x<=hi for x,lo,hi in zip(packet['position'],plan['command_lower'],plan['command_upper'])):
        raise ValueError('Command position outside contract')
    if any(v!=0 for v in packet['velocity']+packet['effort']):raise ValueError('Nonzero feedforward is not allowed')
    for key,limit in (('stiffness','max_stiffness'),('damping','max_damping')):
        if any(x<0 or x>hi for x,hi in zip(packet[key],plan[limit])):raise ValueError('Gain outside contract')
    old=packet['measured_position'] if previous is None else previous
    if any(abs(x-y)>step for x,y,step in zip(packet['position'],old,plan['max_target_step'])):
        raise ValueError('Command step outside contract')
    return list(packet['position'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--consent',type=Path,required=True);p.add_argument('--isolated-prefix');a=p.parse_args()
    if a.isolated_prefix:
        if os.environ.get('ROS_DOMAIN_ID')!='197' or not re.fullmatch('/supermimic/isolated_[a-z0-9_]+',a.isolated_prefix):
            raise ValueError('Isolated testing requires domain 197 and isolated prefix')
    elif os.environ.get('ROS_DOMAIN_ID')=='197':raise ValueError('Hardware transport is prohibited in test domain')
    plan=validated_contract(a.plan,a.consent,time.time(),isolated=bool(a.isolated_prefix))
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
    from std_msgs.msg import String
    from aimdk_msgs.msg import JointCommand,JointCommandArray
    rclpy.init();node=Node('supermimic_x2_hardware_publisher',enable_rosout=False,start_parameter_services=False)
    qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.VOLATILE)
    topics={g:(a.isolated_prefix+'/hal/'+g+'/command' if a.isolated_prefix else '/aima/hal/joint/'+g+'/command')
            for g in JOINT_GROUPS}
    target_topic=(a.isolated_prefix+'/control_target' if a.isolated_prefix else '/supermimic/x2/control_target')
    # Allow DDS discovery before creating any motor publisher.
    deadline=time.monotonic()+2.
    while time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.05)
    if any(node.count_publishers(t) for t in topics.values()):raise ValueError('Another motor command publisher exists')
    validated_contract(a.plan,a.consent,time.time(),isolated=bool(a.isolated_prefix))
    supervisor=TargetLeaseSupervisor(plan['joint_names'],plan['fallback_damping'],
        max_state_age=plan['max_state_age'],max_target_age=plan['max_target_age'],
        max_active_seconds=(plan['manual_supervision']['max_trial_seconds']
                            if plan.get('safety_mode')=='manual_supervised_trial' else None))
    publishers={g:node.create_publisher(JointCommandArray,t,qos) for g,t in topics.items()}
    previous=None;sent=False
    def receive(msg):
        nonlocal previous
        try:
            packet=json.loads(msg.data);new_target=validate_target(packet,plan,previous)
            if supervisor.accept(packet):previous=new_target
        except (KeyError,TypeError,ValueError) as exc:supervisor.fault_reason='Invalid command: '+str(exc)
    node.create_subscription(String,target_topic,receive,qos)
    def tick():
        nonlocal sent
        now=time.monotonic()
        packet=supervisor.tick(now)
        if packet is None or (packet['fallback'] and not sent):return
        for group,rows in group_command(packet).items():
            msg=JointCommandArray()
            for row in rows:
                joint=JointCommand();joint.name=row['name']
                for field in ('position','velocity','effort','stiffness','damping'):setattr(joint,field,float(row[field]))
                msg.joints.append(joint)
            publishers[group].publish(msg)
        sent=True
    node.create_timer(1./plan['output_hz'],tick)
    try:rclpy.spin(node)
    finally:node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
