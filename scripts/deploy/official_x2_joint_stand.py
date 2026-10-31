"""Gated official MC position-stand trial, never a 11500 policy launcher.

Default operation only prepares a plan via read-only EM queries. Execution can
start the native MC and move motors, so it requires fresh user motor-power
consent bound to the exact plan. No low-level joint command publisher is used.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.request

from supermimic.deploy.supervision import MotorPowerConsent


def em_get(host, endpoint):
    with urllib.request.urlopen('http://'+host+':50080/json/'+endpoint, timeout=5) as response:
        return json.load(response)['data']


def em_start_mc(host):
    # Same exact endpoint/body as the installed aima EmClient.start_app.
    req=urllib.request.Request('http://'+host+':50080/json/start_app',
        data=b'{"app_name":"mc"}',headers={'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=5) as response:return json.load(response)


def config_hash(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def authorize(plan_path, consent_path):
    if consent_path is None:raise ValueError('Explicit user confirmation before motor power is required')
    c=json.loads(consent_path.read_text());sha=hashlib.sha256(plan_path.read_bytes()).hexdigest()
    MotorPowerConsent(c.get('user_confirmed_motor_power') is True,c.get('plan_sha256',''),c.get('confirmed_wall',float('nan'))).require(sha,time.time())
    if c.get('full_weight_chest_hoist_and_feet_grounded') is not True:
        raise ValueError('This trial requires operator-confirmed full-weight hoist support with feet grounded')
    if c.get('operator_at_emergency_stop') is not True:raise ValueError('Operator at emergency stop required')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mc-host',default='10.0.1.40')
    p.add_argument('--candidate',type=Path)
    p.add_argument('--execute',action='store_true');p.add_argument('--consent',type=Path)
    a=p.parse_args()
    if not a.execute:
        if a.plan.exists():raise ValueError('New plan required')
        configs=em_get(a.mc_host,'get_all_apps_config');info=em_get(a.mc_host,'get_all_apps_info')
        if 'mc' not in configs or 'mc' not in info:raise ValueError('Native MC not found')
        if a.candidate is None:raise ValueError('Selected startup candidate required for named-state checks')
        candidate=json.loads(a.candidate.read_text());names=candidate['joint_names']
        if len(names)!=31 or len(set(names))!=31:raise ValueError('Expected 31 unique joint names')
        plan={'schema':1,'scope':'official MC joint-position stand only; no 11500, locomotion or hoist unloading',
            'mc_host':a.mc_host,'native_mc_config_sha256':config_hash(configs['mc']),
            'initial_mc_status':info['mc'].get('status'),'action':'JOINT_DEFAULT','action_value':100,
            'fallback':'DAMPING_DEFAULT','fallback_value':3,'observation_seconds':12.,
            'user_motor_power_confirmation_required':True,'motor_output_enabled':False,
            'startup':'native EM start_app mc if Idle/Idel, then official SetMcAction',
            'verification':'mode accepted plus measured joint settling; selected 11500 pose must be checked separately',
            'joint_names':names,'selected_preparation_position':candidate['startup']['position'],
            'candidate_sha256':hashlib.sha256(a.candidate.read_bytes()).hexdigest(),
            'native_control_after_trial':'MC remains running in requested mode; never stop-app and retain a stiff last command'}
        a.plan.parent.mkdir(parents=True,exist_ok=True);a.plan.write_text(json.dumps(plan,indent=2)+'\n')
        print(json.dumps({'plan':str(a.plan),'plan_sha256':hashlib.sha256(a.plan.read_bytes()).hexdigest(),
                          'mc_status':plan['initial_mc_status'],'power_or_motion_command_sent':False}));return
    # This check occurs before network access or ROS imports for an execution.
    authorize(a.plan,a.consent)
    plan=json.loads(a.plan.read_text())
    if (plan['schema']!=1 or plan['action']!='JOINT_DEFAULT' or plan['action_value']!=100
            or plan['fallback']!='DAMPING_DEFAULT' or plan['fallback_value']!=3
            or plan['mc_host']!=a.mc_host or plan['observation_seconds']!=12.):raise ValueError('Unexpected trial plan')
    current=em_get(a.mc_host,'get_all_apps_config')['mc']
    if config_hash(current)!=plan['native_mc_config_sha256']:raise ValueError('Native MC config changed since plan review')
    a.out.mkdir(parents=True,exist_ok=False)
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
    from aimdk_msgs.srv import GetMcAction,SetMcAction
    from aimdk_msgs.msg import JointStateArray,CommonState
    rclpy.init();node=Node('supermimic_official_joint_stand_trial',enable_rosout=False,start_parameter_services=False)
    setter=node.create_client(SetMcAction,'/aimdk_5Fmsgs/srv/SetMcAction')
    getter=node.create_client(GetMcAction,'/aimdk_5Fmsgs/srv/GetMcAction')
    groups={};samples=[];events=[];failure=None;started_mc=False;stand_requested=False;last_mode=None
    qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.VOLATILE)
    def receive(msg,group):
        groups[group]={'received':time.monotonic(),'stamp_ns':msg.header.stamp.sec*10**9+msg.header.stamp.nanosec,
            'domain':int(msg.state.value),'joints':[{'name':j.name,'position':j.position,'velocity':j.velocity,
                                                   'effort':j.effort,'error_code':j.error_code} for j in msg.joints]}
    subs=[node.create_subscription(JointStateArray,'/aima/hal/joint/'+g+'/state',lambda m,g=g:receive(m,g),qos)
          for g in ('leg','waist','head','arm')]
    def action(name,value):
        req=SetMcAction.Request();req.header.stamp=node.get_clock().now().to_msg()
        req.source='node.set_mc_action';req.command.action_desc=name;req.command.action.value=value
        f=setter.call_async(req);rclpy.spin_until_future_complete(node,f,timeout_sec=3.)
        if not f.done() or f.result() is None:raise RuntimeError('Action response ambiguous; no blind retransmit')
        result=f.result();events.append({'time':time.time(),'requested':name,'response':str(result)})
        if result.response.status.value!=CommonState.SUCCESS:raise RuntimeError('Native action rejected')
    try:
        # Observe first; no power command until a fresh complete fault-free state.
        deadline=time.monotonic()+5
        while len(groups)<4 and time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.01)
        if len(groups)!=4:raise RuntimeError('Missing pre-power joint groups')
        for group in groups.values():
            if time.monotonic()-group['received']>.05 or any(j['error_code'] for j in group['joints']):
                raise RuntimeError('Pre-power joint state stale/faulted')
        initial=[j for group in groups.values() for j in group['joints']]
        if (len(initial)!=31 or set(j['name'] for j in initial)!=set(plan['joint_names'])
                or any(not all(math.isfinite(j[k]) for k in ('position','velocity','effort'))
                       or abs(j['velocity'])>.05 for j in initial)):
            raise RuntimeError('Pre-power joint set, values or stationary condition invalid')
        info=em_get(a.mc_host,'get_all_apps_info')['mc'];state=info.get('status')
        if state in ('Idle','Idel'):
            # Recheck consent immediately before the operation that may energize motors.
            authorize(a.plan,a.consent);events.append({'time':time.time(),'start_mc_reply':em_start_mc(a.mc_host)})
            started_mc=True
        elif state not in ('Running','Run'):raise RuntimeError('Unexpected MC runtime state: '+str(state))
        if not setter.wait_for_service(timeout_sec=10.) or not getter.wait_for_service(timeout_sec=5.):
            raise RuntimeError('Native MC action service not ready')
        stand_requested=True;action('JOINT_DEFAULT',100)
        started=time.monotonic();last_poll=0.;mode_received=started
        while time.monotonic()-started<plan['observation_seconds']:
            rclpy.spin_once(node,timeout_sec=.002);now=time.monotonic()
            if any(now-group['received']>.15 or group['domain'] in (2,3,4)
                   or any(j['error_code'] for j in group['joints']) for group in groups.values()):
                raise RuntimeError('Joint stream stale, disabled, disconnected or faulted during stand')
            if now-last_poll>.2:
                req=GetMcAction.Request();req.request.header.stamp=node.get_clock().now().to_msg()
                future=getter.call_async(req);rclpy.spin_until_future_complete(node,future,timeout_sec=.15)
                if future.done() and future.result() is not None:
                    info=future.result().info;last_mode={'value':info.current_action.value,'description':info.action_desc,'status':info.status.value}
                    mode_received=time.monotonic()
                    if last_mode['value']!=100 and not (last_mode['status']==200 and now-started<5.):
                        raise RuntimeError('Native mode differs from joint stand')
                samples.append({'elapsed':now-started,'mode':last_mode,'groups':json.loads(json.dumps(groups))});last_poll=now
            if now-mode_received>1.:raise RuntimeError('Native mode feedback stale')
        if last_mode is None or last_mode['value']!=100 or last_mode['status']!=100:
            raise RuntimeError('Joint-position stand mode not confirmed running')
    except Exception as exc:
        failure=str(exc)
        if (started_mc or stand_requested) and setter.service_is_ready():
            try:action('DAMPING_DEFAULT',3)
            except Exception as fallback:events.append({'fallback_unverified':str(fallback)})
    finally:
        result={'scope':plan['scope'],'failure':failure,'started_native_mc':started_mc,
            'stand_requested':stand_requested,'last_mode':last_mode,'events':events,'samples':samples,
            'selected_11500_pose_verified':False,'policy_process_started':False,'user_motor_power_confirmation':True}
        tail=[s for s in samples if s['elapsed']>=10.]
        if tail:
            latest={j['name']:j for group in tail[-1]['groups'].values() for j in group['joints']}
            result['final_max_joint_speed_rad_s']=max(abs(j['velocity']) for sample in tail
                for group in sample['groups'].values() for j in group['joints'])
            if set(latest)==set(plan['joint_names']):
                result['selected_pose_differences_deg']={name:math.degrees(latest[name]['position']-target)
                    for name,target in zip(plan['joint_names'],plan['selected_preparation_position'])}
            result['official_mode_running_and_low_speed']=failure is None and result['final_max_joint_speed_rad_s']<.1
        (a.out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='samples'}))
        node.destroy_node();rclpy.try_shutdown()
    if failure:raise SystemExit(1)


if __name__=='__main__':main()
