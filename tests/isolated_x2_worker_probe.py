"""Exercise worker + publisher using synthetic sensors in isolated DDS only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from supermimic.deploy.x2_command_groups import JOINT_GROUPS


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--release',type=Path,required=True);p.add_argument('--prefix',required=True);a=p.parse_args()
    if os.environ.get('ROS_DOMAIN_ID')!='197' or not a.prefix.startswith('/supermimic/isolated_'):
        raise ValueError('Requires isolated domain and topics')
    from aimdk_msgs.msg import JointCommandArray
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
    from std_msgs.msg import String
    candidate_path=a.release/'startup-candidate.json';profile=a.release/'fk-only/profile.json';manifest=a.release/'dataset/manifest.json'
    candidate=json.loads(candidate_path.read_text());names=candidate['joint_names'];n=len(names);now=time.time()
    evidence={'test_only':True,'policy_sha256':sha(a.release/'policy.pt'),'verified_wall':now,**{k:True for k in (
        'ready_for_hardware_trial','preparation_and_handover_validated','command_limits_validated',
        'state_pipeline_validated','independent_hardware_failsafe_validated','native_mc_stopped_verified')}}
    ep=a.folder/'synthetic-evidence.json';ep.write_text(json.dumps(evidence))
    fixture=Path('/home/agi/lxh-supermimic/policy-fixture.json')
    numpy_policy=Path('/home/agi/lxh-supermimic/readiness_work_20261031_r1/numpy-policy-11500.npz')
    plan={'schema':1,'policy_sha256':evidence['policy_sha256'],'readiness_path':str(ep),'readiness_sha256':sha(ep),
        'release_path':str(a.release),'candidate_sha256':sha(candidate_path),'profile_sha256':sha(profile),
        'policy_fixture_path':str(fixture),'policy_fixture_sha256':sha(fixture),
        'policy_backend':'numpy','numpy_policy_path':str(numpy_policy),'numpy_policy_sha256':sha(numpy_policy),
        'dataset_manifest_sha256':sha(manifest),'joint_names':names,'command_lower':[-100.]*n,'command_upper':[100.]*n,
        'measured_lower':[-100.]*n,'measured_upper':[100.]*n,'max_target_step':[100.]*n,
        'max_stiffness':[100.]*n,'max_damping':[100.]*n,'fallback_damping':[3.]*n,
        'max_state_age':.05,'max_target_age':.05,'output_hz':100,'prep_kp':[1.]*n,'prep_kd':[1.]*n,
        'policy_kp':[1.]*n,'policy_kd':[1.]*n,'max_velocity':[10.]*n,'max_acceleration':[10.]*n,
        'minimum_duration':.1,'blend_seconds':.2,'max_tick_gap':.031,'max_tilt':3.,'max_angular_speed':10.}
    pp=a.folder/'synthetic-plan.json';pp.write_text(json.dumps(plan));cp=a.folder/'synthetic-consent.json'
    cp.write_text(json.dumps({'user_confirmed_motor_power':True,'plan_sha256':sha(pp),'confirmed_wall':now,
                              'scope':'SYNTHETIC ISOLATED TEST ONLY; NOT USER CONSENT'}))
    (a.folder/'fixtures-ready').write_text('synthetic isolated test only\n')
    rclpy.init();node=Node('supermimic_synthetic_probe',enable_rosout=False,start_parameter_services=False)
    reliable=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.VOLATILE)
    best=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.VOLATILE)
    state=node.create_publisher(String,a.prefix+'/policy_state',best);trigger=node.create_publisher(String,a.prefix+'/policy_request',reliable)
    groups={g:[] for g in JOINT_GROUPS}; subs=[];phases=[]
    for group in JOINT_GROUPS:
        def capture(msg,g=group):groups[g].append((time.monotonic(),[j.stiffness for j in msg.joints]))
        subs.append(node.create_subscription(JointCommandArray,a.prefix+'/hal/'+group+'/command',capture,reliable))
    start=time.monotonic();requested=False;target_sub=None;state_enabled=True;stop_at=None;request_process=None
    q=candidate['startup']['position'];quat=candidate['startup']['root_quat_wxyz'];position=candidate['startup']['root_position']
    def capture_target(msg):phases.append(json.loads(msg.data)['phase'])
    def publish_state():
        if not state_enabled:return
        t=time.time_ns();ts=t*1e-9
        row={'schema':2,'session_id':'synthetic-only','motor_output_enabled':False,'joint_names':names,
             'state_stamp_ns':t,'state_stamp':ts,'published_wall':ts,'joint_stamps_ns':[t]*4,
             'joint_stamps':[ts]*4,'joint_received_wall':[ts]*4,'joint_domains':[0]*4,
             'joint_position':q,'joint_velocity':[0.]*n,'quat_wxyz':quat,'position':position,'angular_velocity_body':[0.]*3}
        msg=String();msg.data=json.dumps(row);state.publish(msg)
    node.create_timer(.02,publish_state)
    try:
        while time.monotonic()-start<30:
            rclpy.spin_once(node,timeout_sec=.01)
            if target_sub is None and node.count_subscribers(a.prefix+'/policy_request')==1:
                target_sub=node.create_subscription(String,a.prefix+'/control_target',capture_target,reliable)
            if not requested and phases.count('prepare')>=12:
                request_process=subprocess.Popen([sys.executable,str(a.folder/'scripts/deploy/x2_11500_worker.py'),
                    '--plan',str(pp),'--consent',str(cp),'--isolated-prefix',a.prefix,'--request-policy'],
                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                requested=True
            if 'active' in phases and stop_at is None and request_process is not None and request_process.poll() is not None:
                state_enabled=False;stop_at=time.monotonic()
            if stop_at is not None and time.monotonic()-stop_at>.4:break
        if not requested or 'active' not in phases:raise ValueError('Synthetic worker did not reach active')
        if request_process is None:raise ValueError('Missing policy request process')
        request_output,_=request_process.communicate(timeout=5)
        if request_process.returncode!=0 or 'acknowledged' not in request_output:
            raise ValueError('Policy request CLI did not receive acknowledgement: '+request_output)
        for group,items in groups.items():
            if not any(v and all(x==1. for x in v) for _,v in items):raise ValueError('Missing command: '+group)
            if not any(t>=stop_at and v and all(x==0. for x in v) for t,v in items):raise ValueError('Missing fallback: '+group)
        result={'scope':__doc__,'synthetic_test_only':True,'state_stopped_after_active':True,
                'observed_phases':sorted(set(phases)),'policy_request_cli_acknowledged':True,
                'fallback_all_groups':True,'hardware_output':False}
        (a.folder/'worker-probe-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    finally:
        if request_process is not None and request_process.poll() is None:
            request_process.terminate();request_process.wait(timeout=2)
        node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
