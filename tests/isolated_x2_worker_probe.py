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
    p.add_argument('--release',type=Path,required=True);p.add_argument('--prefix',required=True)
    p.add_argument('--hold-active-seconds',type=float,default=.2);a=p.parse_args()
    if os.environ.get('ROS_DOMAIN_ID')!='197' or not a.prefix.startswith('/supermimic/isolated_'):
        raise ValueError('Requires isolated domain and topics')
    if not .2<=a.hold_active_seconds<=180.:raise ValueError('Invalid synthetic hold duration')
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
        'minimum_duration':.1,'blend_seconds':.2,'max_tick_gap':.031,'max_tilt':3.,'max_angular_speed':10.,
        'history_source':'policy'}
    pp=a.folder/'synthetic-plan.json';pp.write_text(json.dumps(plan));cp=a.folder/'synthetic-consent.json'
    cp.write_text(json.dumps({'user_confirmed_motor_power':True,'plan_sha256':sha(pp),'confirmed_wall':now,
                              'scope':'SYNTHETIC ISOLATED TEST ONLY; NOT USER CONSENT'}))
    (a.folder/'fixtures-ready').write_text('synthetic isolated test only\n')
    rclpy.init();node=Node('supermimic_synthetic_probe',enable_rosout=False,start_parameter_services=False)
    reliable=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.VOLATILE)
    groups={g:[] for g in JOINT_GROUPS}; subs=[];phases=[];phase_times=[]
    for group in JOINT_GROUPS:
        def capture(msg,g=group):groups[g].append((time.monotonic(),[j.stiffness for j in msg.joints]))
        subs.append(node.create_subscription(JointCommandArray,a.prefix+'/hal/'+group+'/command',capture,reliable))
    start=time.monotonic();requested=False;target_sub=None;stop_at=None;request_process=None;active_at=None
    source_command=[sys.executable,str(a.folder/'tests/isolated_x2_state_source.py'),
        '--release',str(a.release),'--prefix',a.prefix,'--out',str(a.folder/'synthetic-source-timing.json')]
    if os.environ.get('SM_TEST_SOURCE_CPU'):
        cpu=os.environ['SM_TEST_SOURCE_CPU']
        if not cpu.isdigit():raise ValueError('Invalid test CPU')
        source_command=['taskset','-c',cpu,*source_command]
    sensor_process=subprocess.Popen(source_command,
        stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
    def capture_target(msg):
        phase=json.loads(msg.data)['phase'];phases.append(phase);phase_times.append((time.monotonic(),phase))
    try:
        while time.monotonic()-start<30+a.hold_active_seconds:
            rclpy.spin_once(node,timeout_sec=.01)
            if target_sub is None and node.count_subscribers(a.prefix+'/policy_request')==1:
                target_sub=node.create_subscription(String,a.prefix+'/control_target',capture_target,reliable)
            if not requested and phases.count('prepare')>=12:
                request_process=subprocess.Popen([sys.executable,str(a.folder/'scripts/deploy/x2_11500_worker.py'),
                    '--plan',str(pp),'--consent',str(cp),'--isolated-prefix',a.prefix,'--request-policy'],
                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                requested=True
            if 'active' in phases and active_at is None:active_at=time.monotonic()
            if (active_at is not None and time.monotonic()-active_at>=a.hold_active_seconds and stop_at is None
                    and request_process is not None and request_process.poll() is not None):
                stop_at=time.monotonic();sensor_process.terminate();sensor_process.wait(timeout=2)
            if stop_at is not None and time.monotonic()-stop_at>.4:break
        if not requested or 'active' not in phases or stop_at is None:raise ValueError('Synthetic worker did not sustain active')
        if not any(t>=stop_at-.06 and phase=='active' for t,phase in phase_times):
            raise ValueError('Active target stream stopped before hold ended')
        if request_process is None:raise ValueError('Missing policy request process')
        request_output,_=request_process.communicate(timeout=5)
        if request_process.returncode!=0 or 'acknowledged' not in request_output:
            raise ValueError('Policy request CLI did not receive acknowledgement: '+request_output)
        for group,items in groups.items():
            if not any(v and all(x==1. for x in v) for _,v in items):raise ValueError('Missing command: '+group)
            if any(active_at<=t<stop_at and v and all(x==0. for x in v) for t,v in items):
                raise ValueError('Premature watchdog fallback: '+group)
            if not any(t>=stop_at and v and all(x==0. for x in v) for t,v in items):raise ValueError('Missing fallback: '+group)
        result={'scope':__doc__,'synthetic_test_only':True,'state_stopped_after_active':True,
                'observed_phases':sorted(set(phases)),'active_hold_s':stop_at-active_at,
                'policy_request_cli_acknowledged':True,
                'fallback_all_groups':True,'hardware_output':False}
        (a.folder/'worker-probe-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    finally:
        if sensor_process.poll() is None:sensor_process.terminate();sensor_process.wait(timeout=2)
        if request_process is not None and request_process.poll() is None:
            request_process.terminate();request_process.wait(timeout=2)
        node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
