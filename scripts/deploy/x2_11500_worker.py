"""Gated X2 11500 worker: prepare, then accept a policy handover request.

No motor power API is used. Publishes only proposals to the separately gated
hardware publisher. Not an independent hardware or network failsafe.
"""
import argparse
import json
import os
from pathlib import Path
import re
import time
import numpy as np

from x2_hardware_publisher import digest,validated_contract,POLICY_SHA
from supermimic.deploy.early_handover import EarlyHandover,Phase
from supermimic.deploy.policy_packet import validate_policy_packet


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--consent',type=Path,required=True);p.add_argument('--request-policy',action='store_true')
    p.add_argument('--keyboard-port',type=int,help='Loopback-only keyboard page; no motor failsafe')
    p.add_argument('--zero-horizontal-anchor-feedback',action='store_true',
                   help='Remove world XY anchor errors only for stand and in-place turns')
    p.add_argument('--isolated-prefix');p.add_argument('--startup-ready-file',type=Path);a=p.parse_args()
    if a.request_policy and a.startup_ready_file:
        raise ValueError('Warmup marker is only valid for the preparation worker')
    if a.keyboard_port is not None and (a.request_policy or not 1024<=a.keyboard_port<=65535):
        raise ValueError('Invalid keyboard listener configuration')
    if a.isolated_prefix:
        if os.environ.get('ROS_DOMAIN_ID')!='197' or not re.fullmatch('/supermimic/isolated_[a-z0-9_]+',a.isolated_prefix):
            raise ValueError('Isolated testing requires domain 197 and isolated prefix')
    elif os.environ.get('ROS_DOMAIN_ID')=='197':raise ValueError('Hardware worker is prohibited in test domain')
    plan=validated_contract(a.plan,a.consent,time.time(),isolated=bool(a.isolated_prefix));plan_hash=digest(a.plan)
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
    from std_msgs.msg import String
    rclpy.init();node=Node('supermimic_x2_policy_request' if a.request_policy else 'supermimic_x2_11500_worker',
                          enable_rosout=False,start_parameter_services=False)
    qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.VOLATILE)
    trigger=(a.isolated_prefix+'/policy_request' if a.isolated_prefix else '/supermimic/x2/policy_request')
    if a.request_policy:
        target_topic=(a.isolated_prefix+'/control_target' if a.isolated_prefix else '/supermimic/x2/control_target')
        acknowledged=[]
        def ack(msg):
            try:
                phase=json.loads(msg.data).get('phase')
                if phase in ('handover','active'):acknowledged.append(phase)
            except (ValueError,TypeError):pass
        node.create_subscription(String,target_topic,ack,qos)
        pub=node.create_publisher(String,trigger,qos);deadline=time.monotonic()+3.
        while node.count_subscribers(trigger)!=1 and time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.05)
        if node.count_subscribers(trigger)!=1:raise ValueError('No unique preparation worker')
        msg=String();msg.data=plan_hash;pub.publish(msg)
        deadline=time.monotonic()+3.
        while not acknowledged and time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.05)
        if not acknowledged:raise ValueError('Policy handover was not acknowledged')
        print('policy request acknowledged: '+acknowledged[-1],flush=True)
        node.destroy_node();rclpy.try_shutdown();return
    release=Path(plan['release_path']);candidate_path=release/'startup-candidate.json';policy_path=release/'policy.pt'
    profile=release/'fk-only/profile.json';dataset=release/'dataset'
    for path,key in ((candidate_path,'candidate_sha256'),(profile,'profile_sha256'),(dataset/'manifest.json','dataset_manifest_sha256')):
        if digest(path)!=plan[key]:raise ValueError('Release asset hash mismatch: '+key)
    if digest(policy_path)!=POLICY_SHA:raise ValueError('Wrong policy file')
    candidate=json.loads(candidate_path.read_text());names=plan['joint_names']
    if candidate['joint_names']!=names:raise ValueError('Candidate joint order mismatch')
    import torch
    os.environ['SM_ROBOT']='x2_ultra';os.environ['SM_ROBOT_SPEC']=str(profile)
    from supermimic.deploy.skill_goal import SkillGoal
    from supermimic.deploy.torch_policy import BufferedTorchPolicy
    torch.set_num_threads(1)
    if plan.get('policy_backend')=='numpy':
        from supermimic.deploy.numpy_policy import NumpyPolicy
        policy=NumpyPolicy(Path(plan['numpy_policy_path']),expected_sha256=plan['numpy_policy_sha256'])
    elif plan.get('policy_backend')=='torch':policy=BufferedTorchPolicy(policy_path)
    else:raise ValueError('Explicit policy backend required')
    fixture_path=Path(plan['policy_fixture_path'])
    if digest(fixture_path)!=plan['policy_fixture_sha256']:raise ValueError('Policy fixture hash mismatch')
    fixture=json.loads(fixture_path.read_text())
    for _ in range(20):fixture_output=policy.infer(fixture['obs'],fixture['goal'])
    if np.max(np.abs(fixture_output-np.asarray(fixture['expected'])))>1e-4:
        raise ValueError('Policy fixture mismatch')
    # Load FK/reference assets and warm inference before subscribing. Doing this
    # in the first callback leaves real sensor packets queued for seconds;
    # they correctly fail the 50 ms age guard when dispatch resumes.
    goal_options = {}
    if a.zero_horizontal_anchor_feedback:
        goal_options['zero_horizontal_anchor_modes'] = ('stand','turn_left_inplace','turn_right_inplace')
    tracker=SkillGoal(dataset,profile,candidate['startup']['root_position'],candidate['startup']['root_quat_wxyz'], **goal_options)
    print('zero world-horizontal anchor modes:', goal_options.get('zero_horizontal_anchor_modes', ()), flush=True)
    anchor_config_path=(a.startup_ready_file.parent if a.startup_ready_file else a.plan.parent.parent)/'anchor-feedback.json'
    anchor_config_path.write_text(json.dumps({
        'zero_world_horizontal_modes': list(goal_options.get('zero_horizontal_anchor_modes', ())),
        'height_and_orientation_preserved': True, 'estimator_reset': False})+'\n')
    if a.keyboard_port is not None:
        # SkillPhase appends seam and mode events indefinitely. Keep recent
        # diagnostics bounded; requested mode changes are printed separately.
        from collections import deque
        tracker.scheduler.events=deque(tracker.scheduler.events,maxlen=4096)
    for _ in range(10):
        warm_goal=tracker.goal(candidate['startup']['root_position'],candidate['startup']['root_quat_wxyz'],
                               candidate['startup']['position'],'stand')
        policy.infer(np.zeros(990,dtype=np.float32),warm_goal)
    target_topic=(a.isolated_prefix+'/control_target' if a.isolated_prefix else '/supermimic/x2/control_target')
    state_topic=(a.isolated_prefix+'/policy_state' if a.isolated_prefix else '/supermimic/x2_online_lio/policy_state')
    if a.startup_ready_file:
        # Models are warm, but no target publisher has been created yet.
        with a.startup_ready_file.open('x') as ready:
            json.dump({'model_warm':True,'backend':plan['policy_backend'],
                       'wall':time.time(),'target_publisher_created':False},ready)
        deadline=time.monotonic()+30.
        while time.monotonic()<deadline:
            if node.count_publishers(state_topic)==1 and node.count_subscribers(target_topic)==1:break
            rclpy.spin_once(node,timeout_sec=.05)
    else:
        deadline=time.monotonic()+2.
        while time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.05)
    if node.count_publishers(target_topic):raise ValueError('Another policy worker exists')
    state_publishers=node.count_publishers(state_topic);target_subscribers=node.count_subscribers(target_topic)
    if state_publishers!=1 or target_subscribers!=1:
        raise ValueError(f'Expected one state producer and one motor publisher; '
                         f'got state publishers {state_publishers} and target subscribers {target_subscribers}')
    keyboard=None
    if a.keyboard_port is not None:
        from x2_keyboard import Keyboard
        keyboard=Keyboard(a.keyboard_port)
        print('keyboard page listening on 127.0.0.1:%d; waits for active policy'%a.keyboard_port,flush=True)
    pub=node.create_publisher(String,target_topic,qos);controller=None;session=None;seq=0;last_receipt=None;dropped=0;seen=None;diag_prints=[0];last_mode=None
    def request(msg):
        if controller is not None and msg.data==plan_hash:
            try:controller.request_handover();print('policy handover requested',flush=True)
            except ValueError as exc:print('handover rejected: '+str(exc),flush=True)
    node.create_subscription(String,trigger,request,qos)
    def receive(msg):
        nonlocal controller,session,seq,last_receipt,dropped,seen,last_mode
        if controller is not None and controller.phase is Phase.FAULT:return
        try:
            now_wall=time.time();now_mono=time.monotonic()
            row=json.loads(msg.data)
            # A packet that was already late when dispatch resumed carries no
            # control history: drop it and step on the next fresh one. No stale
            # state is ever used, and the controller's tick-gap guard stays the
            # hard limit, so one late packet no longer aborts the whole trial.
            late=[now_wall-row['state_stamp'],*[now_wall-t for t in row['joint_stamps']],
                  now_wall-row['published_wall'],*[now_wall-t for t in row['joint_received_wall']]]
            if any(t>plan['max_state_age'] for t in late):
                dropped+=1
                if dropped<4:print('dropped late packet: age %.4f s'%late[0],flush=True)
                return
            # A repeated observation instant carries no new state and would read as
            # a zero timing gap in the controller; drop it the same way.
            if seen is not None and row['state_stamp']<=seen:
                dropped+=1
                if dropped<4:print('dropped duplicate packet',flush=True)
                return
            seen=row['state_stamp']
            row=validate_policy_packet(row,names,now_wall,max_age=plan['max_state_age'])
            if session is not None and row['session_id']!=session:raise ValueError('State session changed')
            session=row['session_id'];q=np.asarray(row['joint_position']);qd=row['joint_velocity'];quat=np.asarray(row['quat_wxyz'])
            w,x,y,z=quat;gravity=[2*(w*y-x*z),-2*(w*x+y*z),-1+2*(x*x+y*y)]
            if controller is None:
                def factory(position,orientation):
                    tracker.position=np.asarray(position).copy()
                    tracker.yaw=tracker.rotation(orientation).as_euler('xyz')[2]
                    return tracker
                prep_target=np.asarray(plan.get('prep_target') or candidate['startup']['position'],dtype=float)
                unclipped_waist=plan.get('unclipped_waist_pitch_target',False)
                unclipped_all=plan.get('unclipped_all_joint_targets',False)
                if type(unclipped_waist) is not bool:raise ValueError('Invalid waist pitch override')
                if type(unclipped_all) is not bool:raise ValueError('Invalid all-joint override')
                controller=EarlyHandover(q,prep_target,candidate['action_default'],candidate['action_scale'],
                    plan['prep_kp'],plan['prep_kd'],plan['policy_kp'],plan['policy_kd'],plan['measured_lower'],plan['measured_upper'],
                    plan['command_lower'],plan['command_upper'],plan['max_target_step'],max_velocity=plan['max_velocity'],
                    max_acceleration=plan['max_acceleration'],minimum_duration=plan['minimum_duration'],
                    blend_seconds=plan['blend_seconds'],max_tick_gap=plan['max_tick_gap'],max_tilt=plan['max_tilt'],
                    max_angular_speed=plan['max_angular_speed'],policy=policy,tracker_factory=factory,
                    history_source=plan['history_source'],max_tilt_delta=plan.get('max_tilt_delta'),
                    handover_max_tilt=plan.get('handover_max_tilt'),
                    zero_stand_anchor_xy=plan.get('zero_stand_anchor_xy',False),
                    unclipped_target_indices=tuple(range(len(names))) if unclipped_all else
                        (names.index('waist_pitch_joint'),) if unclipped_waist else ())
                print('legacy body-frame stand XY override:',controller.zero_stand_anchor_xy,flush=True)
                print('waist pitch target range clipping disabled:',unclipped_waist or unclipped_all,flush=True)
                print('all-joint target range clipping disabled:',unclipped_all,flush=True)
                return  # Reacquire fresh input after setup; no command sent yet.
            mode=keyboard.read() if keyboard is not None else 'stand'
            out=controller.step(row['state_stamp'],row['position'],quat,q,qd,row['angular_velocity_body'],gravity,mode=mode)
            if keyboard is not None and out['phase']=='active':keyboard.set_ready(True)
            if mode!=last_mode:
                print('requested skill:',mode,flush=True);last_mode=mode
            generated=time.monotonic();state_sampled=generated-(time.time()-row['state_stamp'])
            packet={'names':names,'sequence':seq,'generated':generated,'state_sampled':state_sampled,
                    'measured_position':q.tolist(),'phase':out['phase'],
                    **{k:out[k].tolist() for k in ('position','velocity','effort','stiffness','damping')}}
            m=String();m.data=json.dumps(packet,allow_nan=False);pub.publish(m);seq+=1;last_receipt=now_mono
            diag=getattr(controller,'diagnostics',None)
            if diag is not None:diag_prints[0]+=1
            dump_path=os.environ.get('SM_DIAG_OBS')
            if dump_path and getattr(controller,'obs_dump',None) is not None:
                Path(dump_path).write_text(json.dumps(controller.obs_dump,allow_nan=False))
                controller.obs_dump=None
            if seq%50==0 or (diag is not None and diag_prints[0]<=6):
                sat=[names[i] for i in getattr(controller,'saturation_joints',[])][:4]
                print('phase %s seq %d saturated %d max_over %.3f rate_limited %d max_step_over %.3f sat_joints %s'
                      % (out['phase'],seq,getattr(controller,'saturated',0),
                         getattr(controller,'saturation_max',0.),getattr(controller,'rate_limited',0),
                         getattr(controller,'rate_limit_max',0.),','.join(sat) or '-'),flush=True)
                if diag is not None:
                    gap=np.abs(diag['desired']-diag['measured'])
                    worst=np.argsort(-gap)[:5]
                    print('  policy tick %d: desired-measured '%diag_prints[0]
                          +', '.join('%s %+.3f'%(names[i],diag['desired'][i]-diag['measured'][i]) for i in worst),
                          flush=True)
                    if controller.unclipped_target_indices:
                        i=names.index('waist_pitch_joint')
                        print('  waist pitch: action %.6f desired %.6f worker_target %.6f measured %.6f rad'
                              % (diag['action'][i],diag['desired'][i],out['position'][i],diag['measured'][i]),flush=True)

        except Exception as exc:
            # No target has been sent yet, so a stale startup packet carries no
            # control history: wait for a fresh one instead of latching a fault
            # on the producer's startup transient. Once publishing, a stale
            # packet is a real gap and still latches.
            if seq==0 and 'stale or future' in str(exc):
                print('dropped stale startup packet: '+str(exc),flush=True);return
            if keyboard is not None:keyboard.set_ready(False)
            if controller is not None:controller.phase=Phase.FAULT;controller.reason=str(exc)
            print('policy worker fault: '+str(exc),flush=True)
            # The contract refuses to say which joint, so record the context: the
            # controller keeps the offending indices and values on every fault.
            if controller is not None and controller.fault_context:
                print('fault context: '+json.dumps(controller.fault_context),flush=True)
            # The independent publisher lease detects the absence of fresh targets.
    # History requires every real sample. KEEP_LAST(1) discards an intervening
    # sample whenever dispatch is briefly delayed, turning regular 20 ms source
    # ticks into a 40 ms history gap. Retain a bounded queue; receive() still
    # rejects stale packets at 50 ms and the controller keeps its gap guard.
    sensor_qos=QoSProfile(depth=10,reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.VOLATILE)
    node.create_subscription(String,state_topic,receive,sensor_qos)
    try:rclpy.spin(node)
    finally:
        if keyboard is not None:keyboard.close()
        node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
