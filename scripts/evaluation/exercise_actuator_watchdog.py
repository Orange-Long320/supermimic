"""CPU-only three-process fault injection for an actuator-watchdog MODEL.

No ROS, socket to a robot, motor API or power operation. The actuator process
models a missing independent device; this is never hardware readiness evidence.
"""
import argparse
from dataclasses import asdict
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import time
import uuid

from supermimic.deploy.actuator_watchdog import ActuatorDeadlineWatchdog


def latest(q, item):
    try:
        q.put_nowait(item)
    except queue.Full:
        try:q.get_nowait()
        except queue.Empty:pass
        try:q.put_nowait(item)
        except queue.Full:pass


def source(challenges, responses, stop, start, ready, freeze):
    ready.put('source');start.wait()
    seq=0;state=0
    while not stop.is_set():
        try:c=challenges.get(timeout=.01)
        except queue.Empty:continue
        if not freeze.is_set():state+=1
        latest(responses,dict(session=c['session'],nonce=c['nonce'],command_sequence=seq,
                              state_sequence=state,state_age=0.))
        seq+=1


def broker(challenges, requests, responses, applied, stop, start, ready, drop, replay):
    ready.put('publisher');start.wait();previous=None
    while not stop.is_set():
        try:c=challenges.get(timeout=.002)
        except queue.Empty:c=None
        if c is not None and not drop.is_set():latest(requests,c)
        try:frame=responses.get(timeout=.002)
        except queue.Empty:continue
        if not drop.is_set():
            latest(applied,previous if replay.is_set() and previous is not None else frame)
        previous=frame


def actuator(challenges, frames, stop, start, ready, warmed, results, session):
    ready.put('actuator_model');start.wait()
    watchdog=ActuatorDeadlineWatchdog(command_timeout=.05,max_state_age=.05,max_round_trip=.025)
    began=time.monotonic();watchdog.begin(session,began,inhibit_observed=True)
    accepted=0;next_issue=began;first_active=None
    try:
        while not stop.is_set():
            now=time.monotonic()
            if now>=next_issue:
                c=watchdog.challenge(now)
                if c is not None:latest(challenges,asdict(c))
                next_issue=now+.01
            while True:
                try:frame=frames.get_nowait()
                except queue.Empty:break
                if watchdog.accept(**frame,now=time.monotonic()):
                    accepted+=1
                    if first_active is None:first_active=time.monotonic()
            now=time.monotonic();watchdog.tick(now)
            if accepted>=10:warmed.set()
            if watchdog.reason is not None:break
            time.sleep(.001)
        now=time.monotonic()
        results.put(dict(accepted_frames=accepted,inhibit_required=watchdog.inhibit_required,
            reason=watchdog.reason,last_valid_at=watchdog.last_accepted_at,stopped_at=now,
            deadline=watchdog.deadline,first_active_at=first_active,
            elapsed_s=now-began))
    except BaseException as exc:
        results.put(dict(error=repr(exc)))
        raise


def run_case(case):
    ctx=mp.get_context('spawn')
    channels=[ctx.Queue(maxsize=2) for _ in range(4)]
    challenges,requests,responses,frames=channels
    ready=ctx.Queue();results=ctx.Queue();stop=ctx.Event();start=ctx.Event();warmed=ctx.Event()
    freeze=ctx.Event();drop=ctx.Event();replay=ctx.Event();session=uuid.uuid4().hex
    children=[ctx.Process(target=source,args=(requests,responses,stop,start,ready,freeze),name='model_source'),
        ctx.Process(target=broker,args=(challenges,requests,responses,frames,stop,start,ready,drop,replay),name='model_publisher'),
        ctx.Process(target=actuator,args=(challenges,frames,stop,start,ready,warmed,results,session),name='model_actuator')]
    paused=False;injected=None
    try:
        for child in children:child.start()
        for _ in children:ready.get(timeout=10.)
        start.set()
        if not warmed.wait(timeout=3.):
            raise RuntimeError('Healthy startup failed: '+str(results.get(timeout=1.)))
        injected=time.monotonic()
        if case=='healthy':time.sleep(.2);stop.set()
        elif case=='worker_exit':children[0].terminate()
        elif case=='worker_freeze':os.kill(children[0].pid,signal.SIGSTOP);paused=True
        elif case=='publisher_exit':children[1].terminate()
        elif case=='command_host_loss':
            children[0].terminate();children[1].terminate()
        elif case=='transport_loss':drop.set()
        elif case=='frozen_source_state':freeze.set()
        elif case=='replayed_command':replay.set()
        else:raise ValueError('Unknown case')
        result=results.get(timeout=2.)
        if 'error' in result:raise RuntimeError(result['error'])
        result.update(case=case,injected_at=injected)
        result['model_passed']=(result['accepted_frames']>=10 and
            ((result['reason'] is None and not result['inhibit_required']) if case=='healthy' else
             (result['reason'] is not None and result['inhibit_required'])))
        result['inhibit_after_injection_ms']=(result['stopped_at']-injected)*1000
        if result['last_valid_at'] is not None:
            result['inhibit_after_last_valid_ms']=(result['stopped_at']-result['last_valid_at'])*1000
        return result
    finally:
        stop.set();start.set()
        if paused and children[0].is_alive():os.kill(children[0].pid,signal.SIGCONT)
        for child in children:
            if child.pid is None:continue
            child.join(timeout=1.)
            if child.is_alive():child.terminate();child.join(timeout=1.)
            if child.is_alive():child.kill();child.join(timeout=1.)
        for channel in channels+[ready,results]:
            channel.cancel_join_thread();channel.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('New output required')
    cases=['healthy','worker_exit','worker_freeze','publisher_exit','command_host_loss',
           'transport_loss','frozen_source_state','replayed_command']
    rows=[run_case(case) for case in cases]
    result=dict(scope=__doc__,test_only=True,simulation_only=True,
        independent_hardware_failsafe_validated=False,ready_for_hardware_trial=False,
        power_or_control_operation_sent=False,cases=rows,
        all_protocol_model_cases_passed=all(r['model_passed'] for r in rows),
        not_covered=['Actual X2 HAL expiry/inhibit','Actuator safety-device failure',
                     'Actual drive/host power interruption','Physical stopping behavior and support'])
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if not result['all_protocol_model_cases_passed']:raise SystemExit(1)


if __name__=='__main__':main()
