"""Apply the existing full-motion Webster gates to an aligned policy capture.

Uses the optimizer's root-up angle convention and one full terminal second.
Collision measurements must first be checked by audit_native_control_trace.py.
This script neither executes nor changes physics.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--native-audit',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():parser.error('Use a new result path')
    trajectory=args.capture/'trajectory.npz'
    source=json.loads((args.capture/'result.json').read_text())
    independent=json.loads(args.native_audit.read_text())
    digest=hashlib.sha256(trajectory.read_bytes()).hexdigest()
    if independent['trajectory_sha256']!=digest:
        raise ValueError('Independent native audit is for a different trajectory')
    if max(independent['maximum_recorded_metric_difference'])>=1e-5:
        raise ValueError('Recorded metrics do not agree with independent native reconstruction')
    with np.load(trajectory) as trace:
        rows=trace['metrics'].copy()
        physics_dt=float(trace['physics_dt'][0]);action_dt=float(trace['action_dt'][0])
        action_count=len(trace['actions'])
    stride=round(action_dt/physics_dt)
    if abs(action_dt-.02)>1e-8 or abs(stride*physics_dt-action_dt)>1e-8:
        raise ValueError('Canonical gates require the existing 50 Hz control clock')
    if len(rows)!=action_count*stride or rows.shape[1]!=16 or not np.isfinite(rows).all():
        raise ValueError('Incomplete or invalid state-aligned capture')
    sampled=rows[stride-1::stride]
    pitch=np.unwrap(sampled[:,10])
    air=np.flatnonzero((sampled[:,7:9].min(1)>.02)&(sampled[:,9]>.005)).tolist()
    first=next((i for i in range(len(sampled)-2) if all(j in air for j in range(i,i+3))),None)
    end=first
    if first is not None:
        while end+1 in air:end+=1
    window=list(range(max(0,first-8),first)) if first is not None else []
    single=[i for i in range(len(sampled)) if sampled[i,7]<.01 and sampled[i,8]>.04]
    last=next((i for i in reversed(window) if sampled[i,7:9].min()<.01),None)
    support=(len(window)>=4 and last in single and sum(i in single for i in window)>=4
             and all(sampled[i,8]>.04 for i in window))
    tail=sampled[-50:]
    first_rotation=float(pitch[end]-pitch[first]) if first is not None else 0.
    total_rotation=float(pitch.max()-pitch[0])
    gates={
        'single_left_support_at_launch':bool(support),
        'at_least_four_air_frames':len(air)>=4,
        'total_forward_rotation':total_rotation>5.5,
        'first_air_rotation':first_rotation>3.5,
        'inverted_in_air':bool(len(air) and sampled[air,11].min()<-.7),
        'final_upright':bool(sampled[-1,11]>.85),
        'last_one_second_stable':bool(len(tail)==50 and (tail[:,11]>.85).all()
            and (tail[:,0]>.55).all() and (tail[:,5]<.5).all() and (tail[:,6]<1).all()
            and (tail[:,7:9].min(1)<.01).all()),
        'hard_joint_ranges':bool(rows[:,15].max()<1e-4),
        'joint_speed':bool(rows[:,13].max()<=1),
        'motor_effort':bool(rows[:,14].max()<=1.00001),
        'nonfoot_clearance':bool(rows[:,9].min()>.005),
        'foot_compression':bool(rows[:,7:9].min()>-.005),
    }
    duration=len(rows)*physics_dt
    uninterrupted=duration>=7.-1e-8 and not source['failure_physics_steps']
    result={
        'scope':'Canonical optimizer full-motion gates and a full terminal second; independent geometry audit required',
        'capture':str(args.capture),'trajectory_sha256':digest,
        'native_audit_sha256':hashlib.sha256(args.native_audit.read_bytes()).hexdigest(),
        'policy_alone':source.get('policy_alone'),'seconds':duration,
        'uninterrupted_seven_seconds':uninterrupted,'gates':gates,
        'physical_candidate_passed':bool(uninterrupted and all(gates.values())),
        'approved_for_training':False,
        'first_continuous_air_start_s':(first+1)*action_dt if first is not None else None,
        'first_continuous_air_end_s':(end+1)*action_dt if end is not None else None,
        'canonical_maximum_forward_rotation_rad':total_rotation,
        'canonical_first_air_rotation_rad':first_rotation,
        'maximum_hard_violation_rad':float(rows[:,15].max()),
        'maximum_foot_compression_m':float(max(-rows[:,7:9].min(),0)),
        'minimum_nonfoot_height_m':float(rows[:,9].min()),
        'failed_gates':[key for key,value in gates.items() if not value],
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2))
    args.out.with_suffix('.as-run.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
