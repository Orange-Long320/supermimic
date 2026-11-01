"""Offline single-step input substitution at the largest simulated target jump.

Interventions hold all other logged inputs fixed; effects do not add up, and
are neither physical interventions nor proof of a dynamic closed-loop cause.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from supermimic.deploy.torch_policy import TorchScriptPolicy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('frames','candidate','policy','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('Output already exists')
    if hashlib.sha256(a.policy.read_bytes()).hexdigest()!='4424f231fc1896da098b2fca366e8522df6791596638b2b9257ba682520bbe4c':
        raise ValueError('Wrong policy')
    c=json.loads(a.candidate.read_text());names=c['joint_names'];scale=np.asarray(c['action_scale']);default=np.asarray(c['action_default'])
    rows=[json.loads(line) for line in a.frames.open()];rows=[r for r in rows if r['phase']=='active' and 'policy_trace' in r]
    if len(rows)<2 or len(names)!=31:raise ValueError('Insufficient logged policy frames')
    targets=np.asarray([r['target'] for r in rows]);k,j=np.unravel_index(np.abs(np.diff(targets,axis=0)).argmax(),(len(rows)-1,31))
    old,new=rows[k]['policy_trace'],rows[k+1]['policy_trace'];o0=np.asarray(old['observation']);o1=np.asarray(new['observation'])
    g0=np.asarray(old['goal']);g1=np.asarray(new['goal'])
    if o0.shape!=(990,) or o1.shape!=(990,) or g0.shape!=(729,) or g1.shape!=(729,):raise ValueError('Trace shape')
    torch.set_num_threads(1);policy=TorchScriptPolicy(a.policy)
    def predict(o,g):return default+scale*np.asarray(policy.infer(tuple(o),tuple(g)))
    y0=predict(o0,g0);y1=predict(o1,g1)
    if max(np.max(np.abs(y0-targets[k])),np.max(np.abs(y1-targets[k+1])))>1e-5:
        raise ValueError('Logged target does not match policy replay')
    blocks={'joint_position':(0,310),'joint_velocity':(310,620),'angular_velocity':(620,650),
            'gravity':(650,680),'previous_action':(680,990)}
    cases={}
    for name,(start,end) in blocks.items():
        intervention=o1.copy();intervention[start:end]=o0[start:end]
        y=predict(intervention,g1);cases[name]={'target_with_previous_block_rad':float(y[j]),
             'change_from_current_rad':float(y[j]-y1[j]),'input_max_change':float(np.max(np.abs(o1[start:end]-o0[start:end])))}
    for name,start,end in [('reference_window',0,720),('pelvis_anchor',720,729)]:
        intervention=g1.copy();intervention[start:end]=g0[start:end]
        y=predict(o1,intervention);cases[name]={'target_with_previous_block_rad':float(y[j]),
             'change_from_current_rad':float(y[j]-y1[j]),'input_max_change':float(np.max(np.abs(g1[start:end]-g0[start:end])))}
    result={'scope':__doc__,'time_before_s':rows[k]['time_s'],'time_after_s':rows[k+1]['time_s'],'joint':names[j],
            'target_before_rad':float(y0[j]),'target_after_rad':float(y1[j]),'cases':cases,
            'policy_replay_matches_log':True,'hardware_output':False}
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
