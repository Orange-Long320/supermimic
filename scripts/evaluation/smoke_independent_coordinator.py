"""Exercise 11500 prepare/request/handover on fixed synthetic observations.

No physics, live sensors, ROS or actuator commands; not a hardware acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from supermimic.deploy.entry import EntryController, EntryPhase, EntryThresholds, QuinticPose
from supermimic.deploy.independent_policy import IndependentPolicyCoordinator
from supermimic.deploy.skill_goal import SkillGoal
from supermimic.deploy.torch_policy import TorchScriptPolicy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('policy','candidate','dataset','profile','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists(): raise ValueError('Output must be new')
    digest=hashlib.sha256(a.policy.read_bytes()).hexdigest()
    if digest!='4424f231fc1896da098b2fca366e8522df6791596638b2b9257ba682520bbe4c': raise ValueError('Wrong policy')
    c=json.loads(a.candidate.read_text()); default=np.asarray(c['action_default']); scale=np.asarray(c['action_scale'])
    q=np.asarray(c['startup']['position']);quat=np.asarray(c['startup']['root_quat_wxyz']);pos=np.asarray(c['startup']['root_position'])
    gravity=np.asarray(c['startup']['gravity']);n=len(q)
    path=QuinticPose(q,q,np.ones(n),np.ones(n),minimum_duration=.1)
    entry=EntryController(path,EntryThresholds(.1,.1,.1,.2,.1,2.,.031,10,.2),np.ones(n),np.ones(n),np.ones(n),np.ones(n))
    torch.set_num_threads(1); policy=TorchScriptPolicy(a.policy)
    coordinator=IndependentPolicyCoordinator(entry,default,scale,policy,lambda p,q:SkillGoal(a.dataset,a.profile,p,q))
    phases=[];targets=[];request_tick=None
    for i in range(75):
        if entry.phase is EntryPhase.HISTORY and len(coordinator.history)==10 and request_tick is None:
            coordinator.request_handover();request_tick=i
        command=coordinator.step(i*.02,pos,quat,q,np.zeros(n),np.zeros(3),gravity,support_ready=True)
        phases.append(entry.phase.value);targets.append(command[0].tolist())
    if request_tick is None or entry.phase is not EntryPhase.ACTIVE: raise ValueError('Synthetic handover failed')
    result={'scope':__doc__,'policy_sha256':digest,'synthetic_fixed_observations':True,'request_tick':request_tick,
            'phases':phases,'max_raw_proposed_step_rad':float(np.max(np.abs(np.diff(targets,axis=0)))),
            'control_transferred':False,'motor_output_enabled':False,'hardware_ready':False}
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
