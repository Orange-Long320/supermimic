"""Measure world-model errors at Webster's observed early joint-limit failure."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from supermimic.robots.dimensions import PROFILE,XI_DIM
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel,WorldModelCfg


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt',type=Path,required=True)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--horizon',type=int,default=8)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    if args.ckpt.name=='ckpt_latest.pt':raise ValueError('Use an immutable checkpoint')
    ck=torch.load(args.ckpt,map_location='cpu',weights_only=False)
    cfg=dict(ck['config']['world_model']);cfg['state_spec']=StateSpecCfg(**cfg['state_spec'])
    wm=WorldModel(WorldModelCfg(**cfg),None,'cpu');wm.load_state_dict(ck['wm']);wm.eval();wm.requires_grad_(False)
    m=mujoco.MjModel.from_binary_path(str(args.capture/'native-model.mjb'))
    physics=np.load(args.capture/'baseline-physics.npz');control=np.load(args.capture/'baseline-trajectory.npz')
    initial=np.r_[physics['initial_qpos'],physics['initial_qvel']]
    actual=np.vstack([initial,np.c_[control['qpos'],control['qvel']]])
    if not 1<=args.horizon<=len(control['actions']):raise ValueError('Invalid prediction horizon')
    mu=float(m.geom_friction[next(i for i in range(m.ngeom) if 'foot_' in m.geom(i).name),0])
    states=torch.tensor(actual,dtype=torch.float32);actions=torch.tensor(control['actions'],dtype=torch.float32)
    xi=torch.zeros((args.horizon,XI_DIM));xi[:,0]=mu
    with torch.no_grad():
        one=wm.step(states[:args.horizon],actions[:args.horizon],xi).numpy()
        s=states[:1];roll=[]
        for i in range(args.horizon):s=wm.step(s,actions[i:i+1],xi[:1]);roll.append(s[0].numpy().copy())
    roll=np.asarray(roll);target=actual[1:args.horizon+1]
    j=list(PROFILE['joint_names']).index('right_ankle_pitch_joint');index=7+j
    lo,hi=m.jnt_range[1:].T
    violation=lambda q:float(np.maximum(np.maximum(lo-q[:,7:m.nq],q[:,7:m.nq]-hi),0).max())
    report={'scope':'Recorded-action predictions, with true boundaries for one-step and free state rollout for H-step; no policy feedback in this error diagnostic',
        'horizon_s':args.horizon*.02,'context_foot_friction':mu,'right_ankle_pitch_hard_range_rad':[float(lo[j]),float(hi[j])],
        'actual_control_state_hard_violation_rad':violation(target),'teacher_forced_predicted_hard_violation_rad':violation(one),
        'free_rollout_predicted_hard_violation_rad':violation(roll),
        'maximum_one_step_joint_position_error_rad':float(abs(one[:,7:m.nq]-target[:,7:m.nq]).max()),
        'maximum_free_rollout_joint_position_error_rad':float(abs(roll[:,7:m.nq]-target[:,7:m.nq]).max()),
        'right_ankle_rows':[{'time_s':(i+1)*.02,'actual_rad':float(target[i,index]),
            'one_step_predicted_rad':float(one[i,index]),'free_rollout_predicted_rad':float(roll[i,index])} for i in range(args.horizon)],
        'input_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [args.ckpt,args.capture/'baseline-physics.npz',args.capture/'baseline-trajectory.npz']}}
    (args.out/'result.json').write_text(json.dumps(report,indent=2));np.savez(args.out/'predictions.npz',one_step=one,free_rollout=roll,actual=target)
    (args.out/'as-run.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(report))


if __name__=='__main__':main()
