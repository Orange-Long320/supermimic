"""Collect native counterfactual transitions for local dynamics calibration.

Each sample starts in an independent recorded-state fixture. These are model
supervision samples, not a continuous motion or a successful Webster teacher.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    from supermimic.robots.dimensions import PROFILE
    model=mujoco.MjModel.from_binary_path(str(args.capture/'native-model.mjb'));data=mujoco.MjData(model)
    names=list(PROFILE['joint_names'])
    if [model.joint(int(j)).name.rsplit('/',1)[-1] for j in model.actuator_trnid[:,0]]!=names:
        raise ValueError('Native actuator order differs from the robot profile')
    ratio=round(.02/model.opt.timestep)
    if abs(ratio*model.opt.timestep-.02)>1e-8:raise ValueError('Unexpected control clock')
    q0=np.array([PROFILE['default_joint_pos'][n] for n in names]);scale=PROFILE['action_scale']
    scale=np.array([scale[n] for n in names]) if isinstance(scale,dict) else np.full(len(names),scale)
    with np.load(args.capture/'baseline-physics.npz') as physics:initial=np.r_[physics['initial_qpos'],physics['initial_qvel']]
    with np.load(args.capture/'baseline-trajectory.npz') as trace:
        states=np.vstack([initial,np.c_[trace['qpos'],trace['qvel']]]);actions=trace['actions'].copy()
    if len(actions)!=350:raise ValueError('This calibration uses the complete seven-second baseline')
    indices=np.array(sorted(set(range(76))|set(range(80,len(actions),10))))
    validation={4,11,18,25,32,39,46,53,55,60,67,74,90,140,190,240,290,340}
    training=np.array([int(i) not in validation for i in indices])
    rng=np.random.default_rng(args.seed);epsilon=.01
    base=[];plus=[];minus=[];directions=[];errors=[];started=time.monotonic()
    def step(state,action):
        mujoco.mj_resetData(model,data);data.qpos[:]=state[:model.nq];data.qvel[:]=state[model.nq:]
        data.ctrl[:]=q0+action*scale;mujoco.mj_forward(model,data)
        for _ in range(ratio):mujoco.mj_step(model,data);mujoco.mj_forward(model,data)
        return np.r_[data.qpos,data.qvel].copy()
    for count,index in enumerate(indices):
        state=states[index];action=actions[index];next_state=step(state,action)
        errors.append([float(abs(next_state[:model.nq]-states[index+1,:model.nq]).max()),
            float(abs(next_state[model.nq:]-states[index+1,model.nq:]).max())])
        random=rng.normal(size=(8,len(names)));random/=np.max(abs(random),axis=1,keepdims=True)
        direction=np.r_[np.eye(len(names)),random]
        base.append(next_state);directions.append(direction)
        plus.append([step(state,action+epsilon*d) for d in direction])
        minus.append([step(state,action-epsilon*d) for d in direction])
        if count%10==0:print(json.dumps({'collected_states':count+1,'total_states':len(indices),'elapsed_s':time.monotonic()-started}),flush=True)
    errors=np.asarray(errors)
    passed=bool(errors[:,0].max()<1e-5 and errors[:,1].max()<1e-3)
    report={'scope':'Independent native recorded-state fixtures for model supervision; not a continuous motion or approved teacher',
        'robot':'x2_ultra','state_count':len(indices),'directions_per_state':len(names)+8,
        'training_states':int(training.sum()),'validation_states':int((~training).sum()),
        'holdout_is_by_recorded_state_not_just_action':True,'epsilon':epsilon,
        'native_zero_perturbation_maximum_qpos_error':float(errors[:,0].max()),
        'native_zero_perturbation_maximum_qvel_error':float(errors[:,1].max()),
        'source_alignment_passed':passed,'physical_candidate_passed':False,'approved_for_training':False,
        'context_foot_friction':float(model.geom_friction[next(i for i in range(model.ngeom) if 'foot_' in model.geom(i).name),0]),
        'elapsed_s':time.monotonic()-started,
        'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.capture/'native-model.mjb',args.capture/'baseline-physics.npz',args.capture/'baseline-trajectory.npz']}}
    np.savez_compressed(args.out/'transitions.npz',states=states[indices],actions=actions[indices],
        base=np.asarray(base),plus=np.asarray(plus),minus=np.asarray(minus),directions=np.asarray(directions),
        epsilon=np.array([epsilon]),training=training,indices=indices,baseline_alignment_errors=errors)
    report['dataset_sha256']=hashlib.sha256((args.out/'transitions.npz').read_bytes()).hexdigest()
    (args.out/'result.json').write_text(json.dumps(report,indent=2));(args.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report),flush=True)
    if not passed:raise ValueError('Native fixture reproduction failed; calibration data is unusable')


if __name__=='__main__':main()
