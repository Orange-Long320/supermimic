"""Compare predicted collision heights with a frozen real Webster recording."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from supermimic.data.motion_convert import load_g1_model
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.robots.dimensions import PROFILE,XI_DIM
from supermimic.world_model.physical_reward import PhysicalGeometry
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel,WorldModelCfg


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt',type=Path,required=True)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    if args.ckpt.name=='ckpt_latest.pt':raise ValueError('Use an immutable checkpoint')
    torch.set_num_threads(1)
    ck=torch.load(args.ckpt,map_location='cpu',weights_only=False)
    cfg=dict(ck['config']['world_model']);cfg['state_spec']=StateSpecCfg(**cfg['state_spec'])
    wm=WorldModel(WorldModelCfg(**cfg),None,'cpu');wm.load_state_dict(ck['wm'])
    wm.eval();wm.requires_grad_(False)
    model=load_g1_model();fk=G1Kinematics(model,'cpu')
    geometry=PhysicalGeometry(model,fk,PROFILE['tracked_bodies'],'cpu')
    native=mujoco.MjModel.from_binary_path(str(args.capture/'native-model.mjb'))
    mu=float(native.geom_friction[next(i for i in range(native.ngeom) if 'foot_' in native.geom(i).name),0])
    with np.load(args.capture/'baseline-physics.npz') as physics:
        initial=np.r_[physics['initial_qpos'],physics['initial_qvel']]
    with np.load(args.capture/'trajectory.npz') as trace:
        observed=trace['metrics'][3::4,7:10].copy()
    with np.load(args.capture/'baseline-trajectory.npz') as control:
        states=torch.tensor(np.vstack([initial,np.c_[control['qpos'],control['qvel']]]),dtype=torch.float32)
        actions=torch.tensor(control['actions'],dtype=torch.float32)
    if len(observed)!=len(actions):raise ValueError('Recorded physics and control cadence differ')
    def height(state):
        body=fk(state[...,:model.nq]);return geometry.heights(body['pos'],body['quat'])
    def context(n):
        xi=torch.zeros((n,XI_DIM));xi[:,0]=mu;return xi
    actual=[];predicted=[]
    with torch.no_grad():
        for start in range(0,len(actions),32):
            a=actions[start:start+32];s=states[start:start+len(a)]
            actual.append(height(states[start+1:start+len(a)+1]).numpy())
            predicted.append(height(wm.step(s,a,context(len(a)))).numpy())
        actual=np.vstack(actual);predicted=np.vstack(predicted)
        alignment=float(abs(actual-observed).max())
        if alignment>=1e-5:raise ValueError('Diagnostic geometry differs from native recorded heights')
        error=predicted-actual
        phases={}
        for label,(first,last) in {'early':(0,8),'preparation':(8,35),'flight':(35,50),'landing':(50,75),'standing':(75,len(actions))}.items():
            a=actual[first:last];p=predicted[first:last];e=error[first:last]
            if not len(a):continue
            phases[label]={'actual_minimum_heights_m':a.min(0).tolist(),
                'predicted_minimum_heights_m':p.min(0).tolist(),'mean_signed_height_error_m':e.mean(0).tolist(),
                'maximum_absolute_height_error_m':abs(e).max(0).tolist()}
        rollouts=[]
        for first in [0,32,45,52]:
            if first+8>len(actions):continue
            state=states[first:first+1];values=[]
            for i in range(first,first+8):
                state=wm.step(state,actions[i:i+1],context(1));values.append(height(state)[0].numpy())
            values=np.asarray(values);target=actual[first:first+8]
            rollouts.append({'start_time_s':first*.02,'horizon_s':.16,
                'actual_minimum_heights_m':target.min(0).tolist(),'predicted_minimum_heights_m':values.min(0).tolist(),
                'maximum_absolute_height_error_m':abs(values-target).max(0).tolist()})
    report={'scope':'Recorded actions, true-boundary one-step and eight-step free predictions; no policy-feedback or gradient-accuracy claim',
        'geometry_columns':['left_foot','right_foot','nonfoot'],'geometry_alignment_maximum_error_m':alignment,
        'positive_signed_error_means_predicted_clearance_is_too_high':True,
        'one_step_by_phase':phases,'free_eight_step_windows':rollouts,
        'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.ckpt,args.capture/'trajectory.npz',args.capture/'baseline-trajectory.npz']}}
    np.savez(args.out/'heights.npz',actual=actual,predicted=predicted)
    (args.out/'result.json').write_text(json.dumps(report,indent=2))
    (args.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
