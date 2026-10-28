"""Audit local action sensitivities of a frozen world model against native steps.

Independent MjData fixtures and finite differences are diagnostics only, never
executed motion candidates or a validation of long-horizon policy gradients.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from supermimic.data.design_metrics import BatchedDesignMetrics
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
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    if args.ckpt.name=='ckpt_latest.pt':raise ValueError('Use an immutable checkpoint')
    ck=torch.load(args.ckpt,map_location='cpu',weights_only=False)
    cfg=dict(ck['config']['world_model']);cfg['state_spec']=StateSpecCfg(**cfg['state_spec'])
    wm=WorldModel(WorldModelCfg(**cfg),None,'cpu');wm.load_state_dict(ck['wm'])
    wm.eval();wm.requires_grad_(False)
    model=load_g1_model();fk=G1Kinematics(model,'cpu')
    geometry=PhysicalGeometry(model,fk,PROFILE['tracked_bodies'],'cpu')
    native=mujoco.MjModel.from_binary_path(str(args.capture/'native-model.mjb'));data=mujoco.MjData(native)
    names=list(PROFILE['joint_names'])
    if [native.joint(int(j)).name.rsplit('/',1)[-1] for j in native.actuator_trnid[:,0]]!=names:
        raise ValueError('Native actuator order differs from the robot profile')
    ratio=round(.02/native.opt.timestep)
    if abs(ratio*native.opt.timestep-.02)>1e-8:raise ValueError('Unexpected control clock')
    q0=np.array([PROFILE['default_joint_pos'][n] for n in names]);scale=PROFILE['action_scale']
    scale=np.array([scale[n] for n in names]) if isinstance(scale,dict) else np.full(len(names),scale)
    measure=BatchedDesignMetrics(native,'cpu');vertices=measure.vertices.numpy();ids=measure.geom_ids.numpy()
    groups=[group.numpy() for group in measure.groups]
    with np.load(args.capture/'baseline-physics.npz') as physics:initial=np.r_[physics['initial_qpos'],physics['initial_qvel']]
    with np.load(args.capture/'baseline-trajectory.npz') as trace:
        states=np.vstack([initial,np.c_[trace['qpos'],trace['qvel']]]);actions=trace['actions'].copy()
    joint=7+names.index('right_ankle_pitch_joint')
    xi=torch.zeros((1,XI_DIM));xi[0,0]=native.geom_friction[next(i for i in range(native.ngeom) if 'foot_' in native.geom(i).name),0]
    def native_step(state,action):
        mujoco.mj_resetData(native,data);data.qpos[:]=state[:native.nq];data.qvel[:]=state[native.nq:]
        data.ctrl[:]=q0+action*scale;mujoco.mj_forward(native,data)
        for _ in range(ratio):mujoco.mj_step(native,data);mujoco.mj_forward(native,data)
        z=(data.geom_xmat.reshape(-1,3,3)[ids,2]*vertices).sum(-1)+data.geom_xpos[ids,2]
        return np.r_[[z[g].min() for g in groups],data.qpos[joint]],data.qpos.copy()
    def compare(first,second):
        denom=np.linalg.norm(first,axis=1)*np.linalg.norm(second,axis=1)
        return [float(x) if d>1e-12 else None for x,d in zip(
            (first*second).sum(1)/np.maximum(denom,1e-12),denom)]
    windows=[];arrays={}
    for index in [0,4,51,55]:
        state=torch.tensor(states[index:index+1],dtype=torch.float32)
        action=torch.tensor(actions[index:index+1],dtype=torch.float32,requires_grad=True)
        predicted=wm.step(state,action,xi);bodies=fk(predicted[...,:model.nq])
        features=torch.cat([geometry.heights(bodies['pos'],bodies['quat']),predicted[:,joint:joint+1]],-1)[0]
        jacobian=torch.stack([torch.autograd.grad(v,action,retain_graph=True)[0][0] for v in features]).numpy()
        baseline,q=native_step(states[index],actions[index]);differences=[]
        for epsilon in [.002,.01]:
            columns=[]
            for j in range(len(names)):
                perturb=np.zeros(len(names));perturb[j]=epsilon
                high,_=native_step(states[index],actions[index]+perturb)
                low,_=native_step(states[index],actions[index]-perturb)
                columns.append((high-low)/(2*epsilon))
            differences.append(np.asarray(columns).T)
        windows.append({'start_time_s':index*.02,'native_zero_perturbation_qpos_error':float(abs(q-states[index+1,:native.nq]).max()),
            'native_features':baseline.tolist(),'learned_features':features.detach().numpy().tolist(),
            'native_fd_cosine_between_steps':compare(*differences),
            'learned_vs_native_fd_cosine_small_step':compare(jacobian,differences[0]),
            'learned_vs_native_fd_cosine_large_step':compare(jacobian,differences[1]),
            'learned_jacobian_norms':np.linalg.norm(jacobian,axis=1).tolist(),
            'native_small_step_jacobian_norms':np.linalg.norm(differences[0],axis=1).tolist()})
        arrays[f'learned_{index}']=jacobian
        arrays[f'native_small_{index}']=differences[0];arrays[f'native_large_{index}']=differences[1]
    report={'scope':'One control-step local action Jacobians on independent native fixtures; no motion-success or long-horizon gradient certificate',
        'feature_columns':['left_foot_min_m','right_foot_min_m','nonfoot_min_m','right_ankle_pitch_rad'],
        'action_columns':names,'native_finite_difference_steps':[.002,.01],'windows':windows,
        'checkpoint_sha256':hashlib.sha256(args.ckpt.read_bytes()).hexdigest(),
        'capture_sha256':hashlib.sha256((args.capture/'baseline-trajectory.npz').read_bytes()).hexdigest()}
    np.savez(args.out/'jacobians.npz',**arrays)
    (args.out/'result.json').write_text(json.dumps(report,indent=2));(args.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
