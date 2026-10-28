"""Audit experimental reward geometry and gradients against recorded physics."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from supermimic.data.motion_convert import load_g1_model
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.robots.dimensions import PROFILE
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.physical_reward import PhysicalGeometry,PhysicalMotionHeads,PhysicalRewardCfg
from supermimic.world_model.wm_trainer import reference_at


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--hard-joint-weight',type=float,default=0.)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    model=load_g1_model();fk=G1Kinematics(model,'cpu')
    geometry=PhysicalGeometry(model,fk,PROFILE['tracked_bodies'],'cpu')
    q0=torch.tensor([PROFILE['default_joint_pos'][n] for n in PROFILE['joint_names']])
    library=MotionLibrary([args.reference],tuple(PROFILE['tracked_bodies']),q0,device='cpu')
    z=np.load(args.capture/'trajectory.npz');states=torch.tensor(np.c_[z['qpos'],z['qvel']],dtype=torch.float32)
    expected=z['metrics'][:,7:10];actual=[];bonuses=[];reference_mask=[]
    if not 0<=args.hard_joint_weight<=1000:p.error('Invalid hard-joint reward weight')
    cfg=PhysicalRewardCfg(hard_joint_weight=args.hard_joint_weight)
    with torch.no_grad():hard=geometry.joint_excess(states).max(-1).values.numpy()
    hard_error=float(abs(hard-z['metrics'][:,15]).max())
    if hard_error>=1e-6:raise ValueError('Hard joint ranges differ from native recorded audit')
    hard_gradient=None
    if cfg.hard_joint_weight:
        index=int(hard.argmax());sample=states[index:index+1].clone().requires_grad_(True)
        joint=int(geometry.joint_excess(sample).argmax())
        penalty=-cfg.hard_joint_weight*geometry.joint_excess(sample,cfg.hard_joint_margin_rad).sum()
        derivative=torch.autograd.grad(penalty,sample)[0][0,7+joint]
        expected_gradient=cfg.hard_joint_weight if sample[0,7+joint]<geometry.hard_limits[joint,0] else -cfg.hard_joint_weight
        if abs(float(derivative)-expected_gradient)>1e-6:raise ValueError('Hard joint gradient points out of range')
        hard_gradient={'joint':PROFILE['joint_names'][joint],'gradient':float(derivative)}
    for start in range(0,len(states),64):
        s=states[start:start+64];frame=torch.minimum(torch.arange(start,start+len(s))//4+1,library.clip_last[0])
        ref=library.bodies(frame)
        with torch.no_grad():values=geometry.components(s,ref.pos_w,ref.quat_w,cfg)
        actual.append(values['heights'].numpy());bonuses.append(values['air_rotation'].numpy());reference_mask.append(values['reference_air'].numpy())
    actual=np.vstack(actual);bonuses=np.concatenate(bonuses);reference_mask=np.concatenate(reference_mask)
    error=float(abs(actual-expected).max())
    if error>=1e-5:raise ValueError(f'FK collision heights differ from native capture: {error}')
    peak=int(np.argmin(expected[:,:2].min(1)))
    s=states[peak:peak+1].clone().requires_grad_(True)
    frame=torch.minimum(torch.tensor([peak//4+1]),library.clip_last[0]);ref=library.bodies(frame)
    values=geometry.components(s,ref.pos_w,ref.quat_w,cfg)
    gradient=torch.autograd.grad(values['clearance'].sum(),s)[0]
    if not torch.isfinite(gradient).all() or float(gradient[0,2])<=0:raise ValueError('Invalid clearance gradient')
    shifted=s.detach().clone();shifted[:,2]+=.01
    with torch.no_grad():translated=geometry.components(shifted,ref.pos_w,ref.quat_w,cfg)
    translation_error=float(abs(translated['heights']-values['heights'].detach()-.01).max())
    if translation_error>=1e-6:raise ValueError('Collision height translation check failed')
    ids=torch.linspace(0,len(states)-1,8).long();selected=states[ids]
    frame=torch.minimum(ids//4+1,library.clip_last[0]);refs=reference_at(library,frame)
    centre=model.jnt_range[1:].mean(1);half=.45*np.diff(model.jnt_range[1:],axis=1).ravel()
    limits=torch.tensor(np.c_[centre-half,centre+half],dtype=torch.float32)
    heads=AnalyticHeads(fk,tuple(PROFILE['tracked_bodies']),PROFILE['anchor_body'],PROFILE['root_body'],limits,q0)
    zero=PhysicalMotionHeads(heads,geometry,PhysicalRewardCfg(foot_weight=0.,nonfoot_weight=0.,air_rotation_weight=0.))
    actions=torch.zeros(len(ids),len(PROFILE['joint_names']))
    with torch.no_grad():identity_error=float(abs(heads.reward(selected,actions,actions,refs)['total']-zero.reward(selected,actions,actions,refs)['total']).max())
    if identity_error!=0:raise ValueError('Zero-weight reward changes original objective')
    if np.any(bonuses[reference_mask==0]!=0):raise ValueError('Air-turn bonus leaks outside reference flight')
    # This is a static manager-interface test on a separate audit environment,
    # not a motion rollout. Fixture writes must never count as motion success.
    from types import SimpleNamespace
    from supermimic.envs import SuperMimicEnvCfgArgs,make_env
    from supermimic.envs.state_io import StateIO
    from supermimic.world_model.physical_reward import install_physical_reward
    env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(args.reference)],num_envs=1,
        action_noise_std=0.,obs_noise=False,reset_noise=False,domain_randomization=False,
        sampling_mode='start',record_transitions=False,anchor_obs=True),'cpu')
    manager_error=0.;manager_values=[]
    try:
        env.reset();robot=env.scene['robot']
        base=AnalyticHeads(fk,tuple(PROFILE['tracked_bodies']),PROFILE['anchor_body'],PROFILE['root_body'],
            robot.data.soft_joint_pos_limits[0].clone(),robot.data.default_joint_pos[0].clone())
        trainer=SimpleNamespace(env=env,wm=SimpleNamespace(heads=base),device='cpu',max_step_reward=base.reward_cfg.max_step_reward)
        install_physical_reward(trainer,model,cfg)
        io=StateIO(env);cmd=env.command_manager.get_term('motion')
        term_index=env.reward_manager._term_names.index('physical_motion')
        for index in sorted({0,19,159,225,245,len(states)-1}):
            io.write(torch.tensor([0]),states[index:index+1]);env.sim.sense()
            cmd.frame[:]=min(index//4+1,int(cmd.library.clip_last[0]));cmd._refresh_ref();cmd.update_relative_body_poses()
            env.reward_manager.compute(env.step_dt)
            real=env.reward_manager._step_reward[0,term_index]*env.step_dt
            reference=reference_at(cmd.library,cmd.frame)
            imagined=trainer.wm.heads.reward(io.read(),actions[:1],actions[:1],reference)['physical_motion'][0]
            manager_error=max(manager_error,float(abs(real-imagined)))
            manager_values.append(float(real))
    finally:env.close()
    if manager_error>=1e-6:raise ValueError('Real and imagined physical reward scaling differs')
    report={'native_geometry_maximum_error_m':error,'root_height_translation_error_m':translation_error,
        'physical_reward_cfg':asdict(cfg),'hard_violation_maximum_error_rad':hard_error,'hard_joint_gradient':hard_gradient,
        'clearance_gradient_root_z':float(gradient[0,2]),'zero_weight_original_reward_maximum_error':identity_error,
        'hull_vertices':len(geometry.points),'physical_samples':len(states),'maximum_air_bonus_per_second':float(bonuses.max()),
        'reference_flight_samples':int(reference_mask.sum()),'passed':True,'physical_candidate_passed':False,
        'static_manager_fixture_maximum_error':manager_error,'static_manager_fixture_values':manager_values,
        'recorded_state_writes_in_separate_static_audit_only':True,
        'scope':'Reward geometry, differentiability and zero-weight compatibility only; no motion-success certificate',
        'sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [args.reference,args.capture/'trajectory.npz']}}
    (args.out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
    (args.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    import supermimic.world_model.physical_reward as module
    (args.out/'physical_reward.py').write_bytes(Path(module.__file__).read_bytes())


if __name__=='__main__':main()
