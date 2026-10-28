"""Propose early command pulses while predicting the entire measured preparation.

Native CPU previews include the frozen policy feedback. Proposals still require
independent native Warp replay and unchanged full physical acceptance gates.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.optimize import minimize

from supermimic.data.native_policy_preview import NativePolicyPreview
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.robots.dimensions import PROFILE


class BudgetReached(Exception):
    pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--ckpt', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--trust', type=float, default=.08)
    p.add_argument('--iterations', type=int, default=8)
    p.add_argument('--max-seconds', type=float, default=180)
    p.add_argument('--seed', type=Path, help='Previously native-tested expanded parameter candidate; baseline calibration is still checked independently')
    p.add_argument('--all-preparation-commands', action='store_true', help='Allow later preparation commands to compensate for early contact corrections')
    p.add_argument('--finite-difference-step', type=float, default=1e-3)
    args = p.parse_args()
    if not 0<args.trust<=.2 or args.iterations<1 or args.max_seconds<=0 or args.ckpt.name=='ckpt_latest.pt' or not 0<args.finite_difference_step<=.01:
        p.error('Use an immutable checkpoint and bounded positive search settings')
    args.out.mkdir(parents=True, exist_ok=False); torch.set_num_threads(1)
    digest = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    config = json.loads((args.trace/'configuration.json').read_text()); a = config['arguments']
    calibration = json.loads(args.calibration.read_text())
    source = Path(a['source']); solution = Path(a['warmstart'])
    for path in [args.ckpt, source, solution]:
        if calibration['input_sha256'].get(str(path))!=digest(path):
            raise ValueError('Full-preparation calibration does not match the proposal inputs')
    model = mujoco.MjModel.from_binary_path(str(args.trace/'native-model.mjb'))
    if digest(args.trace/'native-model.mjb')!=calibration['native_model_sha256']:
        raise ValueError('Calibrated native model changed')
    unsupported = [k for k in ['tail_of','landing_ckpt','consistent_retiming','initial_command_margin',
        'bounded_target_joints','joint_stop_margin','physics_command_projection','native_contact_preview'] if a.get(k)]
    if unsupported:raise ValueError(f'Unsupported captured controllers: {unsupported}')
    original = np.load(solution)['parameters']; count = len(config['groups'])
    if original.shape!=(1+4*count,):raise ValueError('Expected preparation with an initial knot; pulse expansion must start explicitly from zero')
    expanded = np.r_[original, np.zeros(2*count)]
    policy, policy_config = load_policy_from_checkpoint(args.ckpt, 'cpu'); policy.requires_grad_(False)
    q0 = torch.tensor([PROFILE['default_joint_pos'][n] for n in PROFILE['joint_names']])
    library = MotionLibrary([source], tuple(PROFILE['tracked_bodies']), q0, device='cpu')
    observations = np.load(args.trace/'baseline-observations.npz')
    physics = np.load(args.trace/'baseline-physics.npz')
    preview = NativePolicyPreview(model,policy,library,config['groups'],expanded,a['takeoff_frame'],a['landing_frame'],
        observations['reference_frame'][0],policy_config['env'].get('anchor_obs',False),policy_config['env'].get('future_frames',10),True)
    horizon = len(observations['policy'])
    if calibration.get('horizon_control_frames')!=horizon or not calibration.get('reference_clock_matched'):
        raise ValueError('Calibration must cover the complete measured preparation')
    if calibration['maximum_errors_by_horizon'][-1]['geometry_height_m']>=2e-5:
        raise ValueError('Full-preparation calibration is insufficient')
    labels = [group[0][0] for group in config['groups']]
    selected_groups = [i for i,n in enumerate(labels) if any(s in n for s in ['hip_','knee_','ankle_'])]
    columns = [1+4*count+k*count+g for k in range(2) for g in selected_groups]
    if args.all_preparation_commands:
        columns=[1+k*count+g for k in range(3) for g in selected_groups]+[1+3*count+g for g in selected_groups]+columns
    seed=np.zeros(len(columns))
    if args.seed:
        candidate=np.load(args.seed)['parameters']
        if candidate.shape!=expanded.shape:raise ValueError('Seed must use the explicit expanded parameter layout')
        frozen=np.ones(len(expanded),dtype=bool);frozen[columns]=False
        if not np.array_equal(candidate[frozen],expanded[frozen]):raise ValueError('Seed changes frozen parameters')
        seed=candidate[columns]-expanded[columns]
        if abs(seed).max()>args.trust:raise ValueError('Seed lies outside the requested trust region')
    lo, hi = model.jnt_range[1:].T
    speed = np.array([PROFILE['velocity_limit'][n] for n in PROFILE['joint_names']])
    root = int(model.jnt_bodyid[0]); nq = model.nq
    start = time.monotonic(); cache = {}; best = None; evaluations = 0
    (args.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    import supermimic.data.native_policy_preview as module
    (args.out/'native_policy_preview.py').write_bytes(Path(module.__file__).read_bytes())
    (args.out/'inputs.json').write_text(json.dumps({'trace':str(args.trace),'ckpt':str(args.ckpt),'calibration':str(args.calibration),
        'calibration_sha256':digest(args.calibration),'expanded_from_dimensions':[len(original),len(expanded)],
        'seed_sha256':digest(args.seed) if args.seed else None,
        'columns':columns,'groups':config['groups'],'horizon':horizon,'arguments':vars(args)},indent=2,default=str))

    def evaluate(x):
        nonlocal best, evaluations
        key=x.tobytes()
        if key in cache:return cache[key]
        if evaluations and time.monotonic()-start>args.max_seconds:raise BudgetReached()
        parameters=expanded.copy();parameters[columns]+=x
        preview.parameters=torch.tensor(parameters,dtype=torch.float32)
        result=preview.predict(physics['initial_qpos'],physics['initial_qvel'],observations['policy'][0],0.,horizon=horizon)
        states=result['states'];q=states[:,7:nq];height=result['heights']
        mujoco.mj_subtreeVel(model,preview.data)
        com=preview.data.subtree_com[root];velocity=preview.data.subtree_linvel[root]
        apex=float(com[2]+max(velocity[2],0)**2/(2*abs(model.opt.gravity[2])))
        momentum=float(preview.data.subtree_angmom[root,1])
        hard=max(float((lo-q).max()),float((q-hi).max()),0.)
        foot=max(-float(height[:,:2].min()),0.);body=float(height[:,2].min())
        maximum_speed=float((abs(states[:,nq+6:])/speed).max())
        eligible=apex>=.83 and velocity[2]>=.2 and momentum>=30 and height[-1,:2].min()>=.05
        quality=(not eligible,hard>=1e-4,max(hard-1e-4,0),body<.005,max(.005-body,0),
            maximum_speed>1,max(maximum_speed-1,0),foot>=.005,max(foot-.005,0),float(x@x))
        constraints=np.r_[((height[:,:2]+.004)/.005).ravel(),((height[:,2]-.006)/.01).ravel(),
            ((q-lo-.0005)/.05).ravel(),((hi-.0005-q)/.05).ravel(),
            (1-abs(states[:,nq+6:])/speed).ravel(),(apex-.83)/.1,(velocity[2]-.2),
            (momentum-30)/10,(height[-1,:2]-.05)/.1]
        summary={'evaluation':evaluations,'elapsed_s':time.monotonic()-start,'predicted_maximum_hard_rad':hard,
            'predicted_maximum_foot_compression_m':foot,'predicted_minimum_nonfoot_height_m':body,
            'predicted_maximum_speed_ratio':maximum_speed,'terminal_apex_estimate_m':apex,
            'terminal_vz_m_s':float(velocity[2]),'terminal_forward_momentum':momentum,
            'terminal_preservation_eligible':bool(eligible),'command_delta_norm':float(np.linalg.norm(x)),
            'pulse_norm':float(np.linalg.norm(parameters[1+4*count:])),'approved_for_training':False}
        cache[key]=(constraints,quality);evaluations+=1
        if best is None or quality<best[0]:
            best=(quality,parameters,summary)
            np.savez(args.out/'best-parameters.npz',parameters=parameters)
            (args.out/'best.json').write_text(json.dumps(summary,indent=2))
        with (args.out/'progress.jsonl').open('a') as f:f.write(json.dumps(summary)+'\n')
        return cache[key]

    zero=np.zeros(len(columns));evaluate(zero)
    # Zero-pulse expansion must reproduce the complete calibration trajectory.
    baseline=preview.predict(physics['initial_qpos'],physics['initial_qvel'],observations['policy'][0],0.,horizon=horizon)
    error=float(abs(baseline['states'][:,:nq]-physics['qpos']).max())
    if error>=2e-5:raise ValueError(f'Expanded baseline differs from native capture: {error}')
    budget_hit=False;result=None
    try:
        lower=np.maximum(-args.trust,-1.5-expanded[columns]);upper=np.minimum(args.trust,1.5-expanded[columns])
        result=minimize(lambda x:.5*float(x@x),seed,jac=lambda x:x,method='SLSQP',
            bounds=list(zip(lower,upper)),constraints=[{'type':'ineq','fun':lambda x:evaluate(x)[0]}],
            options={'maxiter':args.iterations,'eps':args.finite_difference_step,'ftol':1e-8})
    except BudgetReached:budget_hit=True
    report={'evaluations':evaluations,'elapsed_s':time.monotonic()-start,'budget_hit':budget_hit,
        'optimizer_success':bool(result.success) if result is not None else False,
        'optimizer_message':str(result.message) if result is not None else 'Preview time budget reached',
        'expanded_baseline_qpos_error':error,'best':best[2],
        'scope':'Full measured preparation forecast only; independent native replay and unchanged full-motion gates required',
        'approved_for_training':False}
    (args.out/'completed.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
