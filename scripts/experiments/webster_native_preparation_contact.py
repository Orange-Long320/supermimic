"""Small CPU-native preparation contact proposals with frozen policy feedback.

The complete 0.7-second forecast must reproduce the measured baseline first.
Proposals only adjust normal commands; full independent execution is required.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import minimize
import torch

from supermimic.data.native_policy_preview import NativePolicyPreview
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.robots.dimensions import PROFILE


class BudgetReached(Exception):
    pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-seconds', type=float, default=180.)
    p.add_argument('--iterations', type=int, default=8)
    args = p.parse_args()
    if args.out.exists() or args.ckpt.name == 'ckpt_latest.pt' or not 0 < args.max_seconds <= 600 or not 1 <= args.iterations <= 32:
        p.error('Use a fresh output, immutable policy and bounded search')
    torch.set_num_threads(1)
    digest = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    info = json.loads((args.capture/'result.json').read_text())
    source = Path(info['arguments']['source'])
    if not info['policy_alone'] or info['checkpoint_sha256'] != digest(args.ckpt) or info['source_sha256'] != digest(source):
        raise ValueError('Expected the matching actual pure-policy capture')
    args.out.mkdir(parents=True)
    (args.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    model = mujoco.MjModel.from_binary_path(str(args.capture/'native-model.mjb'))
    policy, cfg = load_policy_from_checkpoint(args.ckpt, 'cpu')
    policy.requires_grad_(False)
    names = list(PROFILE['joint_names'])
    q0 = torch.tensor([PROFILE['default_joint_pos'][n] for n in names])
    library = MotionLibrary([source], tuple(PROFILE['tracked_bodies']), q0, device='cpu')
    observations = np.load(args.capture/'baseline-observations.npz')
    physics = np.load(args.capture/'baseline-physics.npz')
    trajectory = np.load(args.capture/'trajectory.npz')
    preview = NativePolicyPreview(model, policy, library, [], None, None, None,
        observations['reference_frame'][0], cfg['env'].get('anchor_obs', False), cfg['env']['future_frames'])
    horizon = 35
    initial_q = physics['initial_qpos']; initial_v = physics['initial_qvel']
    initial_obs = observations['policy'][0]
    baseline = preview.predict(initial_q, initial_v, initial_obs, 0., horizon=horizon)
    actual = np.c_[physics['qpos'][:horizon*4], physics['qvel'][:horizon*4]]
    state_error = np.abs(baseline['states']-actual)
    calibration = {'qpos_maximum_error':float(state_error[:,:model.nq].max()),
        'qvel_maximum_error':float(state_error[:,model.nq:].max()),
        'height_maximum_error_m':float(abs(baseline['heights']-trajectory['metrics'][:horizon*4,7:10]).max())}
    (args.out/'calibration.json').write_text(json.dumps(calibration,indent=2))
    if calibration['qpos_maximum_error'] >= 2e-5 or calibration['qvel_maximum_error'] >= 1e-3 or calibration['height_maximum_error_m'] >= 2e-5:
        raise ValueError('Complete preparation forecast is insufficient; no search performed')
    joints = [j for j,n in enumerate(names) if any(s in n for s in ['hip_pitch','knee','ankle_pitch'])]
    knots = np.array([-.02,0.,.06,.12,.22,.40,.60,.70])
    basis = np.zeros((350,len(knots)-2))
    for i,t in enumerate(np.arange(350)*.02):
        if knots[0] < t < knots[-1]:
            k = np.searchsorted(knots,t,side='right')-1
            u = (t-knots[k])/(knots[k+1]-knots[k]);u=u*u*(3-2*u)
            if k>0:basis[i,k-1]+=1-u
            if k+1<len(knots)-1:basis[i,k]+=u
    scale = PROFILE['action_scale']
    scale = np.array([scale[n] for n in names]) if isinstance(scale,dict) else np.full(len(names),scale)
    mapping = np.zeros((len(joints),len(names)))
    for g,j in enumerate(joints):
        mapping[g,j] = PROFILE['effort_limit'][names[j]]/PROFILE['stiffness'][names[j]]/scale[j]
    shape = (basis.shape[1],len(joints)); lo,hi=model.jnt_range[1:].T
    speed=np.array([PROFILE['velocity_limit'][n] for n in names])
    start=time.monotonic();cache={};best=None;evaluations=0
    def evaluate(x):
        nonlocal best,evaluations
        key=x.tobytes()
        if key in cache:return cache[key]
        if evaluations and time.monotonic()-start>args.max_seconds:raise BudgetReached()
        residual=basis@x.reshape(shape)@mapping
        predicted=preview.predict(initial_q,initial_v,initial_obs,0.,horizon=horizon,correction=residual[:horizon])
        state=predicted['states'];q=state[:,7:model.nq];h=predicted['heights']
        hard=float(np.maximum(np.maximum(lo-q,q-hi),0).max())
        foot=float(max(-h[:,:2].min(),0));body=float(h[:,2].min())
        terminal=state[-1]-baseline['states'][-1]
        qdelta=float(abs(terminal[:model.nq]).max());vdelta=float(abs(terminal[model.nq:]).max())
        eligible=qdelta<=.03 and vdelta<=.5 and body>.005 and float((abs(state[:,model.nq+6:])/speed).max())<=1
        quality=(not eligible,hard>=1e-4,max(hard-1e-4,0),max(foot-.005,0),float(x@x))
        constraints=np.r_[((h[:,:2]+.0048)/.005).ravel(),
            ((q-lo+5e-5)/.05).ravel(),((hi+5e-5-q)/.05).ravel(),
            (h[:,2]-.006)/.02,(1-abs(state[:,model.nq+6:])/speed).ravel(),
            (.03-terminal[:model.nq])/.03,(.03+terminal[:model.nq])/.03,
            (.5-terminal[model.nq:])/.5,(.5+terminal[model.nq:])/.5]
        summary={'evaluation':evaluations,'elapsed_s':time.monotonic()-start,'hard_rad':hard,
            'foot_compression_m':foot,'nonfoot_min_m':body,'terminal_qpos_maximum_change':qdelta,
            'terminal_qvel_maximum_change':vdelta,'terminal_preserved':bool(eligible),'parameter_norm':float(np.linalg.norm(x)),
            'scope':'Native forecast proposal only; actual full7s execution required','approved_for_training':False}
        if best is None or quality<best[0]:
            best=(quality,summary)
            np.savez(args.out/'best-residual.npz',actions=residual.astype(np.float32),fps=np.array([50.]),
                source_checkpoint_sha256=np.array(digest(args.ckpt)),reference_sha256=np.array(digest(source)),
                parameters=x.reshape(shape),knot_times=knots,joint_names=np.array(names),
                selected_joint_names=np.array([names[j] for j in joints]))
            (args.out/'best.json').write_text(json.dumps(summary,indent=2))
        with (args.out/'progress.jsonl').open('a') as f:f.write(json.dumps(summary)+'\n')
        evaluations+=1;cache[key]=constraints
        return constraints
    zero=np.zeros(np.prod(shape));evaluate(zero)
    result=None;budget=False
    try:
        result=minimize(lambda x:.5*float(x@x),zero,jac=lambda x:x,method='SLSQP',
            bounds=[(-.12,.12)]*len(zero),constraints=[{'type':'ineq','fun':evaluate}],
            options={'maxiter':args.iterations,'eps':1e-3,'ftol':1e-8})
    except BudgetReached:budget=True
    report={'arguments':vars(args),'evaluations':evaluations,'elapsed_s':time.monotonic()-start,
        'budget_hit':budget,'optimizer_success':bool(result.success) if result is not None else False,
        'optimizer_message':str(result.message) if result is not None else 'Budget reached',
        'calibration':calibration,'best':best[1],'source_sha256':digest(source),'checkpoint_sha256':digest(args.ckpt),
        'physical_candidate_passed':False,'approved_for_training':False}
    (args.out/'completed.json').write_text(json.dumps(report,indent=2,default=str));print(json.dumps(report,default=str),flush=True)


if __name__=='__main__':
    main()
