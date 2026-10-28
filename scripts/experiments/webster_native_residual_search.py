"""Bounded command search in unchanged native simulation, without model gradients.

Only ordinary position actions are corrected. Measurements use separate Torch
FK of post-integration states; the executing simulation is never refreshed.
Search scores are proposals, not full-motion acceptance certificates.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import mujoco
import numpy as np
import torch

from supermimic.data.motion_convert import load_g1_model
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.state_io import StateIO
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.kinematics.rotations import quat_apply
from supermimic.robots.dimensions import NQ, PROFILE
from supermimic.world_model.physical_reward import PhysicalGeometry


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--envs', type=int, default=128)
    p.add_argument('--iterations', type=int, default=12)
    p.add_argument('--max-seconds', type=float, default=300.)
    p.add_argument('--seconds', type=float, default=2.)
    p.add_argument('--sigma', type=float, default=.04)
    p.add_argument('--initialize', type=Path, help='Same-source residual coefficients for local refinement')
    p.add_argument('--terminal-seconds', type=float, default=.3)
    args = p.parse_args()
    if args.out.exists() or args.ckpt.name == 'ckpt_latest.pt':
        p.error('Use a fresh directory and immutable checkpoint')
    if not 8 <= args.envs <= 256 or not 0 <= args.iterations <= 32 or not 1.8 <= args.seconds <= 7.:
        p.error('Invalid bounded native search')
    if not 0 < args.sigma <= .1 or not 0 < args.max_seconds <= 900:
        p.error('Invalid search budget')
    if not .3 <= args.terminal_seconds <= 1. or args.terminal_seconds > args.seconds:
        p.error('Invalid terminal stability window')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.gpu):
        raise ValueError('Explicit physical GPU selection is required')
    query = lambda kind, fields: subprocess.check_output(
        ['nvidia-smi', '-i', str(args.gpu), '--query-'+kind+'='+fields,
         '--format=csv,noheader,nounits'], text=True).strip()
    occupants = query('compute-apps', 'pid')
    free = int(query('gpu', 'memory.free'))
    if occupants or free < 16000:
        raise ValueError('This experiment requires a verified idle GPU with 16 GB headroom')
    digest = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    baseline = json.loads((args.baseline/'result.json').read_text())
    if not baseline['policy_alone'] or baseline['checkpoint_sha256'] != digest(args.ckpt) or baseline['source_sha256'] != digest(args.source):
        raise ValueError('Baseline must be an actual capture of this policy/reference')
    args.out.mkdir(parents=True)
    (args.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.15)
    device = 'cuda:0'
    policy, cfg = load_policy_from_checkpoint(args.ckpt, device)
    policy.requires_grad_(False)
    model = load_g1_model()
    fk = G1Kinematics(model, device)
    geometry = PhysicalGeometry(model, fk, PROFILE['tracked_bodies'], device)
    # Calibrate the exact score geometry before using it for selection.
    with np.load(args.baseline/'trajectory.npz') as z:
        expected = z['metrics'][:, 7:10].copy()
        states = np.c_[z['qpos'], z['qvel']]
    errors = []
    with torch.no_grad():
        for start in range(0, len(states), 64):
            s = torch.tensor(states[start:start+64], dtype=torch.float32, device=device)
            poses = fk(s[:, :NQ])
            h = geometry.heights(poses['pos'], poses['quat']).cpu().numpy()
            errors.append(float(abs(h-expected[start:start+len(s)]).max()))
    if max(errors) >= 1e-5:
        raise ValueError('FK score geometry differs from independent native measurements')
    env = make_env(SuperMimicEnvCfgArgs(npz_files=[str(args.source)], num_envs=args.envs,
        future_frames=cfg['env']['future_frames'], action_noise_std=0., obs_noise=False,
        reset_noise=False, domain_randomization=False, sampling_mode='start',
        record_transitions=False, episode_length_s=1e5,
        anchor_obs=cfg['env'].get('anchor_obs', False), seed=0), device)
    try:
        io = StateIO(env)
        cmd = env.command_manager.get_term('motion')
        original_termination = env.termination_manager.compute
        failure = torch.zeros(args.envs, dtype=torch.bool, device=device)
        def no_reset():
            cmd.clip_end.zero_()
            done = original_termination()
            failure.logical_or_(done)
            return torch.zeros_like(done)
        env.termination_manager.compute = no_reset
        captured = []
        original_step = env.sim.step
        def capture(*a, **kw):
            result = original_step(*a, **kw)
            captured.append(io.read())
            return result
        env.sim.step = capture
        names = list(PROFILE['joint_names'])
        joints = [i for i, n in enumerate(names) if any(s in n for s in ['hip_', 'knee_', 'ankle_', 'waist_'])]
        knots = np.array([.70, .82, .96, 1.10, 1.26, 1.50, 1.70])
        times = np.arange(350)*.02
        basis = np.zeros((350, len(knots)-2))
        for row, t in enumerate(times):
            if knots[0] < t < knots[-1]:
                k = np.searchsorted(knots, t, side='right')-1
                u = (t-knots[k])/(knots[k+1]-knots[k]); u = u*u*(3-2*u)
                if k > 0: basis[row, k-1] += 1-u
                if k+1 < len(knots)-1: basis[row, k] += u
        action_scale = PROFILE['action_scale']
        scale = np.array([action_scale[n] for n in names]) if isinstance(action_scale, dict) else np.full(len(names), action_scale)
        mapping = torch.zeros(len(joints), len(names), device=device)
        for g, j in enumerate(joints):
            mapping[g, j] = PROFILE['effort_limit'][names[j]]/PROFILE['stiffness'][names[j]]/scale[j]
        basis_t = torch.tensor(basis, dtype=torch.float32, device=device)
        speed = torch.tensor([PROFILE['velocity_limit'][n] for n in names], device=device)
        mass = torch.tensor(model.body_mass[1:].copy(), dtype=torch.float32, device=device)
        ipos = torch.tensor(model.body_ipos[1:].copy(), dtype=torch.float32, device=device)
        def state_metrics(s):
            pose = fk(s[..., :NQ], s[..., NQ:])
            h = geometry.heights(pose['pos'], pose['quat'])
            offset = quat_apply(pose['quat'], ipos.expand_as(pose['pos']))
            com = ((pose['pos']+offset)*mass[:, None]).sum(-2)/mass.sum()
            velocity = pose['lin_vel']+torch.cross(pose['ang_vel'], offset, dim=-1)
            velocity = (velocity*mass[:, None]).sum(-2)/mass.sum()
            axis = torch.zeros_like(s[..., :3]); axis[..., 2] = 1
            up = quat_apply(s[..., 3:7], axis)
            return torch.cat([h, torch.atan2(up[..., 0], up[..., 2])[..., None],
                up[..., 2:3], com[..., 2:3], velocity.norm(dim=-1, keepdim=True),
                s[..., NQ+3:NQ+6].norm(dim=-1, keepdim=True),
                geometry.joint_excess(s).max(-1).values[..., None],
                (s[..., NQ+6:].abs()/speed).max(-1).values[..., None]], -1)
        initial_error = None
        @torch.no_grad()
        def rollout(parameters):
            nonlocal initial_error
            cmd.forced_frame[:] = 0
            obs, _ = env.reset()
            failure.zero_()
            state = io.read()
            initial = np.load(args.baseline/'initial-state.npz')['state'][0]
            initial_error = float(abs(state-torch.tensor(initial, device=device)).max())
            if initial_error >= 2e-5: raise ValueError('Original initial states differ; no rewrite permitted')
            residual = torch.einsum('tk,ekg,gj->tej', basis_t, parameters, mapping)
            rows = []
            for step in range(round(args.seconds/.02)):
                captured.clear()
                action = policy(obs['policy'], obs['goal'])+residual[step]
                obs, *_ = env.step(action)
                s = torch.stack(captured, 1)
                rows.append(state_metrics(s))
            return torch.cat(rows, 1).cpu().numpy(), failure.cpu().numpy().copy()
        def score(rows, failed, parameters):
            # Canonical root-up rotation on the original 50 Hz control clock.
            sampled = rows[:, 3::4]
            pitch = np.unwrap(sampled[:, :, 3], axis=1)
            summaries = []
            for i, r in enumerate(rows):
                s = sampled[i]; air = (s[:, :2].min(1)>.02)&(s[:, 2]>.005)
                first = next((j for j in range(len(s)-2) if air[j:j+3].all()), None)
                end = first
                if first is not None:
                    while end+1 < len(s) and air[end+1]: end += 1
                rotation = float(pitch[i,end]-pitch[i,first]) if first is not None else 0.
                total = float(pitch[i].max()-pitch[i,0])
                hard = float(r[:,8].max()); foot = float(max(-r[:,:2].min(),0)); body = float(r[:,2].min())
                tail = s[-round(args.terminal_seconds/.02):]
                support_window = s[max(0,(first or 0)-8):(first or 0)]
                left = (support_window[:,0]<.01)&(support_window[:,1]>.04)
                support = len(left)>=4 and left.sum()>=4 and (support_window[:,1]>.04).all()
                stable = bool((tail[:,4]>.85).all() and (tail[:,5]>.55).all() and (tail[:,6]<.5).all() and (tail[:,7]<1).all())
                core = bool(not failed[i] and support and total>5.5 and air.any() and s[air,4].min()<-.7 and stable)
                loss = (4000*max(hard-.0001,0)+2000*max(foot-.005,0)+2000*max(.005-body,0)
                    +50*max(3.5-rotation,0)+1000*max(float(r[:,9].max())-1,0)
                    +1000*(not core)+.1*float(parameters[i].square().sum()))
                summaries.append(dict(score=float(loss), core_preserved=core, original_failure=bool(failed[i]),
                    hard_rad=hard, foot_m=foot, nonfoot_min_m=body, first_air_rotation_rad=rotation,
                    total_rotation_rad=total, terminal_0p3s_stable=stable, single_left_support=bool(support)))
            return summaries
        dim = (len(knots)-2, len(joints))
        mean = torch.zeros(dim, device=device)
        if args.initialize:
            with np.load(args.initialize) as seed:
                if str(seed['source_checkpoint_sha256']) != digest(args.ckpt) or str(seed['reference_sha256']) != digest(args.source):
                    raise ValueError('Initialization belongs to a different source policy/reference')
                coefficients = seed['parameters'].copy()
                if coefficients.shape != dim or not np.isfinite(coefficients).all() or abs(coefficients).max() > .200001 or not np.array_equal(seed['knot_times'], knots):
                    raise ValueError('Initialization coefficient layout/bounds differ')
                reproduced = torch.einsum('tk,kg,gj->tj', basis_t,
                    torch.tensor(coefficients, dtype=torch.float32, device=device), mapping).cpu().numpy()
                if seed['actions'].shape != reproduced.shape or abs(reproduced-seed['actions']).max() >= 1e-6:
                    raise ValueError('Initialization actions do not match the recorded coefficient basis')
                mean.copy_(torch.tensor(coefficients, device=device))
        std = torch.full(dim, args.sigma, device=device)
        best_parameters = mean.clone(); best_summary = None
        start = time.monotonic(); completed = 0
        for iteration in range(args.iterations+1):
            if iteration and time.monotonic()-start >= args.max_seconds: break
            population = mean+torch.randn(args.envs, *dim, device=device)*std
            population.clamp_(-.2,.2)
            population[0] = best_parameters
            if iteration == 0:
                population[:] = mean
                population[-1].zero_()  # Retain a real unchanged-policy control.
            rows, failed = rollout(population)
            summaries = score(rows, failed, population)
            order = sorted(range(args.envs), key=lambda i:summaries[i]['score'])
            winner = order[0]
            if best_summary is None or summaries[winner]['score'] < best_summary['score']:
                best_summary = summaries[winner]
                best_parameters = population[winner].clone()
                delta = torch.einsum('tk,kg,gj->tj', basis_t, best_parameters, mapping).cpu().numpy()
                np.savez(args.out/'best-residual.npz', actions=delta, fps=np.array([50.]),
                    source_checkpoint_sha256=np.array(digest(args.ckpt)), reference_sha256=np.array(digest(args.source)),
                    parameters=best_parameters.cpu().numpy(), knot_times=knots,
                    joint_names=np.array(names), selected_joint_names=np.array([names[j] for j in joints]))
                (args.out/'best.json').write_text(json.dumps({'iteration':iteration, 'world':winner,
                    **best_summary, 'policy_alone':False, 'physical_candidate_passed':False}, indent=2))
            if iteration:
                elite = population[order[:max(4,args.envs//8)]]
                mean = .2*mean+.8*elite.mean(0)
                std = (.2*std+.8*elite.std(0)).clamp(min=.005)
            entry = {'iteration':iteration,'elapsed_s':time.monotonic()-start,'best':best_summary,
                'population_winner':summaries[winner], 'initial_state_maximum_error':initial_error}
            with (args.out/'progress.jsonl').open('a') as f: f.write(json.dumps(entry)+'\n')
            if iteration == 0:
                np.savez(args.out/'gpu-baseline.npz', metrics=rows, failures=failed)
            print(json.dumps(entry), flush=True)
            completed += 1
        record = {'arguments':vars(args),'completed_populations':completed,'elapsed_s':time.monotonic()-start,
            'score_geometry_maximum_error_m':max(errors),'gpu_occupants_at_start':occupants,
            'free_gpu_mib_at_start':free,'torch_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'torch_peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'best':best_summary,
            'source_sha256':digest(args.source),'checkpoint_sha256':digest(args.ckpt),
            'initialization_sha256':digest(args.initialize) if args.initialize else None,
            'limitation':f'{args.seconds:g}-second native search proposal with live policy feedback; independent seven-second replay required',
            'physical_candidate_passed':False,'approved_for_training':False}
        (args.out/'completed.json').write_text(json.dumps(record,indent=2,default=str))
    finally:
        env.close()


if __name__ == '__main__':
    main()
