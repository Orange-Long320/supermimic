"""Server-only controlled audit: fixed actor and held-out dynamics predictions.

Compare motion-only and mixed-interface dynamics refits from one checkpoint.
This diagnoses fit/coverage; it does not update or certify a deployed controller.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'checkpoint', 'later_checkpoint', 'dataset', 'out'):
        p.add_argument('--' + key.replace('_', '-'), type=Path, required=True)
    p.add_argument('--steps', type=int, default=700)
    p.add_argument('--num-envs', type=int, default=64)
    p.add_argument('--fit-steps', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--windows-dir', type=Path, help='Reuse immutable train and held-out trajectories')
    p.add_argument('--fit-arms', nargs='+', choices=('motion_only', 'mixed'), default=['motion_only', 'mixed'])
    p.add_argument('--fit-window', type=int, choices=(8, 32), default=8)
    p.add_argument('--heldout-dir', type=Path, help='Keep validation trajectories fixed while changing training coverage')
    p.add_argument('--velocity-loss-weight', type=float, default=1.)
    a = p.parse_args()
    if not (64 <= a.steps <= 1500 and 8 <= a.num_envs <= 512 and 1 <= a.fit_steps <= 512 and 1 <= a.velocity_loss_weight <= 10):
        raise ValueError('Outside bounded audit limits')
    expected = Path('/home/2025-Long.xh/HHD/venvs/supermimic').resolve()
    if Path(sys.prefix).resolve() != expected or len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) != 1:
        raise ValueError('Use the dedicated server interpreter and one explicit GPU')
    a.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.source / 'src'))
    import torch
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.locomotion.config import VelocityCfg
    from supermimic.locomotion.task import measured_velocity
    from supermimic.world_model.state_spec import StateSpecCfg
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg
    from mjlab.utils.torch import configure_torch_backends

    torch.set_num_threads(2)
    # make_env configures TF32. Offline-window runs must use the same backend.
    configure_torch_backends()
    torch.cuda.set_per_process_memory_fraction(.65, 0)
    policy, saved = load_policy_from_checkpoint(a.checkpoint, 'cuda:0')
    policy.requires_grad_(False)
    ck = torch.load(a.checkpoint, map_location='cuda:0', weights_only=False)
    config = copy.deepcopy(saved['world_model'])
    config['state_spec'] = StateSpecCfg(**config['state_spec'])
    wm = WorldModel(WorldModelCfg(**config), None, 'cuda:0')
    provenance = dict(argv=sys.argv, pid=os.getpid(), gpu=os.environ['CUDA_VISIBLE_DEVICES'],
                      checkpoint=str(a.checkpoint), checkpoint_sha256=digest(a.checkpoint),
                      later_checkpoint=str(a.later_checkpoint), later_sha256=digest(a.later_checkpoint),
                      manifest_sha256=digest(a.dataset / 'manifest.json'),
                      saved_iteration=ck['iteration'], saved_updates=ck['num_updates'],
                      matmul_precision=torch.backends.cuda.matmul.fp32_precision,
                      protocol=f'Fixed actor. Separate seed 0/7 simulator runs; DR, reset/observation/action noise enabled. No reset-crossing windows. Refit only dynamics on {a.fit_window}-step state loss; held-out 1/8/32-step evaluation. No policy/critic/termination updates.',
                      fit_steps=a.fit_steps, fit_batch=a.batch_size, fit_window=a.fit_window,
                      velocity_loss_weight=a.velocity_loss_weight, num_envs=a.num_envs,
                      fit_lr=saved['wm_trainer']['lr'])
    (a.out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    results = dict(status='collecting', provenance=provenance, collections=[], baselines={}, refits={})

    def save():
        (a.out / 'result.json').write_text(json.dumps(results, indent=2) + '\n')

    @torch.no_grad()
    def collect(seed, k, samples):
        torch.manual_seed(seed)
        env = make_env(SuperMimicEnvCfgArgs(
            npz_files=[str(a.dataset / (name + '.npz')) for name in saved['data']['clips']],
            num_envs=a.num_envs, seed=seed, future_frames=saved['env']['future_frames'],
            episode_length_s=12, action_noise_std=.1, obs_noise=True, reset_noise=True,
            domain_randomization=True, record_transitions=True, replay_capacity=a.steps + 8,
            anchor_obs=saved['env']['anchor_obs'], velocity=VelocityCfg(**saved['velocity'])), device='cuda:0')
        data = {}
        try:
            cmd = env.command_manager.get_term('motion')
            replay = env.recorder_manager.get_term('replay').buffer
            obs, _ = env.reset()
            terminations = timeouts = 0
            for _ in range(a.steps):
                tracking = policy(obs['policy'], obs['goal'])
                velocity = policy.velocity_action(obs['policy'], cmd.velocity_command)
                action = torch.where(cmd.is_velocity[:, None], velocity, tracking)
                obs, _, term, tout, _ = env.step(action)
                terminations += int(term.sum())
                timeouts += int(tout.sum())
            report = dict(seed=seed, steps=a.steps, envs=a.num_envs, terminations=terminations,
                          timeouts=timeouts, sample_counts={})
            valid = replay.window_mask(k)
            for name, mask in [('motion', ~cmd.is_velocity), ('velocity', cmd.is_velocity)]:
                candidates = valid & mask[:, None]
                report['sample_counts'][name] = int(candidates.sum())
                e, t = replay._sample_from_mask(candidates, samples)
                tt = (t[None] + torch.arange(k + 1, device='cuda')[:, None]) % replay.T
                ee = e[None].expand(k + 1, -1)
                data[name] = dict(states=replay.s[ee, tt].clone(),
                                  actions=replay.a_exec[ee[:-1], tt[:-1]].clone(),
                                  xi=replay.xi[e, t].clone())
            results['collections'].append(report)
            torch.save(data, a.out / ('train-windows.pt' if seed == 0 else 'heldout-windows.pt'))
            print(json.dumps(dict(collection=report)), flush=True)
        finally:
            env.close()
        del env
        gc.collect()
        torch.cuda.empty_cache()
        return data

    if a.windows_dir:
        train = torch.load(a.windows_dir / 'train-windows.pt', map_location='cuda:0', weights_only=True)
        test = torch.load(a.windows_dir / 'heldout-windows.pt', map_location='cuda:0', weights_only=True)
        results['reused_windows'] = {name: digest(a.windows_dir / name)
                                    for name in ('train-windows.pt', 'heldout-windows.pt')}
    else:
        train = collect(0, 32, 4096)
        if a.heldout_dir:
            test = torch.load(a.heldout_dir / 'heldout-windows.pt', map_location='cuda:0', weights_only=True)
            results['heldout_sha256'] = digest(a.heldout_dir / 'heldout-windows.pt')
            (a.out/'heldout-windows.pt').symlink_to((a.heldout_dir/'heldout-windows.pt').resolve())
        else:
            test = collect(7, 32, 512)
    if any(d['actions'].shape[0] < a.fit_window for d in train.values()):
        raise ValueError('Recorded windows are shorter than the requested fit window')
    results['status'] = 'fitting'

    def load_wm(checkpoint):
        missing, extra = wm.load_state_dict(checkpoint['wm'], strict=False)
        if missing or any(not key.startswith('heads.') for key in extra):
            raise ValueError((missing, extra))

    @torch.no_grad()
    def evaluate(data_source=None):
        report = {}
        for name, data in (test if data_source is None else data_source).items():
            pred = data['states'][0]
            row = {}
            for j in range(data['actions'].shape[0]):
                pred = wm.step(pred, data['actions'][j], data['xi'])
                if j + 1 in (1, 8, 16, 32):
                    actual = data['states'][j + 1]
                    delta = measured_velocity(pred) - measured_velocity(actual)
                    row[str(j + 1)] = dict(velocity_rmse=delta.square().mean(0).sqrt().tolist(),
                                          velocity_bias=delta.mean(0).tolist(),
                                          nse=float(wm.spec.normalized_sq_error(pred, actual).mean()))
            report[name] = row
        return report

    load_wm(ck)
    results['baselines']['original'] = evaluate()
    results['train_baseline'] = evaluate(train)
    later = torch.load(a.later_checkpoint, map_location='cuda:0', weights_only=False)
    load_wm(later)
    results['baselines']['later_same_trajectories'] = dict(iteration=later['iteration'],
                                                          updates=later['num_updates'], metrics=evaluate())
    del later
    save()
    print(json.dumps(dict(baselines=results['baselines'])), flush=True)
    for arm in a.fit_arms:
        load_wm(ck)
        torch.manual_seed(42)
        optimizer = torch.optim.Adam(wm.parameters(), lr=saved['wm_trainer']['lr'])
        # Adam mutates its moment tensors; each arm must start from its own copy.
        optimizer.load_state_dict(copy.deepcopy(ck['wm_opt']))
        for group in optimizer.param_groups:
            group['lr'] = saved['wm_trainer']['lr']
        datasets = [train['motion']] if arm == 'motion_only' else [train['motion'], train['velocity']]
        merged = {key: torch.cat([d[key] for d in datasets], dim=1 if key != 'xi' else 0)
                  for key in train['motion']}
        rows = []
        started = time.monotonic()
        for update in range(1, a.fit_steps + 1):
            idx = torch.randint(merged['xi'].shape[0], (a.batch_size,), device='cuda')
            states, actions, xi = merged['states'][:, idx], merged['actions'][:, idx], merged['xi'][idx]
            pred = states[0]
            loss = 0.
            for j in range(a.fit_window):
                pred = wm.step(pred, actions[j], xi)
                error = wm.spec.error_vector(pred, states[j + 1]).square()
                loss = loss + (error.sum(-1) + (a.velocity_loss_weight-1)*error[..., 6:12].sum(-1)).mean() / a.fit_window
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(wm.parameters(), saved['wm_trainer']['grad_clip'])
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite refit')
            optimizer.step()
            if update in (16, 64, 128, 256, a.fit_steps):
                row = dict(updates=update, training_state_loss=float(loss.detach()), seconds=time.monotonic()-started,
                           heldout=evaluate())
                rows.append(row)
                results['refits'][arm] = rows
                save()
                print(json.dumps(dict(arm=arm, **row)), flush=True)
        torch.save(dict(wm=wm.state_dict(), config=saved, protocol=provenance), a.out / (arm + '-wm-only.pt'))
        results['refits'][arm][-1]['train_metrics'] = evaluate(train)
    results.update(status='complete', peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    save()


if __name__ == '__main__':
    main()
