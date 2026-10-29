"""Bounded teacher bootstrap of the velocity encoder on real tracking histories.

Execute only on the simulation server with the pinned keyboard source on PYTHONPATH.
Writes a candidate, never overwrites the source checkpoint or optimizer state.
"""
from pathlib import Path
import argparse
import hashlib
import json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--updates', type=int, default=2000)
    p.add_argument('--backward-dagger', action='store_true')
    p.add_argument('--tracking-source', type=Path)
    p.add_argument('--standing-starts', action='store_true')
    a = p.parse_args()
    a.out.mkdir(exist_ok=False)
    import torch
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.locomotion.config import VelocityCfg
    from supermimic.locomotion.task import reference_command_table, bounded_reference
    torch.set_num_threads(2)
    torch.manual_seed(17)
    device = 'cuda:0'
    policy, saved = load_policy_from_checkpoint(a.checkpoint, device)
    if a.tracking_source:
        source = torch.load(a.tracking_source, map_location=device, weights_only=False)
        policy.load_state_dict(source['policy'], strict=False)
    vc = VelocityCfg(**saved['velocity'])
    # Collect only tracking episodes: their frame and physical histories remain
    # paired, independently of the velocity replay's expired reference flags.
    args = SuperMimicEnvCfgArgs(npz_files=[str(x) for x in sorted(a.dataset.glob('*.npz'))],
        num_envs=100, seed=17, future_frames=saved['env']['future_frames'],
        episode_length_s=15., action_noise_std=0., obs_noise=False, reset_noise=False,
        domain_randomization=False, record_transitions=False, anchor_obs=saved['env']['anchor_obs'], velocity=vc if a.standing_starts else None)
    env = make_env(args, device=device)
    rows = []
    if a.backward_dagger:
        policy.requires_grad_(False)
        policy.encoders['velocity'].requires_grad_(True)
        online_opt = torch.optim.Adam(policy.encoders['velocity'].parameters(), lr=2e-4)
    dagger_metrics = []
    try:
        cmd = env.command_manager.get_term('motion')
        table = reference_command_table(cmd.library)
        # Spread the batch across all clips, start at frame zero.
        cmd.forced_frame[:] = cmd.library.clip_start[torch.arange(100, device=device) % cmd.library.num_clips]
        if a.standing_starts:
            cmd.is_velocity[:] = True
            cmd.force_standing_reset = True
        obs, _ = env.reset()
        backward_clip = next(i for i,c in enumerate(cmd.library.clips) if c.name == 'backward')
        backward = cmd.library.frame_clip[cmd.frame] == backward_clip
        total_steps = 2040 if a.backward_dagger else 650
        for step in range(total_steps):
            if a.standing_starts and step and step % 204 == 0:
                obs, _ = env.reset()
            with torch.no_grad():
                command, valid = bounded_reference(table, cmd.frame, vc)
                if a.backward_dagger:
                    # Fixed requested backward speed, independent of frame sway.
                    # Teacher supplies gait phase; off-trajectory student history
                    # is included explicitly rather than inferred from fitting.
                    command[backward] = torch.tensor([-.2,0.,0.],device=device)
                    valid[backward] = True
                latent = policy.encoders['motion'](obs['goal'])
                action = policy.decoder(latent, obs['policy'])
                if step % 2 == 0 and valid.any():
                    rows.append(tuple(t[valid].clone() for t in (obs['policy'], command, latent, action)))
                if a.backward_dagger:
                    student = policy.velocity_action(obs['policy'], command)
                    # Different mixtures expose the student to its own errors.
                    alpha = ((torch.arange(100,device=device) % 5) / 4)[:,None]
                    action = torch.where(backward[:,None], action*(1-alpha)+student*alpha, action)
                obs, _, _, _, _ = env.step(action)
            if a.backward_dagger and (step+1) % 102 == 0:
                pool = [torch.cat([r[i] for r in rows]) for i in range(4)]
                for _ in range(150):
                    neg = (pool[1][:,0] == -.2).nonzero().flatten()
                    other = (pool[1][:,0] != -.2).nonzero().flatten()
                    ids = torch.cat((neg[torch.randint(len(neg),(512,),device=device)], other[torch.randint(len(other),(512,),device=device)]))
                    x,c,z,y = [v[ids] for v in pool]
                    pred = policy.encoders['velocity'](x,c)
                    loss = (pred-z).square().mean() + (policy.decoder(pred,x)-y).square().mean()
                    online_opt.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.encoders['velocity'].parameters(),1.)
                    online_opt.step()
                dagger_metrics.append(dict(step=step+1,loss=float(loss.detach())))
                print(dagger_metrics[-1],flush=True)
    finally:
        env.close()
    data = [torch.cat([r[i] for r in rows]) for i in range(4)]
    count = len(data[0])
    perm = torch.randperm(count, device=device)
    holdout, train = perm[:count//5], perm[count//5:]
    policy.requires_grad_(False)
    encoder = policy.encoders['velocity']
    encoder.requires_grad_(True)
    opt = torch.optim.Adam(encoder.parameters(), lr=2e-4)
    def losses(ids):
        obs, command, target, action = [x[ids] for x in data]
        z = encoder(obs, command)
        return (z-target).square().mean(), (policy.decoder(z, obs)-action).square().mean()
    with torch.no_grad():
        before = [float(x) for x in losses(holdout)]
    for step in range(a.updates):
        ids = train[torch.randint(len(train), (1024,), device=device)]
        latent_loss, action_loss = losses(ids)
        loss = latent_loss + action_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.)
        opt.step()
        if step % 200 == 0:
            print(step, float(latent_loss.detach()), float(action_loss.detach()), flush=True)
    with torch.no_grad():
        after = [float(x) for x in losses(holdout)]
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    # Encoder-only change; stale optimizer state is deliberately removed. This
    # artifact is for evaluation / warm start, not an exact training resume.
    original = source['policy'] if a.tracking_source else ck['policy']
    for name, value in policy.state_dict().items():
        if not name.startswith('encoders.velocity.'):
            assert torch.equal(value.cpu(), original[name]), name
    ck['policy'] = {k:v.cpu() for k,v in policy.state_dict().items()}
    ck.pop('velocity_training', None)
    ck['bootstrap'] = dict(source_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        tracking_source_sha256=hashlib.sha256(a.tracking_source.read_bytes()).hexdigest() if a.tracking_source else None, standing_starts=a.standing_starts,
        updates=a.updates, online_updates=150*len(dagger_metrics), dagger=dagger_metrics, samples=count,
        dataset_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.dataset.glob('*.npz'))}, holdout_before=before, holdout_after=after,
        limitation=('Post-collection row split is not independent of online DAgger fitting; physical acceptance required. Not resumable.' if a.backward_dagger else 'Random row holdout checks fitting only; continuous physical acceptance required. Not resumable.'))
    torch.save(ck, a.out/'candidate.pt')
    (a.out/'report.json').write_text(json.dumps(ck['bootstrap'], indent=2)+'\n')
    print(json.dumps(ck['bootstrap']), flush=True)


if __name__ == '__main__':
    main()
