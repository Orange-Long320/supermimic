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
    vc = VelocityCfg(**saved['velocity'])
    # Collect only tracking episodes: their frame and physical histories remain
    # paired, independently of the velocity replay's expired reference flags.
    args = SuperMimicEnvCfgArgs(npz_files=[str(x) for x in sorted(a.dataset.glob('*.npz'))],
        num_envs=100, seed=17, future_frames=saved['env']['future_frames'],
        episode_length_s=15., action_noise_std=0., obs_noise=False, reset_noise=False,
        domain_randomization=False, record_transitions=False, anchor_obs=saved['env']['anchor_obs'])
    env = make_env(args, device=device)
    rows = []
    try:
        cmd = env.command_manager.get_term('motion')
        table = reference_command_table(cmd.library)
        # Spread the batch across all clips, start at frame zero.
        cmd.forced_frame[:] = cmd.library.clip_start[torch.arange(100, device=device) % cmd.library.num_clips]
        obs, _ = env.reset()
        with torch.no_grad():
            for step in range(650):
                command, valid = bounded_reference(table, cmd.frame, vc)
                latent = policy.encoders['motion'](obs['goal'])
                action = policy.decoder(latent, obs['policy'])
                if step % 2 == 0 and valid.any():
                    rows.append(tuple(t[valid].clone() for t in (obs['policy'], command, latent, action)))
                obs, _, _, _, _ = env.step(action)
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
    original = ck['policy']
    for name, value in policy.state_dict().items():
        if not name.startswith('encoders.velocity.'):
            assert torch.equal(value.cpu(), original[name]), name
    ck['policy'] = {k:v.cpu() for k,v in policy.state_dict().items()}
    ck.pop('velocity_training', None)
    ck['bootstrap'] = dict(source_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        updates=a.updates, samples=count, holdout_before=before, holdout_after=after,
        limitation='Random row holdout checks fitting only; continuous physical acceptance required. Not resumable.')
    torch.save(ck, a.out/'candidate.pt')
    (a.out/'report.json').write_text(json.dumps(ck['bootstrap'], indent=2)+'\n')
    print(json.dumps(ck['bootstrap']), flush=True)


if __name__ == '__main__':
    main()
