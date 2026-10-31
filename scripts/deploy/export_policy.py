"""Export the selected X2 policy for offline integration; no robot connection."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True, help='New output directory')
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('Output directory already exists')
    import torch
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.robots.dimensions import NJ, OBS_DIM, ROBOT
    if ROBOT != 'x2_ultra':
        raise ValueError('Set SM_ROBOT=x2_ultra before starting')
    policy, cfg = load_policy_from_checkpoint(args.checkpoint, 'cpu')
    history = cfg['policy']['history']
    goal_dim = cfg['policy']['goal_dim']
    generator = torch.Generator().manual_seed(177)
    obs = torch.randn(1, history * OBS_DIM, generator=generator)
    goal = torch.randn(1, goal_dim, generator=generator)
    with torch.no_grad():
        exported = torch.jit.trace(policy, (obs, goal), check_trace=True)
        error = float((policy(obs, goal) - exported(obs, goal)).abs().max())
    if error > 1e-5:
        raise RuntimeError(f'Export mismatch: {error}')
    args.out.mkdir(parents=True)
    exported.save(str(args.out / 'policy.pt'))
    meta = {'robot': ROBOT, 'joints': NJ, 'history': history, 'obs_dim': history*OBS_DIM,
            'anchor_obs': cfg['env']['anchor_obs'],
            'policy_sha256': hashlib.sha256((args.out/'policy.pt').read_bytes()).hexdigest(),
            'goal_dim': goal_dim, 'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            'torch_version': torch.__version__, 'single_vector_max_error': error,
            'scope': 'offline export only; inputs, hardware and physical behavior not validated'}
    (args.out/'manifest.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__': main()
