"""Replay saved branch inputs to separate batch-shape precision from label errors."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint', 'probe', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    if Path(sys.prefix).resolve() != Path('/home/2025-Long.xh/HHD/venvs/supermimic').resolve():
        p.error('Dedicated server interpreter required')
    if a.out.exists():
        raise FileExistsError(a.out)
    import torch
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    torch.set_num_threads(2)
    policy, _ = load_policy_from_checkpoint(a.checkpoint, 'cuda:0')
    d = torch.load(a.probe, map_location='cuda:0', weights_only=True)
    rows = []
    with torch.no_grad():
        for precision in ('tf32', 'ieee'):
            torch.backends.cuda.matmul.fp32_precision = precision
            full = policy.standing_skill(d['obs'], d['feedback'])
            subset = policy.standing_skill(d['obs'][:128], d['feedback'][:128])
            lp_full = torch.distributions.Normal(full[:128], .04).log_prob(d['sampled'][:128]).sum(-1)
            lp_subset = torch.distributions.Normal(subset, .04).log_prob(d['sampled'][:128]).sum(-1)
            rows.append(dict(precision=precision, mean_max_error=float((full[:128]-subset).abs().max()),
                             logp_max_error=float((lp_full-lp_subset).abs().max()),
                             mean_kl=float((full[:128]-subset).square().sum(-1).mean()/(2*.04**2))))
    a.out.write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(rows), flush=True)


if __name__ == '__main__':
    main()
