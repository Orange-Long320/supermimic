"""Compare CPU policy inference with an offline fixture; no robot I/O."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--samples', type=int, default=200)
    a = p.parse_args()
    if a.out.exists() or a.samples < 10:
        raise ValueError('Expected new output and at least 10 samples')
    import torch
    torch.set_num_threads(1)
    data = json.loads(a.fixture.read_text())
    obs = torch.tensor([data['obs']], dtype=torch.float32)
    goal = torch.tensor([data['goal']], dtype=torch.float32)
    expected = torch.tensor([data['expected']], dtype=torch.float32)
    if obs.shape != (1,990) or goal.shape != (1,729) or expected.shape != (1,31):
        raise ValueError('Unexpected fixture dimensions')
    model = torch.jit.load(str(a.policy), map_location='cpu').eval()
    elapsed = []
    with torch.inference_mode():
        for _ in range(20):
            model(obs, goal)
        for _ in range(a.samples):
            start = time.perf_counter()
            result = model(obs, goal)
            elapsed.append((time.perf_counter()-start)*1000)
    if result.shape != expected.shape or not torch.isfinite(result).all():
        raise ValueError('Invalid inference output')
    ordered = sorted(elapsed)
    report = {'policy_sha256':hashlib.sha256(a.policy.read_bytes()).hexdigest(),
              'fixture_sha256':hashlib.sha256(a.fixture.read_bytes()).hexdigest(),
              'torch_version':torch.__version__, 'samples':a.samples,
              'max_abs_error':float((result-expected).abs().max()),
              'inference_ms_p50':ordered[len(ordered)//2],
              'inference_ms_p99':ordered[min(len(ordered)-1, int(.99*len(ordered)))],
              'inference_ms_max':max(ordered),
              'scope':'offline fixed-input CPU model inference only; not end-to-end control or real-time validation'}
    a.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if report['max_abs_error'] > 1e-4:
        raise SystemExit('Model output mismatch')


if __name__ == '__main__':
    main()
