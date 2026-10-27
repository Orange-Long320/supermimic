"""Bound this process's Torch allocator while reusing the measured smoke entry."""

import argparse
import runpy
from pathlib import Path
import sys

import torch


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--torch-memory-fraction", type=float, default=0.15)
args, remaining = parser.parse_known_args()
if not 0 < args.torch_memory_fraction <= 0.2:
    raise SystemExit("Bounded smoke requires a Torch memory fraction in (0, 0.2]")
torch.cuda.set_per_process_memory_fraction(args.torch_memory_fraction, 0)
print("Torch allocator fraction", args.torch_memory_fraction, flush=True)
sys.argv = [str(Path(__file__).with_name("profile_train.py")), *remaining]
runpy.run_path(sys.argv[0], run_name="__main__")
