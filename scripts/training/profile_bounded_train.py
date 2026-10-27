"""Run the established training profiler with a per-process Torch memory cap."""

import argparse
from pathlib import Path
import runpy
import sys

import torch


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--torch-memory-fraction", type=float, default=0.2)
    args, rest = parser.parse_known_args()
    if not 0 < args.torch_memory_fraction <= 0.2:
        raise SystemExit("Bounded training requires a Torch fraction in (0, 0.2]")
    torch.cuda.set_per_process_memory_fraction(args.torch_memory_fraction, 0)
    print("Torch allocator fraction", args.torch_memory_fraction, flush=True)
    sys.argv = [str(Path(__file__).with_name("profile_train.py")), *rest]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
