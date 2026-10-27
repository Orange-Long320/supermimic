"""Pin each torchrun rank before CUDA imports and reuse the memory profiler."""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile-dir", required=True, type=Path)
    args, training_args = parser.parse_known_args()
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    local_rank = int(os.environ["LOCAL_RANK"])
    if len(devices) != int(os.environ["LOCAL_WORLD_SIZE"]):
        raise SystemExit("Provide exactly one physical GPU per local rank")
    gpu = devices[local_rank]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["SM_GPU_PINNED"] = "1"
    sys.argv = [sys.argv[0], "--profile-gpu", gpu, "--profile-output",
                str(args.profile_dir / f"rank-{local_rank}.json"), *training_args]
    from profile_train import main as profile

    profile()


if __name__ == "__main__":
    main()
