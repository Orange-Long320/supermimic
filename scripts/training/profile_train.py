"""Run the normal training CLI and record PyTorch and device memory peaks."""

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile-gpu", required=True)
    parser.add_argument("--profile-output", required=True, type=Path)
    args, training_args = parser.parse_known_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.profile_gpu:
        raise SystemExit("CUDA_VISIBLE_DEVICES must match the single --profile-gpu")
    if args.profile_output.exists():
        raise SystemExit(f"Refusing to overwrite {args.profile_output}")

    import torch
    from supermimic.training.cli import main as train

    torch.cuda.reset_peak_memory_stats()
    stop = threading.Event()
    samples = []
    sample_errors = []

    def sample():
        while not stop.is_set():
            try:
                value = subprocess.check_output(
                    ["nvidia-smi", "-i", args.profile_gpu,
                     "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True, timeout=5,
                )
                samples.append({"elapsed_s": time.monotonic() - started,
                                "device_used_mib": int(value.strip())})
            except (subprocess.SubprocessError, ValueError) as error:
                sample_errors.append(str(error))
            stop.wait(1)

    started = time.monotonic()
    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    status = "failed"
    try:
        train(training_args)
        status = "completed"
    finally:
        stop.set()
        monitor.join(timeout=6)
        result = {
            "status": status, "gpu": args.profile_gpu,
            "elapsed_s": time.monotonic() - started,
            "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "device_peak_sampled_mib": max(
                (s["device_used_mib"] for s in samples), default=None),
            "device_sampling_interval_s": 1,
            "note": "Device samples include graphics and other processes; may miss short peaks.",
            "command_args": training_args, "samples": samples,
            "sampling_errors": sample_errors,
        }
        args.profile_output.parent.mkdir(parents=True, exist_ok=True)
        args.profile_output.write_text(json.dumps(result, indent=2) + "\n")
        print("[PROFILE]", json.dumps({k: v for k, v in result.items()
                                       if k not in ("samples", "command_args")}))


if __name__ == "__main__":
    main()
