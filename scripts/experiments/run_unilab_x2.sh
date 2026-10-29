#!/usr/bin/env bash
# Full official FlashSAC runtime on prepared project X2 data; server only.
set -euo pipefail
[[ $# == 3 ]]
backend=$1
prepared=$2
out=$3
[[ $backend == mujoco || $backend == mjwarp ]]
[[ ! -e $out ]]
mkdir -p "$out"
run_root=/home/2025-Long.xh/HHD/data/supermimic/logs/unilab_full_20261029_r1
source_root=/home/2025-Long.xh/SSD/supermimic/vendor/unilab
env_root=/home/2025-Long.xh/HHD/venvs/supermimic-unilab
cd "$source_root"
export CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$run_root/unilab_x2:$source_root/src"
cp "$0" "$out/controller-as-run.sh"
date -u +%FT%TZ > "$out/start-utc.txt"
echo $$ > "$out/controller.pid"
nvidia-smi > "$out/gpu-before.txt"
set +e
timeout --signal=INT --kill-after=30 900 \
  uv run --no-project --python "$env_root/bin/python" python \
  "$run_root/unilab_x2/train.py" --unilab-root "$source_root" \
  --prepared "$prepared" --backend "$backend" --out "$out/train" \
  algo.num_envs=256 algo.batch_size=512 algo.replay_buffer_n=128 \
  algo.learning_starts=16 algo.max_iterations=320 algo.save_interval=320 \
  training.log_interval=10 training.torch_threads.compile_threads=2 \
  training.dp_collector_cpu_ids='[[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]' \
  > "$out/console.log" 2>&1
status=$?
set -e
echo "$status" > "$out/exit-code.txt"
date -u +%FT%TZ > "$out/end-utc.txt"
nvidia-smi > "$out/gpu-after.txt"
exit "$status"
