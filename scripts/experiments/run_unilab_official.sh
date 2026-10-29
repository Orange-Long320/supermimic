#!/usr/bin/env bash
# Server-only bounded official runtime smoke; independent output per invocation.
set -euo pipefail
if [[ $# != 2 ]]; then
  echo 'usage: run_unilab_official.sh {mujoco|mjwarp} NEW_OUTPUT_DIRECTORY' >&2
  exit 2
fi
backend=$1
out=$2
[[ $backend == mujoco || $backend == mjwarp ]]
[[ ! -e $out ]]
mkdir -p "$out"
source_root=/home/2025-Long.xh/SSD/supermimic/vendor/unilab
env_root=/home/2025-Long.xh/HHD/venvs/supermimic-unilab
cd "$source_root"
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH="$source_root/src"
export HF_HUB_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export UV_NO_PROGRESS=1
cp "$0" "$out/controller-as-run.sh"
date -u +%FT%TZ > "$out/start-utc.txt"
echo $$ > "$out/controller.pid"
nvidia-smi > "$out/gpu-before.txt"
ps -eo user,pid,comm > "$out/processes-before.txt"
set +e
timeout --signal=INT --kill-after=30 900 \
  uv run --no-project --python "$env_root/bin/python" train \
  --algo flashsac --task g1_motion_tracking --sim "$backend" \
  algo.num_envs=256 algo.batch_size=512 algo.replay_buffer_n=128 \
  algo.learning_starts=16 algo.max_iterations=320 algo.save_interval=320 \
  training.no_play=true training.export_onnx=false \
  training.log_dir="$out/train" training.log_interval=10 \
  training.torch_threads.compile_threads=2 \
  training.dp_collector_cpu_ids='[[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]' \
  > "$out/console.log" 2>&1
status=$?
set -e
echo "$status" > "$out/exit-code.txt"
date -u +%FT%TZ > "$out/end-utc.txt"
nvidia-smi > "$out/gpu-after.txt"
exit "$status"
