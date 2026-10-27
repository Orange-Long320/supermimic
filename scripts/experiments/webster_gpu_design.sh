#!/usr/bin/env bash
# Dedicated free GPU only. Never shares the six-hour run's GPU 0.
set -euo pipefail
cd /home/2025-Long.xh/SSD/supermimic
SM_TAG=${1:?unique run tag}
SM_PHASE=${2:?takeoff or full}
SM_GPU=${3:?physical free GPU 2 or 3}
SM_WARM=${4:-}
SM_GUIDE=${5:?independent guide NPZ}
SM_TAIL=${6:-}
SM_CHECKPOINT=${7:-logs/webster_transfer_short_20261027_r1_train/ckpt_latest.pt}
SM_TARGET_HY=${8:-25}
SM_TARGET_VZ=${9:-2.4}
[[ "$SM_TAG" =~ ^[a-zA-Z0-9_-]+$ ]]
[[ "$SM_PHASE" == takeoff || "$SM_PHASE" == full ]]
[[ "$SM_GPU" =~ ^[23]$ ]]
test -z "$(nvidia-smi -i "$SM_GPU" --query-compute-apps=pid --format=csv,noheader,nounits)"
SM_FREE=$(nvidia-smi -i "$SM_GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
(( SM_FREE > 16000 ))
SM_RUN="logs/$SM_TAG"
mkdir "$SM_RUN"
exec > "$SM_RUN/controller.log" 2>&1
trap 'echo "$?" > "$SM_RUN/controller.exit"' EXIT
echo "$$" > "$SM_RUN/controller.pid"
export CUDA_VISIBLE_DEVICES="$SM_GPU" MUJOCO_EGL_DEVICE_ID="$SM_GPU"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
unset PYTHONPATH PYTHONHOME
nvidia-smi -i "$SM_GPU" > "$SM_RUN/gpu-before.txt"
cp scripts/data/optimize_webster_policy_cpu.py "$SM_RUN/designer-source.py"
cp src/supermimic/data/design_metrics.py "$SM_RUN/metrics-source.py"
cp "${BASH_SOURCE[0]}" "$SM_RUN/launch.sh"
sha256sum "$SM_CHECKPOINT" "$SM_GUIDE" scripts/data/optimize_webster_policy_cpu.py \
  src/supermimic/data/design_metrics.py > "$SM_RUN/inputs.sha256"
SM_SECONDS=1.2
[[ "$SM_PHASE" != full ]] || SM_SECONDS=3
SM_CMD=(nice -n 19 timeout --signal=TERM --kill-after=30 2100
  bash scripts/run_local.sh scripts/robot.py x2_ultra scripts/data/optimize_webster_policy_cpu.py
  --ckpt "$SM_CHECKPOINT" --source "$SM_GUIDE" --out "$SM_RUN/design"
  --device cuda:0 --phase "$SM_PHASE" --seconds "$SM_SECONDS"
  --target-hy "$SM_TARGET_HY" --target-vz "$SM_TARGET_VZ"
  --num-envs 128 --population 256 --iterations 160 --max-seconds 1800)
if [[ -n "$SM_WARM" ]]; then
  sha256sum "$SM_WARM" >> "$SM_RUN/inputs.sha256"
  SM_CMD+=(--warmstart "$SM_WARM")
fi
if [[ -n "$SM_TAIL" ]]; then
  sha256sum "$SM_TAIL" >> "$SM_RUN/inputs.sha256"
  SM_CMD+=(--tail-of "$SM_TAIL")
fi
printf '%q ' "${SM_CMD[@]}" > "$SM_RUN/command.sh"
printf '\n' >> "$SM_RUN/command.sh"
"${SM_CMD[@]}" > "$SM_RUN/designer.log" 2>&1 &
SM_PID=$!
echo "$SM_PID" > "$SM_RUN/worker.pid"
wait "$SM_PID"
echo 'Design search complete. Three independent full physical replays remain required before training.'
