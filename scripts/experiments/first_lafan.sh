#!/usr/bin/env bash
# Server-only: finish an already-started retarget, prepare LAFAN, profile short runs.
set -euo pipefail
SM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SM_TAG="${1:?unique experiment tag required}"
SM_GPU="${2:?physical GPU index required}"
SM_RETARGET_EXIT="${3:?retarget exit-status file required}"
cd "$SM_ROOT"
SM_RUN="logs/$SM_TAG"
test ! -e "$SM_RUN"
mkdir -p "$SM_RUN"
exec > >(tee "$SM_RUN/controller.log") 2>&1
trap 'code=$?; printf "%s\n" "$code" > "$SM_RUN/controller.exit"' EXIT
export CUDA_VISIBLE_DEVICES="$SM_GPU"
export XDG_CACHE_HOME="$(readlink -f "$SM_ROOT/.cache")"
export WARP_CACHE_PATH="$XDG_CACHE_HOME/warp"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
cp configs/source-lock.json "$SM_RUN/source-lock.json"
printf 'Waiting for existing retarget task: %s\n' "$SM_RETARGET_EXIT"
while [[ ! -f "$SM_RETARGET_EXIT" ]]; do sleep 5; done
test "$(cat "$SM_RETARGET_EXIT")" = 0
test "$(find data/lafan1-retarget-x2_ultra -name '*.csv' | wc -l)" -eq 40
test ! -e data/npz/x2_ultra/lafan1
printf 'Preparing all 40 LAFAN clips\n'
bash scripts/run_local.sh scripts/data/prepare_lafan.py --robot x2_ultra \
  --source data/lafan1-retarget-x2_ultra --dest data/npz/x2_ultra/lafan1 \
  --input-fps 30 --workers 8 --audit "$SM_RUN/preparation.json" \
  > "$SM_RUN/preparation.log" 2>&1
test "$(find data/npz/x2_ultra/lafan1 -name '*.npz' | wc -l)" -eq 40
for SM_ENVS in 128 512; do
  SM_OCCUPANTS="$(nvidia-smi -i "$SM_GPU" --query-compute-apps=pid --format=csv,noheader)"
  if [[ -n "$SM_OCCUPANTS" ]]; then
    printf 'GPU %s has compute processes; stop before launching: %s\n' "$SM_GPU" "$SM_OCCUPANTS"
    exit 1
  fi
  printf 'Starting %s environments on GPU %s\n' "$SM_ENVS" "$SM_GPU"
  bash scripts/run_local.sh scripts/training/profile_train.py \
    --profile-gpu "$SM_GPU" --profile-output "$SM_RUN/profile-$SM_ENVS.json" \
    --robot x2_ultra --preset smoke --run-name "${SM_TAG}_${SM_ENVS}" \
    --env.num-envs "$SM_ENVS" --env.anchor-obs \
    > "$SM_RUN/train-$SM_ENVS.log" 2>&1
done
printf 'Both short runs completed; inspect memory and losses before longer training.\n'
