#!/usr/bin/env bash
# Server-only: measure a larger, compiled single-GPU configuration before long training.
set -euo pipefail
SM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SM_TAG="${1:?unique experiment tag required}"
SM_GPU="${2:?physical GPU index required}"
cd "$SM_ROOT"
SM_RUN="logs/$SM_TAG"
test ! -e "$SM_RUN"
test ! -e "logs/${SM_TAG}_train"
mkdir "$SM_RUN"
exec > >(tee "$SM_RUN/controller.log") 2>&1
trap 'code=$?; printf "%s\n" "$code" > "$SM_RUN/controller.exit"' EXIT
export CUDA_VISIBLE_DEVICES="$SM_GPU"
export XDG_CACHE_HOME="$(readlink -f "$SM_ROOT/.cache")"
export WARP_CACHE_PATH="$XDG_CACHE_HOME/warp" TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
test -z "$(nvidia-smi -i "$SM_GPU" --query-compute-apps=pid --format=csv,noheader)"
test "$(find data/npz/x2_ultra/lafan1 -name '*.npz' | wc -l)" -eq 40
cp configs/source-lock.json "$SM_RUN/source-lock.json"
cp "$0" "$SM_RUN/launch.sh"
env -u PYTHONPATH .venv/bin/python -c 'import importlib.metadata as m; print("\n".join(sorted(d.metadata.get("Name", "unknown") + "==" + d.version for d in m.distributions())))' > "$SM_RUN/environment.txt"
SM_ARGS=(scripts/training/profile_train.py
  --profile-gpu "$SM_GPU" --profile-output "$SM_RUN/profile.json"
  --robot x2_ultra --run-name "${SM_TAG}_train"
  --env.num-envs 1024 --env.replay-capacity 512 --env.anchor-obs
  --wm-trainer.batch-size 1024 --policy-trainer.batch-size 1024
  --schedule.total-iterations 100 --schedule.log-interval 20
  --schedule.checkpoint-interval 100 --schedule.max-hours 1)
printf '%q ' bash scripts/run_local.sh "${SM_ARGS[@]}" > "$SM_RUN/command.txt"
printf '\n' >> "$SM_RUN/command.txt"
bash scripts/run_local.sh scripts/training/train.py "${SM_ARGS[@]:5}" --print-config > "$SM_RUN/config-preview.json"
bash scripts/run_local.sh "${SM_ARGS[@]}" > "$SM_RUN/train.log" 2>&1
