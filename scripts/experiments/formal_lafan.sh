#!/usr/bin/env bash
# Server-only: wait for a free GPU, then run the measured configuration for three hours.
set -euo pipefail
SM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SM_TAG="${1:?unique experiment tag required}"
SM_GPU="${2:?physical GPU index required}"
SM_NOT_BEFORE="${3:-0}"
SM_RESUME="${4:-}"
[[ "$SM_NOT_BEFORE" =~ ^[0-9]+$ ]]
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
cp configs/source-lock.json "$SM_RUN/source-lock.json"
cp "$0" "$SM_RUN/launch.sh"
env -u PYTHONPATH .venv/bin/python -c 'import importlib.metadata as m; print("\n".join(sorted(d.metadata.get("Name", "unknown") + "==" + d.version for d in m.distributions())))' > "$SM_RUN/environment.txt"
test "$(find data/npz/x2_ultra/lafan1 -name '*.npz' | wc -l)" -eq 40
SM_ARGS=(--robot x2_ultra --run-name "${SM_TAG}_train"
  --env.num-envs 1024 --env.replay-capacity 512 --env.anchor-obs
  --wm-trainer.batch-size 1024 --policy-trainer.batch-size 1024
  --schedule.total-iterations 320000 --schedule.log-interval 100
  --schedule.checkpoint-interval 1000 --schedule.max-hours 3)
if [[ -n "$SM_RESUME" ]]; then
  test -f "$SM_RESUME"
  SM_ARGS+=(--resume "$SM_RESUME")
  sha256sum "$SM_RESUME" > "$SM_RUN/resume-checkpoint.sha256"
fi
bash scripts/run_local.sh scripts/training/train.py "${SM_ARGS[@]}" --print-config > "$SM_RUN/config-preview.json"
printf '%q ' bash scripts/run_local.sh scripts/training/profile_train.py --profile-gpu "$SM_GPU" --profile-output "$SM_RUN/profile.json" "${SM_ARGS[@]}" > "$SM_RUN/command.txt"
printf '\n' >> "$SM_RUN/command.txt"
printf '%s\n' "$SM_NOT_BEFORE" > "$SM_RUN/not-before-epoch.txt"
printf '%s Scheduled not before epoch %s\n' "$(date -Is)" "$SM_NOT_BEFORE"
while (( $(date +%s) < SM_NOT_BEFORE )); do sleep 30; done
printf '%s Waiting for GPU %s to have no compute processes\n' "$(date -Is)" "$SM_GPU"
while true; do
  SM_OCCUPANTS="$(nvidia-smi -i "$SM_GPU" --query-compute-apps=pid --format=csv,noheader)"
  if [[ -z "$SM_OCCUPANTS" ]]; then break; fi
  sleep 60
done
nvidia-smi > "$SM_RUN/gpu-before.txt"
ps -eo user,pid,comm > "$SM_RUN/processes-before.txt"
printf '%s Starting on GPU %s; controller PID %s\n' "$(date -Is)" "$SM_GPU" "$$"
bash scripts/run_local.sh scripts/training/profile_train.py --profile-gpu "$SM_GPU" --profile-output "$SM_RUN/profile.json" "${SM_ARGS[@]}" > "$SM_RUN/train.log" 2>&1 &
SM_CHILD=$!
printf '%s\n' "$SM_CHILD" > "$SM_RUN/train.pid"
wait "$SM_CHILD"
