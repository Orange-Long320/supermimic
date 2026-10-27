#!/usr/bin/env bash
# Server-only: direct dual-GPU resume; user requested skipping further short tests.
set -euo pipefail
SM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SM_ROOT"
SM_TAG="${1:?unique run tag required}"
SM_RESUME="${2:?checkpoint required}"
SM_GPU_A="${3:-0}"
SM_GPU_B="${4:-1}"
[[ "$SM_GPU_A" =~ ^[0-3]$ && "$SM_GPU_B" =~ ^[0-3]$ && "$SM_GPU_A" != "$SM_GPU_B" ]]
SM_RUN="logs/$SM_TAG"
test ! -e "$SM_RUN"
test -f "$SM_RESUME"
mkdir "$SM_RUN"
exec > >(tee "$SM_RUN/controller.log") 2>&1
trap 'code=$?; printf "%s\n" "$code" > "$SM_RUN/controller.exit"' EXIT
export XDG_CACHE_HOME="$(readlink -f .cache)"
export WARP_CACHE_PATH="$XDG_CACHE_HOME/warp" TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
cp "$0" "$SM_RUN/launch.sh"
cp configs/source-lock.json "$SM_RUN/source-lock.json"
sha256sum "$SM_RESUME" > "$SM_RUN/resume-checkpoint.sha256"
env -u PYTHONPATH .venv/bin/python -c 'import importlib.metadata as m; print("\n".join(sorted(d.metadata.get("Name", "unknown") + "==" + d.version for d in m.distributions())))' > "$SM_RUN/environment.txt"
SM_BASE=(--robot x2_ultra --resume "$SM_RESUME"
  --env.num-envs 2048 --env.replay-capacity 512 --env.anchor-obs
  --wm-trainer.batch-size 2048 --policy-trainer.batch-size 2048
  --schedule.log-interval 100 --schedule.checkpoint-interval 1000)
wait_free() {
  printf '%s Waiting for GPU(s): %s\n' "$(date -Is)" "$*"
  while true; do
    local busy=0 gpu occupants
    for gpu in "$@"; do
      occupants=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)
      if [[ -n "$occupants" ]]; then busy=1; fi
    done
    if (( busy == 0 )); then return; fi
    sleep 10
  done
}
run_logged() {
  local phase=$1
  shift
  printf '%q ' "$@" > "$SM_RUN/$phase.command.txt"
  printf '\n' >> "$SM_RUN/$phase.command.txt"
  printf '%s Starting phase %s\n' "$(date -Is)" "$phase"
  "$@" > "$SM_RUN/$phase.log" 2>&1 &
  local child=$!
  printf '%s\n' "$child" > "$SM_RUN/$phase.pid"
  wait "$child"
}
test ! -e "logs/${SM_TAG}_train"
wait_free "$SM_GPU_A" "$SM_GPU_B"
export CUDA_VISIBLE_DEVICES="$SM_GPU_A,$SM_GPU_B"
nvidia-smi > "$SM_RUN/gpu-before-formal.txt"
ps -eo user,pid,comm > "$SM_RUN/processes-before-formal.txt"
run_logged train bash scripts/run_local.sh -m torch.distributed.run --standalone --nproc-per-node 2 \
  scripts/training/profile_distributed.py --profile-dir "$SM_RUN/train-profile" "${SM_BASE[@]}" \
  --run-name "${SM_TAG}_train" --schedule.total-iterations 320000 \
  --schedule.max-hours 3
