#!/usr/bin/env bash
# Server-only: single-GPU capacity gate, two-GPU update gate, then resumed training.
set -euo pipefail
SM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SM_ROOT"
SM_TAG="${1:?unique run tag required}"
SM_RESUME="${2:?checkpoint required}"
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
  --env.num-envs 1024 --env.replay-capacity 512 --env.anchor-obs
  --wm-trainer.batch-size 2048 --policy-trainer.batch-size 2048
  --schedule.log-interval 20 --schedule.checkpoint-interval 1000)
SM_START=$(env -u PYTHONPATH .venv/bin/python -c 'import torch,sys; print(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["iteration"])' "$SM_RESUME")
SM_STOP=$((SM_START + 100))
wait_free() {
  printf '%s Waiting for GPU(s): %s\n' "$(date -Is)" "$*"
  while true; do
    local busy=0 gpu occupants
    for gpu in "$@"; do
      occupants=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)
      if [[ -n "$occupants" ]]; then busy=1; fi
    done
    if (( busy == 0 )); then return; fi
    sleep 60
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
validate() {
  bash scripts/run_local.sh - "$SM_RUN" "$1" "$2" "$3" <<'PY'
import json, sys
from pathlib import Path
import torch
root, phase, expected, ranks = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
paths = [root / "capacity.json"] if ranks == 1 else [root / "dual-profile" / f"rank-{r}.json" for r in range(ranks)]
for p in paths:
    result = json.loads(p.read_text())
    assert result["status"] == "completed", result
    assert result["torch_peak_reserved_mib"] < 20 * 1024, result
ck = torch.load(Path("logs") / f"{root.name}_{phase}" / "ckpt_latest.pt", map_location="cpu", weights_only=False)
assert ck["iteration"] == expected, ck["iteration"]
assert all(torch.isfinite(t).all().item() for k in ("wm", "policy", "critic") for t in ck[k].values())
print("Validated", phase, "at", expected)
PY
}
test ! -e "logs/${SM_TAG}_capacity"
test ! -e "logs/${SM_TAG}_dual"
test ! -e "logs/${SM_TAG}_train"
wait_free 0
export CUDA_VISIBLE_DEVICES=0
run_logged capacity bash scripts/run_local.sh scripts/training/profile_train.py \
  --profile-gpu 0 --profile-output "$SM_RUN/capacity.json" "${SM_BASE[@]}" \
  --run-name "${SM_TAG}_capacity" --schedule.total-iterations "$SM_STOP" --schedule.max-hours 1
validate capacity "$SM_STOP" 1
wait_free 0 1
export CUDA_VISIBLE_DEVICES=0,1
run_logged dual bash scripts/run_local.sh -m torch.distributed.run --standalone --nproc-per-node 2 \
  scripts/training/profile_distributed.py --profile-dir "$SM_RUN/dual-profile" "${SM_BASE[@]}" \
  --run-name "${SM_TAG}_dual" --schedule.total-iterations "$SM_STOP" --schedule.max-hours 1
validate dual "$SM_STOP" 2
wait_free 0 1
nvidia-smi > "$SM_RUN/gpu-before-formal.txt"
ps -eo user,pid,comm > "$SM_RUN/processes-before-formal.txt"
run_logged train bash scripts/run_local.sh -m torch.distributed.run --standalone --nproc-per-node 2 \
  scripts/training/profile_distributed.py --profile-dir "$SM_RUN/train-profile" "${SM_BASE[@]}" \
  --run-name "${SM_TAG}_train" --schedule.total-iterations 320000 \
  --schedule.log-interval 100 --schedule.max-hours 3
