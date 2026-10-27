#!/usr/bin/env bash
# Exact project CPU physics only. Does not launch training or allocate a GPU.
set -euo pipefail
cd /home/2025-Long.xh/SSD/supermimic
SM_TAG=${1:?unique run tag}
SM_PHASE=${2:-takeoff}
SM_WARM=${3:-}
SM_GUIDE=${4:-data/video/webster_BV1Xy4y1g7xA_20261027/npz_repaired_ease_r1/webster.npz}
[[ "$SM_TAG" =~ ^[a-zA-Z0-9_-]+$ ]]
[[ "$SM_PHASE" == takeoff || "$SM_PHASE" == full ]]
SM_RUN="logs/$SM_TAG"
mkdir "$SM_RUN"
exec > "$SM_RUN/controller.log" 2>&1
trap 'echo "$?" > "$SM_RUN/controller.exit"' EXIT
echo "$$" > "$SM_RUN/controller.pid"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
unset PYTHONPATH PYTHONHOME
SM_CPU=$(.venv/bin/python - "$SM_RUN/cpu-selection.json" <<'PY'
import json,os,sys,time
from pathlib import Path
def sample():
    rows={}
    for line in Path('/proc/stat').read_text().splitlines():
        fields=line.split()
        if fields[0].startswith('cpu') and fields[0][3:].isdigit():
            values=list(map(int,fields[1:])); rows[int(fields[0][3:])]=(sum(values[:8]),values[3]+values[4])
    return rows
a=sample();time.sleep(.5);b=sample()
allowed=os.sched_getaffinity(0)
busy={k:1-(b[k][1]-a[k][1])/max(b[k][0]-a[k][0],1) for k in allowed}
core=min(busy,key=busy.get)
Path(sys.argv[1]).write_text(json.dumps({'selected_cpu':core,'sampled_busy_fraction':busy[core],
    'nice':19,'threads':1,'cuda_visible_devices':''},indent=2))
print(core)
PY
)
cp scripts/data/optimize_webster_policy_cpu.py "$SM_RUN/designer-source.py"
cp "${BASH_SOURCE[0]}" "$SM_RUN/launch.sh"
SM_CHECKPOINT=logs/webster_transfer_short_20261027_r1_train/ckpt_latest.pt
SM_SOURCE="$SM_GUIDE"
sha256sum "$SM_CHECKPOINT" "$SM_SOURCE" scripts/data/optimize_webster_policy_cpu.py > "$SM_RUN/inputs.sha256"
SM_SECONDS=1.2
[[ "$SM_PHASE" != full ]] || SM_SECONDS=2
SM_CMD=(taskset -c "$SM_CPU" nice -n 19 timeout --signal=TERM --kill-after=30 2100
  bash scripts/run_local.sh scripts/robot.py x2_ultra scripts/data/optimize_webster_policy_cpu.py
  --ckpt "$SM_CHECKPOINT" --source "$SM_SOURCE" --out "$SM_RUN/design"
  --phase "$SM_PHASE" --seconds "$SM_SECONDS" --num-envs 4 --population 24 --iterations 80 --max-seconds 1800)
if [[ -n "$SM_WARM" ]]; then
  sha256sum "$SM_WARM" >> "$SM_RUN/inputs.sha256"
  SM_CMD+=(--warmstart "$SM_WARM")
fi
printf '%q ' "${SM_CMD[@]}" > "$SM_RUN/command.sh"
printf '\n' >> "$SM_RUN/command.sh"
"${SM_CMD[@]}" > "$SM_RUN/designer.log" 2>&1 &
SM_PID=$!
echo "$SM_PID" > "$SM_RUN/worker.pid"
wait "$SM_PID"
echo 'CPU design budget complete. Inspect full physical gates before any dataset export or training.'
