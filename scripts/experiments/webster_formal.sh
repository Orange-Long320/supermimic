#!/usr/bin/env bash
# First bounded Webster run: wait for a free card, then train/evaluate/record.
set -euo pipefail
SM_ROOT=/home/2025-Long.xh/SSD/supermimic
SM_DATA=/home/2025-Long.xh/HHD/data/supermimic
SM_TAG=${1:?unique run name required}
[[ "$SM_TAG" =~ ^[a-zA-Z0-9_-]+$ ]]
cd "$SM_ROOT"
SM_RUN="$SM_DATA/logs/$SM_TAG"
mkdir "$SM_RUN"
exec > >(tee "$SM_RUN/controller.log") 2>&1
trap 'SM_EXIT=$?; echo "$SM_EXIT" > "$SM_RUN/controller.exit"' EXIT
echo "$$" > "$SM_RUN/controller.pid"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$SM_DATA/cache"
export WARP_CACHE_PATH="$XDG_CACHE_HOME/warp" TORCH_HOME="$XDG_CACHE_HOME/torch"
unset PYTHONPATH PYTHONHOME CUDA_VISIBLE_DEVICES
SM_VIDEO="$SM_DATA/data/video/webster_BV1Xy4y1g7xA_20261027"
SM_NPZ="$SM_VIDEO/npz_final_50hz_r1"
SM_INIT=logs/x2_lafan_formal_20261027_0230_train/ckpt_latest.pt
SM_SMOKE="$SM_DATA/logs/webster_smoke_20261027_r2"
echo 'Waiting for smoke validation of the final collision-corrected data'
while [[ ! -f "$SM_SMOKE/controller.exit" ]]; do sleep 5; done
test "$(cat "$SM_SMOKE/controller.exit")" = 0
.venv/bin/python - "$SM_DATA/logs/webster_smoke_20261027_r2_train/ckpt_latest.pt" "$SM_RUN/smoke-verification.json" <<'PY'
import json,sys,torch
from pathlib import Path
checkpoint=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
finite={key:all(bool(torch.isfinite(value).all()) for value in checkpoint[key].values() if torch.is_tensor(value)) for key in ('wm','policy','critic')}
assert all(finite.values()) and checkpoint['iteration']==100 and checkpoint['num_updates']>0
report={'iteration':checkpoint['iteration'],'num_updates':checkpoint['num_updates'],'finite':finite,'data':str(checkpoint['config']['data']['npz_dir'])}
assert report['data'].endswith('/npz_final_50hz_r1')
Path(sys.argv[2]).write_text(json.dumps(report,indent=2)+'\n')
PY
test -f "$SM_NPZ/webster.npz"
.venv/bin/python - "$SM_NPZ/reference-qc.json" <<'PY'
import json,sys
assert json.load(open(sys.argv[1]))['approved_for_bounded_simulation']
PY
test -f "$SM_INIT"
cp scripts/experiments/webster_formal.sh "$SM_RUN/launch.sh"
cp "$SM_VIDEO/robot_final_50hz_r1/fit.json" "$SM_RUN/reference-fit.json"
cp "$SM_VIDEO/robot_arm_floor_r1/arm-floor-fit.json" "$SM_RUN/arm-floor-fit.json"
cp "$SM_NPZ/reference-qc.json" "$SM_RUN/reference-qc.json"
sha256sum "$SM_NPZ/webster.npz" "$SM_INIT" > "$SM_RUN/inputs.sha256"
SM_ARGS=(--robot x2_ultra --run-name "${SM_TAG}_train" \
  --env.num-envs 512 --env.replay-capacity 512 --env.anchor-obs \
  --wm-trainer.batch-size 256 --wm-trainer.batches-per-update 10 --wm-trainer.compile None \
  --policy-trainer.batch-size 128 --policy-trainer.compile None \
  --schedule.max-hours 3 --schedule.log-interval 100 --schedule.checkpoint-interval 1000 \
  --data.npz-dir "$SM_NPZ" --data.manifest "$SM_NPZ/manifest_filtered.csv" \
  --warmstart-world-model "$SM_INIT")
CUDA_VISIBLE_DEVICES='' bash scripts/run_local.sh scripts/training/train.py \
  --print-config "${SM_ARGS[@]}" > "$SM_RUN/config-preview.json"
wait_for_gpu() {
  local SM_PHASE=$1 SM_CARD SM_PIDS SM_FREE
  echo "Waiting for a free GPU 0/1 for $SM_PHASE; no sharing with LAFAN or other users"
  while true; do
    for SM_CARD in 0 1; do
      SM_PIDS=$(nvidia-smi -i "$SM_CARD" --query-compute-apps=pid --format=csv,noheader,nounits)
      SM_FREE=$(nvidia-smi -i "$SM_CARD" --query-gpu=memory.free --format=csv,noheader,nounits)
      if [[ -z "$SM_PIDS" && "$SM_FREE" -ge 16384 ]]; then
        SM_GPU=$SM_CARD
        export CUDA_VISIBLE_DEVICES="$SM_GPU" MUJOCO_EGL_DEVICE_ID="$SM_GPU"
        nvidia-smi -i "$SM_GPU" > "$SM_RUN/gpu-before-$SM_PHASE.txt"
        echo "Selected physical GPU $SM_GPU for $SM_PHASE"
        return
      fi
    done
    sleep 30
  done
}
wait_for_gpu train
sha256sum -c "$SM_RUN/inputs.sha256"
SM_CMD=(bash scripts/run_local.sh scripts/training/profile_bounded_train.py \
  --torch-memory-fraction 0.2 --profile-gpu "$SM_GPU" --profile-output "$SM_RUN/profile.json" "${SM_ARGS[@]}")
printf '%q ' "${SM_CMD[@]}" > "$SM_RUN/command.sh"; printf '\n' >> "$SM_RUN/command.sh"
"${SM_CMD[@]}" > "$SM_RUN/train.log" 2>&1 &
SM_PID=$!; echo "$SM_PID" > "$SM_RUN/train.pid"; wait "$SM_PID"
SM_CHECKPOINT="logs/${SM_TAG}_train/ckpt_latest.pt"
.venv/bin/python - "$SM_CHECKPOINT" "$SM_RUN/checkpoint-verification.json" <<'PY'
import hashlib,json,sys,torch
from pathlib import Path
p=Path(sys.argv[1]); checkpoint=torch.load(p,map_location='cpu',weights_only=False)
finite={key:all(bool(torch.isfinite(value).all()) for value in checkpoint[key].values() if torch.is_tensor(value)) for key in ('wm','policy','critic')}
assert all(finite.values()) and checkpoint['num_updates']>0
report={'iteration':checkpoint['iteration'],'num_updates':checkpoint['num_updates'],'finite':finite,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
Path(sys.argv[2]).write_text(json.dumps(report,indent=2)+'\n')
PY
wait_for_gpu eval
bash scripts/run_local.sh scripts/robot.py x2_ultra scripts/evaluation/eval_benchmark.py \
  --ckpt "$SM_CHECKPOINT" --npz-dir "$SM_NPZ" --num-envs 1 --out "$SM_RUN/evaluation.json" \
  > "$SM_RUN/evaluation.log" 2>&1
wait_for_gpu render
bash scripts/run_local.sh scripts/evaluation/render_policy.py --robot x2_ultra \
  --ckpt "$SM_CHECKPOINT" --npz-dir "$SM_NPZ" --clip webster --seconds 2 \
  --out "$SM_RUN/policy.mp4" > "$SM_RUN/render.log" 2>&1
echo 'First bounded training and evaluation complete; inspect evaluation.json and policy.mp4 for actual outcome.'
