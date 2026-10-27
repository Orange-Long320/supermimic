#!/usr/bin/env bash
# Bounded before/after test of one independently audited repaired reference.
set -euo pipefail
SM_ROOT=/home/2025-Long.xh/SSD/supermimic
SM_DATA=/home/2025-Long.xh/HHD/data/supermimic
SM_CANDIDATE=${1:?candidate directory required}
SM_TAG=${2:?unique run name required}
SM_GPU=${3:?physical GPU required}
[[ "$SM_TAG" =~ ^[a-zA-Z0-9_-]+$ ]]
cd "$SM_ROOT"
SM_RUN="$SM_DATA/logs/$SM_TAG"
mkdir "$SM_RUN"
exec > >(tee "$SM_RUN/controller.log") 2>&1
trap 'echo "$?" > "$SM_RUN/controller.exit"' EXIT
echo "$$" > "$SM_RUN/controller.pid"
export CUDA_VISIBLE_DEVICES="$SM_GPU" MUJOCO_EGL_DEVICE_ID="$SM_GPU"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$SM_DATA/cache" WARP_CACHE_PATH="$SM_DATA/cache/warp" TORCH_HOME="$SM_DATA/cache/torch"
unset PYTHONPATH PYTHONHOME
test -z "$(nvidia-smi -i "$SM_GPU" --query-compute-apps=pid --format=csv,noheader,nounits)"
nvidia-smi -i "$SM_GPU" > "$SM_RUN/gpu-before.txt"
.venv/bin/python - "$SM_CANDIDATE" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
assert json.load(open(p/'reference-qc.json'))['approved_for_bounded_simulation']
assert json.load(open(p/'repair-validation.json'))['approved_for_short_test']
PY
SM_INITIAL=${4:-logs/webster_formal_20261027_r2_train/ckpt_latest.pt}
SM_TARGET_ITER=$(.venv/bin/python - "$SM_INITIAL" "$SM_RUN/initial-checkpoint.json" <<'PY'
import hashlib,json,sys,torch
from pathlib import Path
p=Path(sys.argv[1]); c=torch.load(p,map_location='cpu',weights_only=False)
assert c['config']['robot']=='x2_ultra' and c['config']['env']['anchor_obs']
record={'source':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
        'iteration':c['iteration'],'updates':c['num_updates'],'additional_steps':200,
        'mode':'full controller/world-model/critic transfer; fresh replay and motion sampling statistics'}
Path(sys.argv[2]).write_text(json.dumps(record,indent=2))
print(c['iteration']+200)
PY
)
SM_MANIFEST="$SM_CANDIDATE/manifest_filtered.csv"
test -f "$SM_MANIFEST"
sha256sum "$SM_INITIAL" "$SM_CANDIDATE/webster.npz" > "$SM_RUN/inputs.sha256"
cp "${BASH_SOURCE[0]}" "$SM_RUN/launch.sh"
cp "$SM_CANDIDATE/repair-validation.json" "$SM_RUN/repair-validation.json"
bash scripts/run_local.sh scripts/robot.py x2_ultra scripts/evaluation/eval_benchmark.py \
  --ckpt "$SM_INITIAL" --npz-dir "$SM_CANDIDATE" --manifest "$SM_MANIFEST" --num-envs 1 --out "$SM_RUN/before.json" \
  > "$SM_RUN/before.log" 2>&1
SM_CMD=(bash scripts/run_local.sh scripts/training/profile_bounded_train.py \
  --torch-memory-fraction 0.2 --profile-gpu "$SM_GPU" --profile-output "$SM_RUN/profile.json" \
  --robot x2_ultra --run-name "${SM_TAG}_train" --resume "$SM_INITIAL" \
  --env.num-envs 256 --env.replay-capacity 512 --env.anchor-obs \
  --env.no-domain-randomization --env.no-obs-noise --env.no-reset-noise --env.action-noise-std 0 \
  --wm-trainer.batch-size 256 --wm-trainer.batches-per-update 2 --wm-trainer.compile None \
  --policy-trainer.batch-size 64 --policy-trainer.compile None --policy-trainer.no-obs-noise \
  --schedule.total-iterations "$SM_TARGET_ITER" --schedule.max-hours 0.1666667 \
  --schedule.lr-final-ratio 1 \
  --schedule.log-interval 50 --schedule.checkpoint-interval 100 \
  --data.npz-dir "$SM_CANDIDATE" --data.manifest "$SM_MANIFEST")
printf '%q ' "${SM_CMD[@]}" > "$SM_RUN/command.sh"; printf '\n' >> "$SM_RUN/command.sh"
"${SM_CMD[@]}" > "$SM_RUN/train.log" 2>&1 &
SM_PID=$!; echo "$SM_PID" > "$SM_RUN/train.pid"; wait "$SM_PID"
SM_FINAL="logs/${SM_TAG}_train/ckpt_latest.pt"
bash scripts/run_local.sh scripts/robot.py x2_ultra scripts/evaluation/eval_benchmark.py \
  --ckpt "$SM_FINAL" --npz-dir "$SM_CANDIDATE" --manifest "$SM_MANIFEST" --num-envs 1 --out "$SM_RUN/after.json" \
  > "$SM_RUN/after.log" 2>&1
bash scripts/run_local.sh scripts/evaluation/render_policy.py --robot x2_ultra \
  --ckpt "$SM_FINAL" --npz-dir "$SM_CANDIDATE" --clip webster --seconds 3 \
  --continue-after-failure --out "$SM_RUN/policy.mp4" > "$SM_RUN/render.log" 2>&1
echo 'Short test complete; inspect before.json, after.json and policy.mp4.'
