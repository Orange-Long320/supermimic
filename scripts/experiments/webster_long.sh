#!/usr/bin/env bash
# Single-GPU continuation; use a separate short invocation to measure capacity first.
set -euo pipefail
SM_ROOT=/home/2025-Long.xh/SSD/supermimic
SM_DATA=/home/2025-Long.xh/HHD/data/supermimic
SM_TAG=${1:?unique run name required}
SM_ENVS=${2:?measured environment count required}
SM_POLICY_BATCH=${3:?policy batch required}
SM_HOURS=${4:?time budget required}
SM_TARGET=${5:-320000}
SM_INITIAL=${6:-logs/webster_transfer_short_20261027_r1_train/ckpt_latest.pt}
SM_NPZ=${7:-$SM_DATA/data/video/webster_BV1Xy4y1g7xA_20261027/npz_repaired_ease_r1}
SM_GPU=${8:-0}
SM_RENDER_SECONDS=${9:-3}
[[ "$SM_TAG" =~ ^[a-zA-Z0-9_-]+$ ]]
[[ "$SM_GPU" =~ ^[0-9]+$ ]]
cd "$SM_ROOT"
SM_RUN="$SM_DATA/logs/$SM_TAG"
test ! -e "logs/${SM_TAG}_train"
mkdir "$SM_RUN"
exec > >(tee "$SM_RUN/controller.log") 2>&1
trap 'echo "$?" > "$SM_RUN/controller.exit"' EXIT
echo "$$" > "$SM_RUN/controller.pid"
date -Is > "$SM_RUN/started.txt"
export CUDA_VISIBLE_DEVICES="$SM_GPU" MUJOCO_EGL_DEVICE_ID="$SM_GPU"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$SM_DATA/cache" WARP_CACHE_PATH="$SM_DATA/cache/warp" TORCH_HOME="$SM_DATA/cache/torch"
unset PYTHONPATH PYTHONHOME
test -z "$(nvidia-smi -i "$SM_GPU" --query-compute-apps=pid --format=csv,noheader,nounits)"
nvidia-smi -i "$SM_GPU" > "$SM_RUN/gpu-before.txt"
cp "${BASH_SOURCE[0]}" "$SM_RUN/launch.sh"
for SM_METADATA in reference-qc.json repair-validation.json provenance.json; do
  if [[ -f "$SM_NPZ/$SM_METADATA" ]]; then cp "$SM_NPZ/$SM_METADATA" "$SM_RUN/"; fi
done
cp UPSTREAM.md "$SM_RUN/"
sha256sum "$SM_INITIAL" "$SM_NPZ/webster.npz" "$SM_NPZ/manifest_filtered.csv" > "$SM_RUN/inputs.sha256"
.venv/bin/python - "$SM_INITIAL" "$SM_RUN/environment-and-input.json" <<'PY'
import importlib.metadata as m, json, sys, torch
from pathlib import Path
c=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
assert c['config']['robot']=='x2_ultra' and c['config']['env']['anchor_obs']
assert all(torch.isfinite(v).all().item() for k in ('wm','policy','critic') for v in c[k].values() if torch.is_tensor(v))
record={'python':sys.version,'executable':str(Path(sys.executable).resolve()),
        'packages':{p:m.version(p) for p in ('torch','mujoco','mujoco-warp','warp-lang')},
        'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
        'initial_checkpoint':str(Path(sys.argv[1]).resolve()),'iteration':c['iteration'],'updates':c['num_updates'],
        'limitation':'Exploratory training; input provenance and physical limitations are retained separately. Success requires independent full-motion evaluation.'}
Path(sys.argv[2]).write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
PY
# No Git metadata in this uploaded checkout: preserve exact training source hashes.
find src/supermimic scripts/training -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum > "$SM_RUN/source.sha256"
SM_ARGS=(--robot x2_ultra --run-name "${SM_TAG}_train" --resume "$SM_INITIAL"
  --env.num-envs "$SM_ENVS" --env.replay-capacity 512 --env.anchor-obs
  --env.no-domain-randomization --env.no-obs-noise --env.no-reset-noise --env.action-noise-std 0
  --wm-trainer.batch-size 2048 --wm-trainer.batches-per-update 2 --wm-trainer.compile None
  --policy-trainer.batch-size "$SM_POLICY_BATCH" --policy-trainer.compile None --policy-trainer.no-obs-noise
  --schedule.total-iterations "$SM_TARGET" --schedule.max-hours "$SM_HOURS" --schedule.lr-final-ratio 1
  --schedule.log-interval 50 --schedule.checkpoint-interval 500
  --data.npz-dir "$SM_NPZ" --data.manifest "$SM_NPZ/manifest_filtered.csv")
bash scripts/run_local.sh scripts/training/train.py --print-config "${SM_ARGS[@]}" > "$SM_RUN/config-preview.json"
sha256sum -c "$SM_RUN/inputs.sha256"
SM_CMD=(bash scripts/run_local.sh scripts/training/profile_train.py --profile-gpu "$SM_GPU" --profile-output "$SM_RUN/profile.json" "${SM_ARGS[@]}")
printf '%q ' "${SM_CMD[@]}" > "$SM_RUN/command.sh"; printf '\n' >> "$SM_RUN/command.sh"
"${SM_CMD[@]}" > "$SM_RUN/train.log" 2>&1 &
SM_PID=$!; echo "$SM_PID" > "$SM_RUN/train.pid"; wait "$SM_PID"
.venv/bin/python - "logs/${SM_TAG}_train/ckpt_latest.pt" "$SM_RUN/checkpoint-verification.json" <<'PY'
import json,sys,torch
from pathlib import Path
c=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
finite={k:all(torch.isfinite(v).all().item() for v in c[k].values() if torch.is_tensor(v)) for k in ('wm','policy','critic')}
assert all(finite.values())
Path(sys.argv[2]).write_text(json.dumps({'iteration':c['iteration'],'num_updates':c['num_updates'],'finite':finite},indent=2)+'\n')
PY
# Profiling invocations stop here; long runs also evaluate and record the final policy.
if [[ "$SM_TARGET" -ge 320000 ]]; then
  bash scripts/run_local.sh scripts/robot.py x2_ultra scripts/evaluation/eval_benchmark.py \
    --ckpt "logs/${SM_TAG}_train/ckpt_latest.pt" --npz-dir "$SM_NPZ" --manifest "$SM_NPZ/manifest_filtered.csv" \
    --num-envs 1 --out "$SM_RUN/evaluation.json" > "$SM_RUN/evaluation.log" 2>&1
  bash scripts/run_local.sh scripts/evaluation/render_policy.py --robot x2_ultra \
    --ckpt "logs/${SM_TAG}_train/ckpt_latest.pt" --npz-dir "$SM_NPZ" --clip webster --seconds "$SM_RENDER_SECONDS" \
    --continue-after-failure --out "$SM_RUN/policy.mp4" > "$SM_RUN/render.log" 2>&1
fi
