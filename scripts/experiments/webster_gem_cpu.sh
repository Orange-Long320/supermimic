#!/usr/bin/env bash
# Recover a single source clip on CPU; stop at inspectable FK/BVH artifacts.
set -euo pipefail
SM_ROOT=/home/2025-Long.xh/SSD/supermimic
SM_DATA=/home/2025-Long.xh/HHD/data/supermimic
SM_ENV=/home/2025-Long.xh/HHD/venvs/supermimic-gem
SM_RUN=${1:?unique run name required}
[[ "$SM_RUN" =~ ^[a-zA-Z0-9_-]+$ ]]
cd "$SM_ROOT"
SM_OUT="$SM_DATA/logs/$SM_RUN"
mkdir "$SM_OUT"
exec > >(tee -a "$SM_OUT/controller.log") 2>&1
trap 'SM_EXIT=$?; echo "$SM_EXIT" > "$SM_OUT/controller.exit"' EXIT
echo "$$" > "$SM_OUT/controller.pid"
date -Is
hostname
echo 'Waiting for the isolated environment and verified transferred assets'
while true; do
  if [[ -f "$SM_DATA/logs/webster-gem-setup-20261027/install-r3.exit" ]]; then
    test "$(cat "$SM_DATA/logs/webster-gem-setup-20261027/install-r3.exit")" = 0
    if [[ -f "$SM_DATA/assets/gem-x/READY.json" ]]; then break; fi
  fi
  sleep 10
done
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="$SM_DATA/cache/gem-x"
export WARP_CACHE_PATH="$XDG_CACHE_HOME/warp" TORCH_HOME="$XDG_CACHE_HOME/torch"
export HF_HOME="$XDG_CACHE_HOME/huggingface" TMPDIR="$XDG_CACHE_HOME/tmp"
mkdir -p "$TMPDIR"
unset PYTHONPATH PYTHONHOME
cp "$SM_DATA/assets/gem-x/READY.json" "$SM_OUT/assets.json"
cp "$SM_ROOT/vendor/gem-x/SOURCE_COMMIT" "$SM_OUT/source_commit.txt"
cp "$SM_ROOT/scripts/experiments/webster_gem_cpu.sh" "$SM_OUT/launch.sh"
uv --no-config pip freeze --python "$SM_ENV/bin/python" > "$SM_OUT/environment.txt"
"$SM_ENV/bin/python" - "$SM_DATA/assets/gem-x/weights/body_video_native77.yaml" "$SM_OUT/config.json" <<'PY'
import json,sys
from gem.inference.body_video import load_body_video_config
config=load_body_video_config(sys.argv[1])
config['pose']['batch_size']=1
config['observations']['batch_size']=1
with open(sys.argv[2],'x') as f: json.dump(config,f,indent=2)
PY
SM_SOURCE="$SM_DATA/data/video/webster_BV1Xy4y1g7xA_20261027/raw/BV1Xy4y1g7xA.mp4"
sha256sum "$SM_SOURCE" > "$SM_OUT/source_video.sha256"
SM_CMD=("$SM_ENV/bin/gem-infer" motion-video --task body --input "$SM_SOURCE" \
  --start 256.20 --duration 1.40 --config "$SM_OUT/config.json" --device cpu \
  --output "$SM_OUT/raw_motion.pt")
printf '%q ' "${SM_CMD[@]}" > "$SM_OUT/command.sh"
printf '\n' >> "$SM_OUT/command.sh"
echo 'Starting bounded CPU motion recovery; GPU access disabled'
"${SM_CMD[@]}" > "$SM_OUT/inference.log" 2>&1 &
SM_PID=$!
echo "$SM_PID" > "$SM_OUT/inference.pid"
wait "$SM_PID"
"$SM_ENV/bin/python" scripts/data/export_gem_bvh.py \
  --input "$SM_OUT/raw_motion.pt" --soma-assets "$SM_DATA/assets/gem-x/soma" \
  --out "$SM_OUT/bvh" > "$SM_OUT/export.log" 2>&1
/home/2025-Long.xh/HHD/venvs/supermimic-soma/bin/python scripts/data/check_bvh_roundtrip.py \
  --directory "$SM_OUT/bvh" > "$SM_OUT/import-check.log" 2>&1
"$SM_ENV/bin/python" scripts/evaluation/render_gem_skeleton.py \
  --input "$SM_OUT/raw_motion.pt" --soma-assets "$SM_DATA/assets/gem-x/soma" \
  --out "$SM_OUT/preview" > "$SM_OUT/preview.log" 2>&1
echo 'Recovery/export complete. Visual motion quality and robot feasibility still require review.'
date -Is
