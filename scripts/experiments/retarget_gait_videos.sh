#!/usr/bin/env bash
# Server-only: independent video recovery and X2 reference candidates.
set -euo pipefail
R=/home/2025-Long.xh/SSD/supermimic
D=/home/2025-Long.xh/HHD/data/supermimic
G=/home/2025-Long.xh/HHD/venvs/supermimic-gem
P=/home/2025-Long.xh/HHD/venvs/supermimic/bin/python
RUN=${1:?unique run name}
GPU=${2:?explicit GPU index}
[[ "$RUN" =~ ^[a-zA-Z0-9_-]+$ && "$GPU" =~ ^[0-9]+$ ]]
O="$D/logs/$RUN"
mkdir "$O"
exec > >(tee -a "$O/controller.log") 2>&1
trap 'echo "$?" > "$O/controller.exit"' EXIT
echo "$$" > "$O/controller.pid"
cd "$R"
unset PYTHONPATH PYTHONHOME
export CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="$D/cache/gem-x" HF_HOME="$D/cache/gem-x/huggingface" TORCH_HOME="$D/cache/gem-x/torch" WARP_CACHE_PATH="$D/cache/gem-x/warp" TMPDIR="$D/cache/gem-x/tmp"
export MUJOCO_GL=osmesa
cp scripts/experiments/retarget_gait_videos.sh "$O/launch.sh"
cp "$D/logs/webster_gem_20261027_r2/config.json" "$O/config.json"
cp "$D/assets/gem-x/READY.json" "$O/assets.json"
date -Is
nvidia-smi > "$O/gpu-before.txt"
uv --no-config pip freeze --python "$G/bin/python" > "$O/gem-environment.txt"
for N in 4880 4881 4882 4883; do
  C="$O/IMG_$N"
  mkdir "$C"
  V="$D/data/video/gait_IMG4880_4883_20261029_r1/raw/IMG_$N.MOV"
  sha256sum "$V" > "$C/source.sha256"
  CMD=("$G/bin/gem-infer" motion-video --task body --input "$V" --config "$O/config.json" --device cuda --output "$C/raw_motion.pt")
  printf '%q ' "${CMD[@]}" > "$C/command.sh"
  printf '\n' >> "$C/command.sh"
  "${CMD[@]}" > "$C/inference.log" 2>&1
  "$G/bin/python" scripts/data/export_gem_bvh.py --input "$C/raw_motion.pt" --soma-assets "$D/assets/gem-x/soma" --out "$C/bvh" --name "IMG_$N" > "$C/export.log" 2>&1
  "$P" scripts/data/retarget.py --robot x2_ultra --input "$C/bvh" --output "$C/retarget" --batch-size 1 > "$C/retarget.log" 2>&1
  printf 'filename\nIMG_%s\n' "$N" > "$C/manifest.csv"
  "$P" scripts/data/prepare_lafan.py --robot x2_ultra --source "$C/retarget" --dest "$C/npz" --manifest "$C/manifest.csv" --workers 1 --input-fps 30 --audit "$C/preparation.json" > "$C/prepare.log" 2>&1
  "$P" scripts/evaluation/render_motion_reference.py --robot x2_ultra --npz "$C/npz/IMG_$N.npz" --out "$C/x2-reference.mp4" --follow-root > "$C/render.log" 2>&1
  echo complete > "$C/status.txt"
done
