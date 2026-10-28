#!/usr/bin/env bash
# Recover the inspected box45 clip in the existing isolated GEM environment.
set -euo pipefail
SM_ROOT=/home/2025-Long.xh/SSD/supermimic
SM_DATA=/home/2025-Long.xh/HHD/data/supermimic
SM_GEM=/home/2025-Long.xh/HHD/venvs/supermimic-gem
SM_RUN=${1:?unique run name required}
SM_GPU=${2:?physical GPU index required}
[[ "$SM_RUN" =~ ^[a-zA-Z0-9_-]+$ && "$SM_GPU" =~ ^[0-3]$ ]]
cd "$SM_ROOT"
SM_OUT="$SM_DATA/logs/$SM_RUN"
mkdir "$SM_OUT"
exec > >(tee -a "$SM_OUT/controller.log") 2>&1
trap 'SM_EXIT=$?; echo "$SM_EXIT" > "$SM_OUT/controller.exit"' EXIT
echo "$$" > "$SM_OUT/controller.pid"
date -Is
hostname
SM_UUID=$(nvidia-smi -i "$SM_GPU" --query-gpu=uuid --format=csv,noheader)
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "$SM_OUT/gpu-before.csv"
if grep -Fq "$SM_UUID" "$SM_OUT/gpu-before.csv"; then
  echo "GPU $SM_GPU has a compute process; refusing to start"; exit 1
fi
export CUDA_VISIBLE_DEVICES="$SM_GPU"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="$SM_DATA/cache/gem-x"
export TORCH_HOME="$XDG_CACHE_HOME/torch" HF_HOME="$XDG_CACHE_HOME/huggingface"
export TMPDIR="$XDG_CACHE_HOME/tmp" WARP_CACHE_PATH="$XDG_CACHE_HOME/warp"
unset PYTHONPATH PYTHONHOME
test -d "$TMPDIR"
test "$(cat vendor/gem-x/SOURCE_COMMIT)" = 20555befe2fbbe5b2015e1fff7ff214dc18ea9d6
cp scripts/experiments/recover_box45.sh "$SM_OUT/launch.sh"
cp configs/video/box45-IMG4868.json "$SM_OUT/task.json"
cp "$SM_DATA/assets/gem-x/READY.json" "$SM_OUT/assets.json"
cp vendor/gem-x/SOURCE_COMMIT "$SM_OUT/gem-source.txt"
uv --no-config pip freeze --python "$SM_GEM/bin/python" > "$SM_OUT/environment.txt"
"$SM_GEM/bin/python" - "$SM_DATA" "$SM_OUT" <<'PY'
import hashlib,json,sys
from pathlib import Path
import torch
from gem.inference.body_video import load_body_video_config
data,out=map(Path,sys.argv[1:])
task=json.loads((out/'task.json').read_text())
source=data/'data/video/climb_box45_IMG4868_20261028_r1/raw/IMG_4868.MOV'
assert hashlib.sha256(source.read_bytes()).hexdigest()==task['source']['sha256']
config=load_body_video_config(data/'assets/gem-x/weights/body_video_native77.yaml')
config['pose']['batch_size']=1
config['observations']['batch_size']=1
for section,keys in [('pose',['checkpoint']),('observations',['checkpoint']),('detector',['checkpoint']),('motion',['checkpoint','stats'])]:
    for key in keys:
        assert Path(config[section][key]).is_file(), (section,key)
(out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
(out/'environment.json').write_text(json.dumps({'python':sys.executable,'torch':torch.__version__,'cuda_package':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'physical_gpu':__import__('os').environ['CUDA_VISIBLE_DEVICES']},indent=2)+'\n')
PY
SM_SOURCE="$SM_DATA/data/video/climb_box45_IMG4868_20261028_r1/raw/IMG_4868.MOV"
SM_CMD=("$SM_GEM/bin/gem-infer" motion-video --task body --input "$SM_SOURCE"
  --start 0 --duration 4.4 --config "$SM_OUT/config.json" --device cuda:0
  --output "$SM_OUT/raw_motion.pt")
printf '%q ' "${SM_CMD[@]}" > "$SM_OUT/command.sh"
printf '\n' >> "$SM_OUT/command.sh"
"${SM_CMD[@]}" > "$SM_OUT/inference.log" 2>&1 &
SM_PID=$!
echo "$SM_PID" > "$SM_OUT/inference.pid"
wait "$SM_PID"
export CUDA_VISIBLE_DEVICES=''
"$SM_GEM/bin/python" scripts/data/export_gem_bvh.py --input "$SM_OUT/raw_motion.pt" \
  --soma-assets "$SM_DATA/assets/gem-x/soma" --out "$SM_OUT/bvh" --name box45 > "$SM_OUT/export.log" 2>&1
/home/2025-Long.xh/HHD/venvs/supermimic-soma/bin/python scripts/data/check_bvh_roundtrip.py \
  --directory "$SM_OUT/bvh" --name box45 > "$SM_OUT/import-check.log" 2>&1
"$SM_GEM/bin/python" scripts/evaluation/render_gem_skeleton.py --input "$SM_OUT/raw_motion.pt" \
  --soma-assets "$SM_DATA/assets/gem-x/soma" --out "$SM_OUT/preview" > "$SM_OUT/preview.log" 2>&1
echo 'Recovery complete; box-relative geometry and robot feasibility still require review.'
date -Is
