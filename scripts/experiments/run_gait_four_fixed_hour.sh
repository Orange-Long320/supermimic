#!/usr/bin/env bash
# User-selected final size: 8192 environments, both batches2048, one GPU/hour.
set -euo pipefail
R=${1:?isolated source snapshot}
O=${2:?unique run directory}
GPU=${3:?physical GPU}
DATA=${4:?pinned four-reference dataset}
INITIAL=${5:?same-dataset completed capacity checkpoint}
cd "$R"
mkdir "$O"
exec > >(tee "$O/controller.log") 2>&1
trap 'echo "$?" > "$O/controller.exit"' EXIT
echo "$$" > "$O/controller.pid"
cp "$0" "$O/launch.sh"
unset PYTHONPATH PYTHONHOME SM_ROBOT_SPEC SM_SOLE_BOX_CONFIG
export CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=egl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export XDG_CACHE_HOME=/home/2025-Long.xh/HHD/data/supermimic/cache
export WARP_CACHE_PATH="$XDG_CACHE_HOME/warp" TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor" TMPDIR="$XDG_CACHE_HOME/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
P=/home/2025-Long.xh/HHD/venvs/supermimic/bin/python
COMMON=("$P" "$R/scripts/experiments/train_gait_four.py" --spec "$R/configs/training/gait-four-1h.json" --dataset "$DATA" --gpu "$GPU" --num-envs 8192 --batch-size 2048)
wait_idle() {
  while [[ -n "$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; do
    echo "$(date -Is) waiting for GPU $GPU to become idle"
    sleep 30
  done
}
wait_idle
nvidia-smi > "$O/gpu-before.txt"
ps -eo user,pid,args > "$O/processes-before.txt"
sha256sum "$INITIAL" > "$O/initial-checkpoint.sha256"
"$P" -c 'import sys,torch;print(sys.executable,torch.__version__,torch.version.cuda,torch.cuda.is_available(),torch.cuda.get_device_name(0))' > "$O/environment.txt"
uv --no-config pip freeze --python "$P" > "$O/packages.txt"
printf '%q ' "${COMMON[@]}" --phase capacity --out "$O/capacity" --checkpoint "$INITIAL" > "$O/capacity-command.sh"
"${COMMON[@]}" --phase capacity --out "$O/capacity" --checkpoint "$INITIAL" > "$O/capacity.log" 2>&1 &
CHILD=$!
echo "$CHILD" > "$O/capacity.pid"
wait "$CHILD"
wait_idle
printf '%q ' "${COMMON[@]}" --phase train --out "$O/train_1h" --proof "$O/capacity/result.json" > "$O/train-command.sh"
"${COMMON[@]}" --phase train --out "$O/train_1h" --proof "$O/capacity/result.json" > "$O/train.log" 2>&1 &
CHILD=$!
echo "$CHILD" > "$O/train.pid"
wait "$CHILD"
