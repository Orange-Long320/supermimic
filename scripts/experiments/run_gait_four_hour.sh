#!/usr/bin/env bash
# Run in an immutable source snapshot; one GPU, capacity then a one-hour fit.
set -euo pipefail
R=${1:?isolated source root}
O=${2:?unique controller directory}
GPU=${3:?explicit physical GPU}
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
SPEC="$R/configs/training/gait-four-1h.json"
DATA="$O/dataset"
COMMON=("$P" "$R/scripts/experiments/train_gait_four.py" --spec "$SPEC" --dataset "$DATA" --gpu "$GPU")
"${COMMON[@]}" --phase prepare --out "$O/unused" > "$O/prepare.log" 2>&1
"$P" -c 'import sys,torch;print(sys.executable,torch.__version__,torch.version.cuda,torch.cuda.is_available(),torch.cuda.get_device_name(0))' > "$O/environment.txt"
uv --no-config pip freeze --python "$P" > "$O/packages.txt"
wait_idle() {
  while [[ -n "$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; do
    echo "$(date -Is) waiting for GPU $GPU to become idle"
    sleep 30
  done
}
wait_idle
nvidia-smi > "$O/gpu-before.txt"
ps -eo user,pid,args > "$O/processes-before.txt"
printf '%q ' "${COMMON[@]}" --phase capacity --out "$O/capacity" > "$O/capacity-command.sh"
"${COMMON[@]}" --phase capacity --out "$O/capacity" > "$O/capacity.log" 2>&1 &
echo "$!" > "$O/capacity.pid"
wait "$!"
wait_idle
printf '%q ' "${COMMON[@]}" --phase train --out "$O/train_1h" --proof "$O/capacity/result.json" > "$O/train-command.sh"
"${COMMON[@]}" --phase train --out "$O/train_1h" --proof "$O/capacity/result.json" > "$O/train.log" 2>&1 &
echo "$!" > "$O/train.pid"
wait "$!"
