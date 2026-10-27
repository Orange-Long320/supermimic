#!/usr/bin/env bash
# A bounded smoke may share a GPU only with this user's SuperMimic processes.
set -euo pipefail
SM_ROOT=/home/2025-Long.xh/SSD/supermimic
SM_DATA=/home/2025-Long.xh/HHD/data/supermimic
SM_GPU=${1:?physical GPU required}
SM_TAG=${2:?unique run name required}
[[ "$SM_TAG" =~ ^[a-zA-Z0-9_-]+$ ]]
cd "$SM_ROOT"
SM_RUN="$SM_DATA/logs/$SM_TAG"
mkdir "$SM_RUN"
exec > >(tee "$SM_RUN/controller.log") 2>&1
trap 'SM_EXIT=$?; echo "$SM_EXIT" > "$SM_RUN/controller.exit"' EXIT
echo "$$" > "$SM_RUN/controller.pid"
export CUDA_VISIBLE_DEVICES="$SM_GPU"
export XDG_CACHE_HOME="$SM_DATA/cache"
export WARP_CACHE_PATH="$XDG_CACHE_HOME/warp" TORCH_HOME="$XDG_CACHE_HOME/torch"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
unset PYTHONPATH PYTHONHOME
.venv/bin/python - "$SM_GPU" "$SM_RUN/gpu-before.json" <<'PY'
import json,os,subprocess,sys
gpu,out=sys.argv[1:]
free=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert free>=8192, f'Only {free} MiB free; bounded smoke requires >=8192 MiB'
pids=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()
processes=[]
for pid in pids:
 uid=int(subprocess.check_output(['ps','-o','uid=','-p',pid],text=True).strip())
 command=subprocess.check_output(['ps','-o','args=','-p',pid],text=True).strip()
 assert uid==os.getuid() and '/supermimic/' in command, f'GPU has another task: {pid}'
 processes.append({'pid':int(pid),'uid':uid,'command':command})
with open(out,'x') as f: json.dump({'gpu':gpu,'free_mib':free,'existing_processes':processes,'shared':bool(pids),'torch_memory_fraction':0.15,'note':'Device total memory and throughput include the ongoing LAFAN run; report only per-process Torch peak as smoke memory.'},f,indent=2)
print('GPU guard passed:',gpu,'free MiB',free,'existing project processes',pids)
PY
SM_NPZ="$SM_DATA/data/video/webster_BV1Xy4y1g7xA_20261027/npz_final_50hz_r1"
.venv/bin/python - "$SM_NPZ/reference-qc.json" <<'PY'
import json,sys
assert json.load(open(sys.argv[1]))['approved_for_bounded_simulation']
PY
SM_CMD=(bash scripts/run_local.sh scripts/training/profile_bounded_smoke.py \
  --torch-memory-fraction 0.15 --profile-gpu "$SM_GPU" --profile-output "$SM_RUN/profile.json" \
  --robot x2_ultra --preset smoke --run-name "${SM_TAG}_train" --env.anchor-obs \
  --data.npz-dir "$SM_NPZ" --data.manifest "$SM_NPZ/manifest_filtered.csv")
printf '%q ' "${SM_CMD[@]}" > "$SM_RUN/command.sh"
printf '\n' >> "$SM_RUN/command.sh"
cp scripts/experiments/webster_smoke.sh "$SM_RUN/launch.sh"
"${SM_CMD[@]}" > "$SM_RUN/train.log" 2>&1 &
SM_PID=$!
echo "$SM_PID" > "$SM_RUN/train.pid"
wait "$SM_PID"
echo 'Smoke finished; this does not establish flip success or convergence.'
