#!/usr/bin/env bash
set -euo pipefail
r=/home/2025-Long.xh/HHD/data/supermimic/logs/unilab_supermimic_warp_20261029_r1
s=/home/2025-Long.xh/HHD/data/supermimic/logs/unilab_full_20261029_r1/prepared-x2-r3/x2-scene.xml
i=/home/2025-Long.xh/HHD/data/supermimic/logs/unilab_probe_20261029_r3/initial.npz
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export XDG_CACHE_HOME=/home/2025-Long.xh/HHD/data/supermimic/cache
echo $$ > "$r/controller.pid"
trap 'code=$?; echo "$code" > "$r/controller.exit"' EXIT
for name in supermimic supermimic-unilab; do
  occupants=$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader,nounits)
  if [[ -n ${occupants//[[:space:]]/} ]]; then
    echo "GPU 1 has compute processes; refusing to start: $occupants" >&2
    exit 1
  fi
  date -u +%FT%TZ > "$r/$name.start"
  timeout --signal=INT --kill-after=20 300 \
    "/home/2025-Long.xh/HHD/venvs/$name/bin/python" "$r/audit_warp_trajectory.py" \
    --gpu 1 --scene "$s" --initial "$i" --out "$r/$name.npz" \
    > "$r/$name.log" 2>&1
  date -u +%FT%TZ > "$r/$name.end"
done
