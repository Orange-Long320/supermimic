#!/usr/bin/env bash
# CPU-only bounded local contact-pivot diagnostic; never launches training.
set -euo pipefail
cd /home/2025-Long.xh/SSD/supermimic
SM_TAG=${1:?unique run tag required}
[[ "$SM_TAG" =~ ^[a-zA-Z0-9_-]+$ ]]
SM_RUN="logs/$SM_TAG"
mkdir "$SM_RUN"
exec > "$SM_RUN/controller.log" 2>&1
trap 'echo "$?" > "$SM_RUN/controller.exit"' EXIT
echo "$$" > "$SM_RUN/controller.pid"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
unset PYTHONPATH PYTHONHOME
SM_SOURCE=logs/webster_takeoff_window_20261027_r1/source-window.npz
SM_OUTPUT="data/video/webster_BV1Xy4y1g7xA_20261027/$SM_TAG"
cp scripts/data/refine_webster_takeoff_forces.py "$SM_RUN/solver-source.py"
cp "${BASH_SOURCE[0]}" "$SM_RUN/launch.sh"
sha256sum "$SM_SOURCE" scripts/data/refine_webster_takeoff_forces.py > "$SM_RUN/inputs.sha256"
SM_CMD=(timeout --signal=TERM 600 nice -n 10 bash scripts/run_local.sh scripts/robot.py x2_ultra
  scripts/data/refine_webster_takeoff_forces.py --input "$SM_SOURCE" --out "$SM_OUTPUT"
  --takeoff-frame 11 --switch-frame 0 --landing-frame 14 --contact-patch --contact-anchor lowest
  --free-com --max-evaluations 80 --linear-iterations 1500 --linear-tolerance 1e-6)
printf '%q ' "${SM_CMD[@]}" > "$SM_RUN/command.sh"
printf '\n' >> "$SM_RUN/command.sh"
"${SM_CMD[@]}" > "$SM_RUN/solver.log" 2>&1 &
SM_PID=$!
echo "$SM_PID" > "$SM_RUN/solver.pid"
wait "$SM_PID"
echo 'Numerical fit complete. Full-trajectory recombination and independent audits remain required.'
