#!/usr/bin/env bash
set -euo pipefail
cd /home/2025-Long.xh/SSD/supermimic
SM_RUN=logs/webster_repair_20261027_r1
SM_VIDEO=data/video/webster_BV1Xy4y1g7xA_20261027
SM_REV=${1:?projection revision required, such as r3}
[[ "$SM_REV" =~ ^r[0-9]+$ ]]
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
unset PYTHONPATH PYTHONHOME
exec > "$SM_RUN/prepare-$SM_REV.log" 2>&1
trap 'echo "$?" > "$SM_RUN/prepare-$SM_REV.exit"' EXIT
echo "$$" > "$SM_RUN/prepare-$SM_REV.pid"
while [[ ! -f "$SM_RUN/projection-$SM_REV.exit" ]]; do sleep 5; done
test "$(cat "$SM_RUN/projection-$SM_REV.exit")" = 0
cp scripts/experiments/webster_repair_prepare.sh "$SM_RUN/prepare-$SM_REV.sh"
bash scripts/run_local.sh scripts/robot.py x2_ultra scripts/data/refine_arm_floor_clearance.py \
  --input "$SM_VIDEO/centroidal_repair_$SM_REV/webster.csv" --out "$SM_VIDEO/centroidal_arms_$SM_REV" \
  > "$SM_RUN/arms-$SM_REV.log" 2>&1
bash scripts/run_local.sh scripts/robot.py x2_ultra scripts/data/refine_webster_stance.py \
  --input "$SM_VIDEO/centroidal_arms_$SM_REV/webster.csv" --projection "$SM_VIDEO/centroidal_repair_$SM_REV/projection.json" \
  --lock-low-torque-joints --out "$SM_VIDEO/centroidal_stance_${SM_REV}_joint" > "$SM_RUN/stance-$SM_REV-joint.log" 2>&1
bash scripts/run_local.sh scripts/robot.py x2_ultra scripts/data/finalize_webster_projection.py \
  --input "$SM_VIDEO/centroidal_stance_${SM_REV}_joint/webster.csv" --contacts configs/video/webster-contacts.json \
  --out "$SM_VIDEO/centroidal_ready_$SM_REV" > "$SM_RUN/finalize-$SM_REV.log" 2>&1
echo 'Candidate prepared; independent collision/momentum/force audit still required.'
