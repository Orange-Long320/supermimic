#!/usr/bin/env bash
# Bounded workload comparison, not an algorithm-equivalence benchmark.
set -euo pipefail
out=/home/2025-Long.xh/HHD/data/supermimic/logs/unilab_full_20261029_r1
[[ ! -e "$out/supermimic-eager-compare" ]]
cd /home/2025-Long.xh/SSD/supermimic
export CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1
echo $$ > "$out/supermimic-eager-controller.pid"
date -u +%FT%TZ > "$out/supermimic-eager-start-utc.txt"
set +e
timeout --signal=INT --kill-after=30 900 .venv/bin/python \
  scripts/training/profile_train.py --profile-gpu 3 \
  --profile-output "$out/supermimic-eager-profile.json" \
  --robot x2_ultra --preset default --seed 1 \
  --data.clips dance1_subject2 --env.num-envs 256 --env.replay-capacity 128 \
  --env.no-obs-noise --env.no-reset-noise --env.no-domain-randomization \
  --env.action-noise-std 0 --wm-trainer.batch-size 512 \
  --policy-trainer.batch-size 512 --schedule.total-iterations 320 \
  --wm-trainer.compile None --policy-trainer.compile None \
  --schedule.log-interval 20 --schedule.checkpoint-interval 320 \
  --schedule.max-hours 0.25 --log-root "$out" --run-name supermimic-eager-compare \
  > "$out/supermimic-eager-console.log" 2>&1
status=$?
set -e
echo "$status" > "$out/supermimic-eager-exit-code.txt"
date -u +%FT%TZ > "$out/supermimic-eager-end-utc.txt"
exit "$status"
