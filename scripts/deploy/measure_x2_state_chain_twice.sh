#!/usr/bin/env bash
# Two identical state-chain samples to establish the run-to-run noise band,
# with load and producer CPU recorded per sample. Read-only w.r.t. motors.
set -eo pipefail
base=/home/agi/lxh-supermimic
r=$base/online_lio_20261102_hwclock_cpu045_r12
source /opt/ros/humble/setup.bash
source /home/agi/aimdk/install/setup.bash
source "$r/install/share/livox_ros_driver2/local_setup.bash"
source "$r/install/share/fast_lio/local_setup.bash"
export PYTHONPATH="$r/code/src:$base/python-cpu:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$r/install/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

for i in 1 2; do
  echo "=== sample $i baseline: $(cat /proc/loadavg) ==="
  rm -rf "$r/live-ab-$i"
  flock -n -F "$base/online-pelvis.lock" /usr/bin/python3 "$r/code/scripts/online_x2_lio.py" \
    --binary "$r/install/lib/fast_lio/fastlio_mapping" \
    --urdf "$r/assets/robot-pc2.urdf" --profile "$r/assets/fk-only/profile.json" \
    --extrinsics "$r/extrinsics.json" --out "$r/live-ab-$i" \
    --seconds 100 --state-clock imu-grid --freeze-startup-gc \
    --native-stamp-source lidar --timing-diagnostics > "$r/live-ab-$i.log" 2>&1 &
  producer=$!
  sleep 25
  python3 "$base/deploy-probe-20261102/state_probe.py" --seconds 60 \
    --out "$base/deploy-probe-20261102/ab-$i.json" > /dev/null
  echo "sample $i load at end: $(cat /proc/loadavg)"
  echo "sample $i producer cpu: $(ps -o %cpu= -p $producer 2>/dev/null || echo gone)"
  wait $producer 2>/dev/null || true
done
