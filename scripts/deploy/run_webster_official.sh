#!/usr/bin/env bash
# Isolated simulation-only Webster trial. One unique run directory per trial.
set -eo pipefail
run=${1:?run directory}
trial="$run/trial"
mkdir "$trial"
source "$run/ros-env.sh"
[[ "$ROS_DOMAIN_ID" = 189 && "$ROS_LOCALHOST_ONLY" = 1 ]]
source_root="$run/source"
profile="$source_root/assets/robots/x2_ultra/profile.json"
ref=/home/2025-Long.xh/HHD/data/supermimic/data/video/webster_BV1Xy4y1g7xA_20261027/npz_reference_aligned_authorized_1h_20261028_r1/webster.npz
cmd_pid= state_pid= policy_pid= sim_pid=
cleanup() {
  if [[ -n "$sim_pid" ]]; then
    for child in $(pgrep -P "$sim_pid" -f '^./aima-sim-app' || true); do kill -TERM "$child" 2>/dev/null || true; done
    wait "$sim_pid" 2>/dev/null || true
  fi
  for pid in "$state_pid" "$cmd_pid" "$policy_pid"; do
    if [[ -n "$pid" ]]; then kill -INT "$pid" 2>/dev/null || true; fi
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM
/usr/bin/python3 "$source_root/scripts/deploy/official_sim_command.py" --profile "$profile" --port 19190 --sim-only >"$trial/command.log" 2>&1 & cmd_pid=$!
/usr/bin/python3 "$source_root/scripts/deploy/official_sim_state.py" --port 19189 --sim-only >"$trial/state.log" 2>&1 & state_pid=$!
(
 cd "$source_root"
 export SM_ROBOT=x2_ultra PYTHONPATH="$source_root/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
 exec /home/2025-Long.xh/HHD/venvs/supermimic/bin/python scripts/deploy/webster_official_policy.py \
   --profile "$profile" --dataset "$ref" --policy "$run/export/policy.pt" --out "$trial/policy" \
   --state-port 19189 --command-port 19190 --seconds 25 --sim-only
) >"$trial/policy.log" 2>&1 & policy_pid=$!
for i in $(seq 1 45); do
 if grep -q 'policy ready for simulator' "$trial/policy.log"; then break; fi
 kill -0 "$policy_pid" 2>/dev/null || { cat "$trial/policy.log"; exit 1; }
 sleep 1
done
grep -q 'policy ready for simulator' "$trial/policy.log"
bash "$run/start-sim.sh" >"$trial/sim.log" 2>&1 & sim_pid=$!
wait "$policy_pid"
policy_pid=
