#!/usr/bin/env bash
# One isolated official-simulator experiment, never a hardware launcher.
set -eo pipefail
run=${1:?run directory}
name=${2:?new trial name}
seconds=${3:?wall duration}
fade=${4:-baseline}
trial="$run/$name"
mkdir "$trial"
source "$run/ros-env.sh"
source_root="$run/source"
profile="$source_root/assets/robots/x2_ultra/profile.json"
dataset=/home/2025-Long.xh/HHD/data/supermimic/logs/five_gait_stops_5h_20261030_r2/dataset
model=/home/2025-Long.xh/HHD/data/supermimic/logs/x2_deploy_foundation_20261030_r1/export/policy.pt
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
/usr/bin/python3 "$source_root/scripts/deploy/official_sim_command.py" --profile "$profile" --port 19180 --sim-only >"$trial/command.log" 2>&1 & cmd_pid=$!
/usr/bin/python3 "$source_root/scripts/deploy/official_sim_state.py" --port 19179 --sim-only >"$trial/state.log" 2>&1 & state_pid=$!
args=()
if [[ "$fade" != baseline ]]; then args+=(--jog-brake-fade "$fade"); fi
(
  cd "$source_root"
  export SM_ROBOT=x2_ultra PYTHONPATH="$source_root/src" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
  exec /home/2025-Long.xh/HHD/venvs/supermimic/bin/python scripts/deploy/official_sim_policy.py \
    --profile "$profile" --dataset "$dataset" --policy "$model" --out "$trial/policy" \
    --state-port 19179 --command-port 19180 --seconds "$seconds" --schedule "$run/schedule.json" \
    --compact-log --sim-only "${args[@]}"
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
