#!/usr/bin/env bash
# One isolated official-simulator experiment, never a hardware launcher.
set -eo pipefail
run=${1:?run directory}
name=${2:?new trial name}
seconds=${3:?wall duration}
fade=${4:-baseline}
gain=${5:-1.0}
schedule=${6:-"$run/schedule.json"}
quiet=${7:-0.0}
trial="$run/$name"
mkdir "$trial"
source "$run/ros-env.sh"
ros_domain=${ROS_DOMAIN_ID:-179}
state_port=${SM_SIM_STATE_PORT:-19179}
command_port=${SM_SIM_COMMAND_PORT:-19180}
source_root="$run/source"
profile="$source_root/assets/robots/x2_ultra/profile.json"
dataset=${8:-/home/2025-Long.xh/HHD/data/supermimic/logs/five_gait_stops_5h_20261030_r2/dataset}
model=${9:-/home/2025-Long.xh/HHD/data/supermimic/logs/x2_deploy_foundation_20261030_r1/export/policy.pt}
[[ -f "$model" ]] || { echo "Policy does not exist: $model" >&2; exit 1; }
[[ -f "$dataset/manifest.json" ]] || { echo "Dataset manifest missing: $dataset" >&2; exit 1; }
sha256sum "$model" "$dataset/manifest.json" > "$trial/input-sha256.txt"
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
/usr/bin/python3 "$source_root/scripts/deploy/official_sim_command.py" --profile "$profile" --port "$command_port" --ros-domain "$ros_domain" --sim-only >"$trial/command.log" 2>&1 & cmd_pid=$!
/usr/bin/python3 "$source_root/scripts/deploy/official_sim_state.py" --port "$state_port" --ros-domain "$ros_domain" --sim-only >"$trial/state.log" 2>&1 & state_pid=$!
args=()
if [[ ${SM_ANCHOR_OBS:-1} == 0 ]]; then args+=(--no-anchor-obs); fi
if [[ "$schedule" == keyboard:* ]]; then
  keyboard_port=${schedule#keyboard:}
  [[ "$keyboard_port" =~ ^[0-9]+$ ]] && (( keyboard_port > 0 && keyboard_port < 65536 )) || { echo 'Invalid keyboard port' >&2; exit 1; }
  args+=(--keyboard-port "$keyboard_port")
else
  args+=(--schedule "$schedule")
fi
if [[ "$fade" != baseline ]]; then args+=(--jog-brake-fade "$fade"); fi
(
  cd "$source_root"
  export SM_ROBOT=x2_ultra PYTHONPATH="$source_root/src" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
  exec /home/2025-Long.xh/HHD/venvs/supermimic/bin/python scripts/deploy/official_sim_policy.py \
    --profile "$profile" --dataset "$dataset" --policy "$model" --out "$trial/policy" \
    --state-port "$state_port" --command-port "$command_port" --ros-domain "$ros_domain" --seconds "$seconds" --stand-anchor-gain "$gain" --stand-gain-quiet-time "$quiet" \
    --compact-log --sim-only "${args[@]}"
) >"$trial/policy.log" 2>&1 & policy_pid=$!
for i in $(seq 1 45); do
  kill -0 "$state_pid" && kill -0 "$cmd_pid" || { cat "$trial/state.log" "$trial/command.log"; exit 1; }
  if grep -q 'policy ready for simulator' "$trial/policy.log"; then break; fi
  kill -0 "$policy_pid" 2>/dev/null || { cat "$trial/policy.log"; exit 1; }
  sleep 1
done
grep -q 'policy ready for simulator' "$trial/policy.log"
bash "$run/start-sim.sh" >"$trial/sim.log" 2>&1 & sim_pid=$!
wait "$policy_pid"
policy_pid=
