#!/usr/bin/env bash
# User-requested local native simulation only; no ROS or hardware publisher.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
runtime="$root/.local/native-keyboard-20261104"
python="$runtime/venv/bin/python"
[[ -x "$python" && -f "$runtime/assets/manifest.json" ]] || {
  echo 'Local simulation environment/assets are missing; see docs/X2_LOCAL_KEYBOARD_20261104.md' >&2
  exit 1
}
export PYTHONPATH="$runtime/runtime-src:$runtime/vendor-mjlab-math"
export SM_ROBOT=x2_ultra SM_ROBOT_SPEC="$runtime/assets/fk-only/profile.json"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
out="$root/outputs/x2_local_keyboard_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$root/outputs"
exec "$python" "$root/scripts/evaluation/play_native_keyboard.py" \
  --assets "$runtime/assets" --out "$out" "$@"
