#!/usr/bin/env bash
# Explicit user-selected candidate for an isolated official simulation only.
set -euo pipefail
run=${1:?isolated simulator run directory}
name=${2:?new trial name}
seconds=${3:?duration}
input=${4:?schedule JSON path or keyboard:port}
root=$(cd "$(dirname "$0")/../.." && pwd)
[[ "$root" -ef "$run/source" ]] || { echo 'Run source must match this launcher' >&2; exit 1; }
config="$root/configs/deploy/x2_ultra_selected_sim.json"
mapfile -t selection < <(/usr/bin/python3 - "$config" <<'PY'
import hashlib, json, sys
from pathlib import Path
cfg=json.loads(Path(sys.argv[1]).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()
policy=Path(cfg['policy']); dataset=Path(cfg['dataset'])
if sha(policy)!=cfg['policy_sha256'] or sha(dataset/'manifest.json')!=cfg['dataset_manifest_sha256']:
    raise ValueError('selected policy or reference manifest SHA mismatch')
manifest=json.loads((dataset/'manifest.json').read_text())
for clip in manifest['clips']:
    if sha(dataset/(clip['name']+'.npz'))!=clip['sha256']:
        raise ValueError('selected reference SHA mismatch: '+clip['name'])
print(dataset); print(policy)
PY
)
[[ ${#selection[@]} == 2 ]] || { echo 'Selection verification failed' >&2; exit 1; }
exec bash "$root/scripts/deploy/run_official_trial.sh" "$run" "$name" "$seconds" baseline 1.0 "$input" 0 "${selection[0]}" "${selection[1]}"
