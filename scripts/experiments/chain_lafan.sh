#!/usr/bin/env bash
# Attach to an existing run; full evaluation then continue the authorized budget.
set -euo pipefail
cd "$(dirname "$0")/../.."
SM_CURRENT="${1:?existing controller tag required}"
SM_CHAIN="logs/${SM_CURRENT}_chain"
test ! -e "$SM_CHAIN"
mkdir "$SM_CHAIN"
exec > >(tee "$SM_CHAIN/controller.log") 2>&1
trap 'printf "%s\n" "$?" > "$SM_CHAIN/controller.exit"' EXIT
cp "$0" "$SM_CHAIN/launch.sh"
SM_ROUND=0
while true; do
  while [[ ! -f "logs/$SM_CURRENT/controller.exit" ]]; do sleep 1; done
  [[ $(cat "logs/$SM_CURRENT/controller.exit") == 0 ]]
  SM_CKPT="logs/${SM_CURRENT}_train/ckpt_latest.pt"
  SM_EVAL="$SM_CHAIN/eval-$SM_ROUND"
  mkdir "$SM_EVAL"
  cp "logs/$SM_CURRENT/environment.txt" "$SM_EVAL/environment.txt"
  cp "logs/$SM_CURRENT/source-lock.json" "$SM_EVAL/source-lock.json"
  # Start as soon as both training ranks have exited; do not share others' GPUs.
  while [[ -n $(nvidia-smi -i 2,3 --query-compute-apps=pid --format=csv,noheader) ]]; do sleep 1; done
  nvidia-smi > "$SM_EVAL/gpu-before.txt"
  ps -eo user,pid,comm > "$SM_EVAL/processes-before.txt"
  bash scripts/run_local.sh - "$SM_CKPT" "$SM_EVAL" <<'PY'
import json,sys,torch
from pathlib import Path
p,out=Path(sys.argv[1]),Path(sys.argv[2])
c=torch.load(p,map_location='cpu',weights_only=False)
assert all(torch.isfinite(t).all() for k in ('wm','policy','critic') for t in c[k].values())
(out/'iteration.txt').write_text(str(c['iteration']))
names=[r['name'] for r in json.loads(Path('logs/x2_lafan_eval_20261027_0910/results.json').read_text())['clips']]
assert len(names)==len(set(names))==40
for i in range(2): (out/f'clips-{i}.txt').write_text('\n'.join(names[i::2])+'\n')
PY
  sha256sum "$SM_CKPT" > "$SM_EVAL/checkpoint.sha256"
  SM_PIDS=()
  for SM_RANK in 0 1; do
    SM_GPU=$((SM_RANK+2))
    SM_CMD=(env CUDA_VISIBLE_DEVICES="$SM_GPU" bash scripts/run_local.sh scripts/robot.py x2_ultra scripts/evaluation/eval_benchmark.py --ckpt "$SM_CKPT" --clip-list "$SM_EVAL/clips-$SM_RANK.txt" --num-envs 20 --out "$SM_EVAL/results-$SM_RANK.json")
    printf '%q ' "${SM_CMD[@]}" > "$SM_EVAL/command-$SM_RANK.txt"
    printf '\n' >> "$SM_EVAL/command-$SM_RANK.txt"
    (set +e; "${SM_CMD[@]}" > "$SM_EVAL/eval-$SM_RANK.log" 2>&1; SM_CODE=$?; printf '%s\n' "$SM_CODE" > "$SM_EVAL/eval-$SM_RANK.exit"; exit "$SM_CODE") &
    SM_PIDS+=("$!")
    printf '%s\n' "$!" > "$SM_EVAL/eval-$SM_RANK.pid"
  done
  SM_FAILED=0
  for SM_PID in "${SM_PIDS[@]}"; do if ! wait "$SM_PID"; then SM_FAILED=1; fi; done
  [[ "$SM_FAILED" == 0 ]]
  bash scripts/run_local.sh - "$SM_EVAL" <<'PY'
import json,sys
from pathlib import Path
from supermimic.eval.benchmark import ClipResult,save_results
p=Path(sys.argv[1]); rows=[]
for i in range(2): rows.extend(json.loads((p/f'results-{i}.json').read_text())['clips'])
assert len(rows)==len({r['name'] for r in rows})==40
save_results([ClipResult(**r) for r in rows],p/'results.json')
s=json.loads((p/'results.json').read_text())['summary']
b=json.loads(Path('logs/x2_lafan_eval_20261027_0910/results.json').read_text())['summary']
(p/'comparison.json').write_text(json.dumps({'baseline':b,'current':s},indent=2))
# A failed full clip and remaining original step budget warrant continuation.
cont=s['failed']>0 and int((p/'iteration.txt').read_text())<320000
(p/'continue.txt').write_text('yes' if cont else 'no')
print(json.dumps(s))
PY
  [[ $(cat "$SM_EVAL/continue.txt") == yes ]] || break
  SM_ROUND=$((SM_ROUND+1))
  SM_NEXT="${1}_r${SM_ROUND}"
  printf '%s Resuming %s from %s\n' "$(date -Is)" "$SM_NEXT" "$SM_CKPT"
  bash scripts/experiments/direct_dual_lafan.sh "$SM_NEXT" "$SM_CKPT" 2 3
  SM_CURRENT="$SM_NEXT"
done
