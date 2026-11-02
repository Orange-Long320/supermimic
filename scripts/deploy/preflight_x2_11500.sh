#!/usr/bin/env bash
# Run inside lxh only. Fixed-input checks and read-only robot subscriptions.
set -eo pipefail
release=${1:?candidate release directory}
out=${2:?new report directory}
base=/home/agi/lxh-supermimic
mkdir "$out"
source /opt/ros/humble/setup.bash
source /home/agi/aimdk/install/setup.bash
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export SM_ROBOT=x2_ultra SM_ROBOT_SPEC="$release/fk-only/profile.json"
export PYTHONPATH="$release/source/src:$base/python-cpu:$base/vendor-mjlab-math:${PYTHONPATH:-}"
scripts=$release/source/scripts/deploy
python3 "$scripts/inspect_x2_candidate.py" --release "$release" --out "$out/inventory.json" > "$out/inventory.log" 2>&1
python3 -m unittest discover -s "$release/source/tests" -v > "$out/controller-tests.log" 2>&1
python3 "$release/tests/test_deploy_core.py" -v > "$out/core-tests.log" 2>&1
python3 "$release/tests/test_pelvis_state.py" -v > "$out/pelvis-tests.log" 2>&1
python3 "$scripts/verify_policy_fixture.py" --policy "$release/policy.pt" --fixture "$release/policy-fixture.json" --samples 20 --out "$out/policy-fixture.json" > "$out/policy-fixture.log" 2>&1
python3 "$scripts/verify_goal_fixture.py" --dataset "$release/dataset" --profile "$release/fk-only/profile.json" --expected "$release/lxh-goal-fixture.json" --out "$out/goal-fixture.json" > "$out/goal-fixture.log" 2>&1
python3 "$scripts/pc2_readiness.py" --policy "$release/policy.pt" --startup-candidate "$release/startup-candidate.json" --release-manifest "$release/release-manifest.json" --seconds 6 --out "$out/readiness.json" > "$out/readiness.log" 2>&1
python3 - "$release" "$out" <<'PY'
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
from supermimic.deploy.numpy_policy import NumpyPolicy
r,out=map(Path,sys.argv[1:]);fixture=json.loads((r/'policy-fixture.json').read_text())
expected='d59809cb544e70c775bf0180174f5f57b3e5572a6639fe0821697639cf798063'
p=NumpyPolicy(r/'numpy-policy-11500.npz',expected_sha256=expected)
actual=p.infer(fixture['obs'],fixture['goal'])
error=float(np.max(np.abs(actual-fixture['expected'])))
if not np.isfinite(actual).all() or error>1e-4:raise RuntimeError('NumPy policy fixture mismatch')
inventory=json.loads((out/'inventory.json').read_text());readiness=json.loads((out/'readiness.json').read_text())
manifest=json.loads((r/'release-manifest.json').read_text())
report=dict(selected_checkpoint=11500,verified_wall=time.time(),policy_sha256=inventory['policy_sha256'],
 software_checks_passed=True,numpy_fixture_max_abs_error=error,
 timing_issue_deferred_by_user=True,ready_for_hardware_trial=False,
 preparation_and_handover_validated=False,command_limits_validated=False,
 state_pipeline_validated=False,independent_hardware_failsafe_validated=False,
 selected_safety_mode=manifest.get('user_selected_safety_mode','independent_hardware'),
 manual_supervised_trial_validated=False,
 native_mc_stopped_verified=inventory['native_mc'].get('status')=='Idel',
 motor_output_enabled=False,control_transferred=False,persistent_policy_process_started=False,
 motor_command_publishers=inventory['motor_command_publishers'],
 deferred_requirements=readiness.get('deferred_requirements',[]),
 blocking_conditions=list(readiness['blocking_conditions']))
if any(inventory['motor_command_publishers'].values()):
 report['blocking_conditions'].append('Existing motor command publishers require ownership coordination')
(out/'deployment-status.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
PY
