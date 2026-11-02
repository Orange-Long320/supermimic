"""Validate explicit actuator-side safety evidence; no device control.

This checks evidence consistency and coverage, not the truth of a measurement.
Physical results must come from the actual device/independent instrument. A
PC2 process supervisor or simulated watchdog is insufficient for hardware.
"""
import hashlib
import json
import math
from pathlib import Path


REQUIRED_CASES = frozenset(('worker_exit','publisher_freeze','command_host_loss',
    'transport_loss','frozen_state_replay','guardian_failure','receiver_restart_latched'))


def protected_contract_sha256(plan):
    fields=('policy_sha256','joint_names','command_lower','command_upper','measured_lower',
            'measured_upper','max_stiffness','max_damping','fallback_damping',
            'max_state_age','max_target_age','output_hz')
    protected={k:plan[k] for k in fields}
    cfg=plan['independent_failsafe']
    protected['receiver']={k:cfg[k] for k in ('device_id','firmware_identity','failure_domain')}
    return hashlib.sha256(json.dumps(protected,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def require_independent_failsafe(plan, now):
    try:
        cfg=plan['independent_failsafe']
        if cfg['failure_domain'] not in ('actuator_controller','independent_safety_device'):
            raise ValueError('Failsafe must be outside the command-host failure domain')
        for key in ('device_id','firmware_identity'):
            if not isinstance(cfg[key],str) or not cfg[key].strip():raise ValueError('Missing failsafe identity')
        max_age=cfg['proof_max_age_s']
        if type(max_age) not in (int,float) or not math.isfinite(max_age) or not 0<max_age<=86400:
            raise ValueError('Invalid explicit failsafe proof lifetime')
        path=Path(cfg['evidence_path']);raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=cfg['evidence_sha256']:
            raise ValueError('Independent failsafe evidence hash mismatch')
        evidence=json.loads(raw)
        if (evidence.get('schema')!=1 or evidence.get('test_only') is not False
                or evidence.get('simulation_only') is not False):
            raise ValueError('Actual-device failsafe evidence required; test/model evidence is prohibited')
        if evidence.get('mechanism') not in ('actuator_firmware_watchdog','independent_safety_inhibit'):
            raise ValueError('A software process supervisor is not an independent hardware failsafe')
        if any(evidence.get(k)!=cfg[k] for k in ('device_id','firmware_identity','failure_domain')):
            raise ValueError('Independent failsafe device/firmware mismatch')
        if evidence.get('protected_contract_sha256')!=protected_contract_sha256(plan):
            raise ValueError('Failsafe evidence does not protect this command contract')
        tested=evidence.get('tested_wall')
        if type(tested) not in (int,float) or not math.isfinite(tested) or not 0<=now-tested<=max_age:
            raise ValueError('Independent failsafe evidence expired or future')
        if (evidence.get('safe_stop_behavior_observed') is not True
                or evidence.get('fault_latched_after_restart') is not True):
            raise ValueError('Physical stopping and persistent latch have not been verified')
        rows=evidence.get('case_results',[])
        if not isinstance(rows,list):raise ValueError('Invalid failsafe case results')
        cases={row['case']:row for row in rows}
        if len(cases)!=len(rows) or not REQUIRED_CASES<=cases.keys():
            raise ValueError('Incomplete independent failsafe fault coverage')
        bound=min(plan['max_state_age'],plan['max_target_age'])
        for case in REQUIRED_CASES:
            row=cases[case];latency=row.get('measured_inhibit_s')
            if (row.get('passed') is not True or row.get('actuator_inhibit_observed') is not True
                    or row.get('measurement_source') not in ('actuator_feedback','independent_instrument')
                    or type(latency) not in (int,float) or not math.isfinite(latency)
                    or not 0<=latency<=bound):
                raise ValueError('Independent failsafe case failed or exceeded deadline: '+case)
        return evidence
    except (KeyError,TypeError,OSError,json.JSONDecodeError) as exc:
        raise ValueError('Complete independent hardware failsafe evidence required') from exc
