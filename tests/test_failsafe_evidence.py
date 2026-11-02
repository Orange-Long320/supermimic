import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from supermimic.deploy.failsafe_evidence import REQUIRED_CASES,protected_contract_sha256,require_independent_failsafe


def fixture(folder,plan):
    """Synthetic parser fixture only; never used to launch a ROS transport."""
    path=Path(folder)/'failsafe-parser-fixture.json'
    plan['independent_failsafe']=dict(device_id='unit-fixture-device',firmware_identity='unit-fixture-fw',
        failure_domain='actuator_controller',proof_max_age_s=3600.,evidence_path=str(path),evidence_sha256='')
    data=dict(schema=1,test_only=False,simulation_only=False,mechanism='actuator_firmware_watchdog',
        device_id='unit-fixture-device',firmware_identity='unit-fixture-fw',failure_domain='actuator_controller',
        protected_contract_sha256=protected_contract_sha256(plan),tested_wall=100.,
        safe_stop_behavior_observed=True,fault_latched_after_restart=True,
        case_results=[dict(case=c,passed=True,actuator_inhibit_observed=True,
            measurement_source='actuator_feedback',measured_inhibit_s=.02) for c in sorted(REQUIRED_CASES)])
    write(path,data,plan)
    return path,data


def write(path,data,plan):
    path.write_text(json.dumps(data));plan['independent_failsafe']['evidence_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()


class FailsafeEvidenceTests(unittest.TestCase):
    def plan(self):
        return dict(policy_sha256='a'*64,joint_names=['j'],command_lower=[-1.],command_upper=[1.],
            measured_lower=[-1.],measured_upper=[1.],max_stiffness=[100.],max_damping=[10.],
            fallback_damping=[2.],max_state_age=.05,max_target_age=.05,output_hz=100)

    def test_missing_mechanism_and_model_report_never_clear_hardware(self):
        plan=self.plan()
        with self.assertRaisesRegex(ValueError,'Complete independent'):require_independent_failsafe(plan,101.)
        with tempfile.TemporaryDirectory() as d:
            path,data=fixture(d,plan);data['test_only']=True;data['simulation_only']=True;write(path,data,plan)
            with self.assertRaisesRegex(ValueError,'Actual-device'):require_independent_failsafe(plan,101.)

    def test_coverage_measurement_and_deadline_required(self):
        for kind in ('missing','host_only','slow','unobserved','duplicate'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                plan=self.plan();path,data=fixture(d,plan)
                if kind=='missing':data['case_results'].pop()
                if kind=='host_only':data['case_results'][0]['measurement_source']='pc2_process'
                if kind=='slow':data['case_results'][0]['measured_inhibit_s']=.051
                if kind=='unobserved':data['safe_stop_behavior_observed']=False
                if kind=='duplicate':data['case_results'].append(data['case_results'][0])
                write(path,data,plan)
                with self.assertRaises(ValueError):require_independent_failsafe(plan,101.)

    def test_device_contract_clock_and_hash_drift_fail(self):
        for kind in ('firmware','contract','future','expired','hash','failure_domain'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                plan=self.plan();path,data=fixture(d,plan);require_independent_failsafe(plan,101.)
                if kind=='firmware':plan['independent_failsafe']['firmware_identity']='different'
                if kind=='contract':plan['command_upper']=[.9]
                if kind=='future':data['tested_wall']=102.;write(path,data,plan)
                if kind=='expired':data['tested_wall']=-4000.;write(path,data,plan)
                if kind=='hash':path.write_text('{}')
                if kind=='failure_domain':plan['independent_failsafe']['failure_domain']='pc2'
                with self.assertRaises(ValueError):require_independent_failsafe(plan,101.)


if __name__=='__main__':unittest.main()
