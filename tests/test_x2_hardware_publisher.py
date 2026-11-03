"""Offline contract tests; these do not create ROS nodes or hardware publishers."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from supermimic.deploy.x2_command_groups import JOINT_GROUPS
from test_failsafe_evidence import fixture as failsafe_fixture

module_path=Path(__file__).resolve().parents[1]/'scripts/deploy/x2_hardware_publisher.py'
spec=importlib.util.spec_from_file_location('x2_hardware_publisher',module_path)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


class HardwareContractTests(unittest.TestCase):
    def make(self,folder):
        names=[n for group in JOINT_GROUPS.values() for n in group];n=len(names)
        evidence={'policy_sha256':mod.POLICY_SHA,'verified_wall':100.,**{k:True for k in (
            'ready_for_hardware_trial','preparation_and_handover_validated','command_limits_validated',
            'state_pipeline_validated','independent_hardware_failsafe_validated','native_mc_stopped_verified')}}
        ep=folder/'evidence.json';ep.write_text(json.dumps(evidence))
        plan={'schema':1,'policy_sha256':mod.POLICY_SHA,'readiness_path':str(ep),'readiness_sha256':mod.digest(ep),
              'joint_names':names,'command_lower':[-1.]*n,'command_upper':[1.]*n,'measured_lower':[-1.]*n,
              'measured_upper':[1.]*n,'max_target_step':[.1]*n,'max_stiffness':[100.]*n,'max_damping':[10.]*n,
              'fallback_damping':[3.]*n,'max_state_age':.05,'max_target_age':.05,'output_hz':100}
        failsafe_fixture(folder,plan)
        pp=folder/'plan.json';pp.write_text(json.dumps(plan));cp=folder/'consent.json'
        cp.write_text(json.dumps({'user_confirmed_motor_power':True,'plan_sha256':mod.digest(pp),'confirmed_wall':100.}))
        return pp,cp,ep,plan

    def test_consent_and_readiness_required(self):
        with tempfile.TemporaryDirectory() as folder:
            pp,cp,ep,plan=self.make(Path(folder));self.assertEqual(mod.validated_contract(pp,cp,101.),plan)
            data=json.loads(cp.read_text());data['user_confirmed_motor_power']=False;cp.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'confirmation'):mod.validated_contract(pp,cp,101.)
            data['user_confirmed_motor_power']=True;cp.write_text(json.dumps(data))
            evidence=json.loads(ep.read_text());evidence['independent_hardware_failsafe_validated']=False
            ep.write_text(json.dumps(evidence));plan['readiness_sha256']=mod.digest(ep);pp.write_text(json.dumps(plan))
            data['plan_sha256']=mod.digest(pp);cp.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'readiness'):mod.validated_contract(pp,cp,101.)

    def test_wrong_or_large_targets_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            _,_,_,plan=self.make(Path(folder));names=plan['joint_names'];n=len(names)
            target={'names':names,'position':[0.]*n,'measured_position':[0.]*n,'velocity':[0.]*n,'effort':[0.]*n,
                    'stiffness':[20.]*n,'damping':[2.]*n}
            self.assertEqual(mod.validate_target(target,plan,None),[0.]*n)
            target['position'][0]=.2
            with self.assertRaisesRegex(ValueError,'step'):mod.validate_target(target,plan,None)
            target['position'][0]=0.;target['effort'][0]=1.
            with self.assertRaisesRegex(ValueError,'feedforward'):mod.validate_target(target,plan,None)

    def test_synthetic_evidence_cannot_enable_hardware(self):
        with tempfile.TemporaryDirectory() as folder:
            pp,cp,ep,plan=self.make(Path(folder));evidence=json.loads(ep.read_text());evidence['test_only']=True
            ep.write_text(json.dumps(evidence));plan['readiness_sha256']=mod.digest(ep);pp.write_text(json.dumps(plan))
            consent=json.loads(cp.read_text());consent['plan_sha256']=mod.digest(pp);cp.write_text(json.dumps(consent))
            with self.assertRaisesRegex(ValueError,'scope'):mod.validated_contract(pp,cp,101.)
            self.assertEqual(mod.validated_contract(pp,cp,101.,isolated=True),plan)

    def test_only_waist_pitch_can_bypass_command_range(self):
        with tempfile.TemporaryDirectory() as folder:
            _,_,_,plan=self.make(Path(folder));names=plan['joint_names'];n=len(names)
            plan['max_target_step']=[10.]*n
            target={'names':names,'position':[0.]*n,'measured_position':[0.]*n,'velocity':[0.]*n,'effort':[0.]*n,
                    'stiffness':[20.]*n,'damping':[2.]*n}
            waist=names.index('waist_pitch_joint'); other=names.index('waist_roll_joint')
            target['position'][waist]=-2.
            with self.assertRaisesRegex(ValueError,'Command position'):mod.validate_target(target,plan,None)
            plan['unclipped_waist_pitch_target']=True
            self.assertEqual(mod.validate_target(target,plan,None)[waist],-2.)
            target['position'][other]=-2.
            with self.assertRaisesRegex(ValueError,'Command position'):mod.validate_target(target,plan,None)
            target['position'][other]=0.;target['measured_position'][waist]=-2.
            with self.assertRaisesRegex(ValueError,'Measured position'):mod.validate_target(target,plan,None)

    def test_waist_override_forbidden_in_validated_mode(self):
        with tempfile.TemporaryDirectory() as folder:
            pp,cp,ep,plan=self.make(Path(folder));plan['unclipped_waist_pitch_target']=True
            pp.write_text(json.dumps(plan));consent=json.loads(cp.read_text());consent['plan_sha256']=mod.digest(pp);cp.write_text(json.dumps(consent))
            with self.assertRaisesRegex(ValueError,'Joint target override'):mod.validated_contract(pp,cp,101.)

    def test_all_joint_override_skips_targets_but_retains_measured_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            _,_,_,plan=self.make(Path(folder));names=plan['joint_names'];n=len(names)
            plan['max_target_step']=[10.]*n
            target={'names':names,'position':[-2.]*n,'measured_position':[0.]*n,'velocity':[0.]*n,'effort':[0.]*n,
                    'stiffness':[20.]*n,'damping':[2.]*n}
            with self.assertRaisesRegex(ValueError,'Command position'):mod.validate_target(target,plan,None)
            plan['unclipped_all_joint_targets']=True
            self.assertEqual(mod.validate_target(target,plan,None),[-2.]*n)
            target['measured_position'][0]=2.
            with self.assertRaisesRegex(ValueError,'Measured position'):mod.validate_target(target,plan,None)

    def test_all_joint_override_forbidden_in_validated_mode(self):
        with tempfile.TemporaryDirectory() as folder:
            pp,cp,ep,plan=self.make(Path(folder));plan['unclipped_all_joint_targets']=True
            pp.write_text(json.dumps(plan));consent=json.loads(cp.read_text());consent['plan_sha256']=mod.digest(pp);cp.write_text(json.dumps(consent))
            with self.assertRaisesRegex(ValueError,'Joint target override'):mod.validated_contract(pp,cp,101.)

    def test_manual_trial_is_explicit_bounded_and_never_claims_auto_failsafe(self):
        with tempfile.TemporaryDirectory() as folder:
            pp,cp,ep,plan=self.make(Path(folder))
            plan.update(safety_mode='manual_supervised_trial',manual_supervision=dict(max_trial_seconds=10.,
                support_condition='feet_loaded_fall_arrest',automatic_host_loss_protection=False))
            evidence=json.loads(ep.read_text());evidence.update(safety_mode='manual_supervised_trial',
                manual_supervised_trial_validated=True,independent_hardware_failsafe_validated=False)
            ep.write_text(json.dumps(evidence));plan['readiness_sha256']=mod.digest(ep);pp.write_text(json.dumps(plan))
            consent=json.loads(cp.read_text());consent['plan_sha256']=mod.digest(pp);cp.write_text(json.dumps(consent))
            with self.assertRaisesRegex(ValueError,'manual trial confirmation'):
                mod.validated_contract(pp,cp,101.)
            for key in ('user_confirmed_manual_supervision','operator_at_estop',
                        'manual_estop_function_checked','acknowledges_no_automatic_host_loss_protection'):
                consent[key]=True
            cp.write_text(json.dumps(consent));self.assertEqual(mod.validated_contract(pp,cp,101.),plan)
            plan['manual_supervision']['max_trial_seconds']=31.;pp.write_text(json.dumps(plan))
            consent['plan_sha256']=mod.digest(pp);cp.write_text(json.dumps(consent))
            with self.assertRaisesRegex(ValueError,'bounded manual'):mod.validated_contract(pp,cp,101.)
    def test_explicitly_unvalidated_trial_cannot_be_relabelled_as_validated(self):
        with tempfile.TemporaryDirectory() as folder:
            pp,cp,ep,plan=self.make(Path(folder))
            items=['preparation and handover dynamics','command range contract','state pipeline']
            plan.update(safety_mode='manual_supervised_unvalidated_trial',
                manual_supervision=dict(max_trial_seconds=10.,support_condition='feet_loaded_fall_arrest',
                                        automatic_host_loss_protection=False))
            evidence={'policy_sha256':mod.POLICY_SHA,'verified_wall':100.,
                      'explicitly_unvalidated_trial':True,'safety_mode':'manual_supervised_unvalidated_trial',
                      'unvalidated_items':items,
                      **{k:False for k in ('ready_for_hardware_trial','preparation_and_handover_validated',
                          'command_limits_validated','state_pipeline_validated',
                          'independent_hardware_failsafe_validated','manual_supervised_trial_validated')},
                      'native_mc_stopped_verified':True}
            ep.write_text(json.dumps(evidence));plan['readiness_sha256']=mod.digest(ep);pp.write_text(json.dumps(plan))
            consent={'user_confirmed_motor_power':True,'plan_sha256':mod.digest(pp),'confirmed_wall':100.,
                     'user_confirmed_manual_supervision':True,'operator_at_estop':True,
                     'manual_estop_function_checked':True,'acknowledges_no_automatic_host_loss_protection':True,
                     'acknowledges_unvalidated_software':True}
            cp.write_text(json.dumps(consent))
            self.assertEqual(mod.validated_contract(pp,cp,101.),plan)
            consent['acknowledges_unvalidated_software']=False;cp.write_text(json.dumps(consent))
            with self.assertRaisesRegex(ValueError,'acknowledges_unvalidated_software'):
                mod.validated_contract(pp,cp,101.)
            consent['acknowledges_unvalidated_software']=True;cp.write_text(json.dumps(consent))
            evidence['command_limits_validated']=True
            ep.write_text(json.dumps(evidence));plan['readiness_sha256']=mod.digest(ep);pp.write_text(json.dumps(plan))
            consent['plan_sha256']=mod.digest(pp);cp.write_text(json.dumps(consent))
            with self.assertRaisesRegex(ValueError,'every validation as failed'):
                mod.validated_contract(pp,cp,101.)
            evidence['command_limits_validated']=False;evidence['unvalidated_items']=[]
            ep.write_text(json.dumps(evidence));plan['readiness_sha256']=mod.digest(ep);pp.write_text(json.dumps(plan))
            consent['plan_sha256']=mod.digest(pp);cp.write_text(json.dumps(consent))
            with self.assertRaisesRegex(ValueError,'list the unvalidated items'):
                mod.validated_contract(pp,cp,101.)


if __name__=='__main__':unittest.main()
