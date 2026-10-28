"""Unapproved reference inputs must be rejected before GPU allocation."""
import importlib.util
import json
from pathlib import Path

import pytest


def entry():
    path = Path(__file__).resolve().parents[1]/'scripts/experiments/box45_train.py'
    spec = importlib.util.spec_from_file_location('box45_train', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unapproved_geometry_stops_before_reading_other_inputs(tmp_path):
    module = entry()
    reference = tmp_path/'reference.npz'
    reference.write_bytes(b'diagnostic reference')
    audit = tmp_path/'geometry.json'
    audit.write_text(json.dumps({'source_sha256': module.digest(reference), 'geometry_gates_passed': False}))
    with pytest.raises(ValueError, match='geometry'):
        module.validate_reference(reference, audit, tmp_path/'missing-force', tmp_path/'missing-climb')


def test_audit_for_another_reference_is_rejected(tmp_path):
    module = entry()
    reference = tmp_path/'reference.npz'
    reference.write_bytes(b'new reference')
    audit = tmp_path/'geometry.json'
    audit.write_text(json.dumps({'source_sha256': 'old digest', 'geometry_gates_passed': True}))
    with pytest.raises(ValueError, match='geometry'):
        module.validate_reference(reference, audit, tmp_path/'missing-force', tmp_path/'missing-climb')


def test_another_timing_pass_does_not_approve_current_timing(tmp_path):
    module = entry()
    reference = tmp_path/'reference.npz'
    reference.write_bytes(b'dense reference')
    audit = tmp_path/'geometry.json'
    audit.write_text(json.dumps({'source_sha256': module.digest(reference), 'geometry_gates_passed': True}))
    force = tmp_path/'force.json'
    force.write_text(json.dumps({'source_sha256': module.digest(reference), 'summary': {'1.0': {'passed': False}, '8.0': {'passed': True}}}))
    with pytest.raises(ValueError, match='own timing'):
        module.validate_reference(reference, audit, force, tmp_path/'missing-climb')
