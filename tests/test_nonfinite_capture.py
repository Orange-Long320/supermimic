import importlib.util
import json
from pathlib import Path

import pytest
import torch

spec=importlib.util.spec_from_file_location('nonfinite_trace',Path(__file__).resolve().parents[1]/'scripts/experiments/diagnose_training_nonfinite.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_finite_values_pass_unchanged(tmp_path):
    trace=module.Trace(tmp_path)
    value=torch.tensor([1.,-2.])
    trace.check('normal',{'input':value})
    assert torch.equal(value,torch.tensor([1.,-2.]))
    assert not trace.captured and not list(tmp_path.iterdir())


def test_nonfinite_evidence_is_saved_before_raising(tmp_path):
    trace=module.Trace(tmp_path)
    value=torch.tensor([1.,float('nan'),float('inf')])
    with pytest.raises(module.NonfiniteCapture,match='input'):
        trace.check('input',{'state':value})
    folder=tmp_path/'first-nonfinite'
    report=json.loads((folder/'stage.json').read_text())
    assert report['bad']['state']['nonfinite']==2
    saved=torch.load(folder/'batch.pt',weights_only=False)
    assert torch.isnan(saved['values']['state'][1])
    assert torch.isinf(saved['values']['state'][2])
