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


def test_light_capture_uses_native_gradient_rejection_and_preserves_finite_path(tmp_path,monkeypatch):
    from supermimic.data.replay_buffer import ReplayBuffer
    from supermimic.world_model.wm_trainer import WorldModelTrainer
    from supermimic.imagination import policy_trainer
    # Register originals before install() patches them inside this test process.
    for owner,name in [(ReplayBuffer,'add'),(ReplayBuffer,'sample_starts'),
                       (WorldModelTrainer,'loss_on'),(policy_trainer,'policy_objective'),
                       (policy_trainer.PolicyTrainer,'update'),
                       (torch.nn.utils,'clip_grad_norm_'),(torch.nn.functional,'mse_loss')]:
        monkeypatch.setattr(owner,name,getattr(owner,name))
    trace=module.Trace(tmp_path,light=True)
    trace.install()
    p=torch.nn.Parameter(torch.zeros(1));p.grad=torch.tensor([2.])
    torch.nn.utils.clip_grad_norm_([p],1.,error_if_nonfinite=True)
    torch.testing.assert_close(p.grad,torch.ones(1))
    assert not list(tmp_path.iterdir())
    p.grad=torch.tensor([float('nan')])
    with pytest.raises(module.NonfiniteCapture,match='raw_gradient'):
        torch.nn.utils.clip_grad_norm_([p],1.,error_if_nonfinite=True)
    report=json.loads((tmp_path/'first-nonfinite/stage.json').read_text())
    assert report['bad']['total_norm']['nonfinite']==1
