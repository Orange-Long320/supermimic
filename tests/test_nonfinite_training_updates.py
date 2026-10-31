from types import SimpleNamespace

import pytest
import torch

from supermimic.world_model.wm_trainer import WorldModelTrainer, WMTrainerCfg


@pytest.mark.parametrize('bad_gradient_only',[False,True])
def test_bad_update_leaves_parameters_and_optimizer_state_unchanged(bad_gradient_only):
    owner=WorldModelTrainer.__new__(WorldModelTrainer)
    owner.wm=torch.nn.Linear(1,1,bias=False)
    owner.wm.weight.data.zero_()
    owner.opt=torch.optim.Adam(owner.wm.parameters(),lr=.001)
    owner.cfg=WMTrainerCfg(batch_size=1,batches_per_update=1)
    def loss_on(batch):
        weight=owner.wm.weight
        # sqrt(x^2) at x=0 has a finite forward but an undefined gradient.
        loss=weight.square().sqrt().sum() if bad_gradient_only else weight.sum()*float('nan')
        return loss,{}
    owner.loss_on=loss_on
    buffer=SimpleNamespace(sample_windows=lambda *args:None)
    with pytest.raises((FloatingPointError,RuntimeError),match='[Nn]on-finite'):
        owner.update(buffer)
    assert torch.equal(owner.wm.weight,torch.zeros_like(owner.wm.weight))
    assert not owner.opt.state
