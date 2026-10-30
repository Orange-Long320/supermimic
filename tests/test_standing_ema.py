"""EMA arithmetic, isolation and comparison provenance without a simulator."""

import copy
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
helper = ROOT / 'scripts/experiments/supermimic_stop_ema.py'
if not helper.exists():
    helper = ROOT / 'scripts/supermimic_stop_ema.py'
spec = importlib.util.spec_from_file_location('standing_ema_test_helper', helper)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
StandingEMA, validate_pair = module.StandingEMA, module.validate_pair


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0., 2.]))
        self.register_buffer('scale', torch.tensor([3.]))
        self.register_buffer('counter', torch.tensor(0))


def test_average_skips_rejected_steps_and_does_not_change_training_or_rng():
    actor = Actor()
    rng = torch.random.get_rng_state().clone()
    ema = StandingEMA(actor, .5)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert not ema.model.training
    assert all(not p.requires_grad for p in ema.model.parameters())
    with torch.no_grad():
        actor.weight.add_(2)
    ema.update(actor, changed=False)
    assert ema.updates == 0
    assert torch.equal(ema.model.weight, torch.tensor([0., 2.]))
    ema.update(actor, changed=True)
    assert torch.equal(ema.model.weight, torch.tensor([2., 4.]))
    with torch.no_grad():
        actor.weight.add_(4)
        actor.scale.fill_(9)
        actor.counter.fill_(7)
    actor.weight.grad = torch.tensor([1., 2.])
    ema.update(actor, changed=True)
    assert ema.updates == 2
    assert torch.equal(ema.model.weight, torch.tensor([4., 6.]))
    assert torch.equal(actor.weight, torch.tensor([6., 8.]))
    assert torch.equal(actor.weight.grad, torch.tensor([1., 2.]))
    assert ema.model.scale.item() == 9
    assert ema.model.counter.item() == 7
    assert torch.equal(rng, torch.random.get_rng_state())


def pair():
    actor = Actor()
    ema = StandingEMA(actor, .5)
    ema.update(actor, changed=True)
    with torch.no_grad():
        actor.weight.add_(2)
    ema.update(actor, changed=True)
    online_state = {'decoder.weight': torch.tensor([17.]),
                    **{'standing_skill.' + k: v.clone() for k, v in actor.state_dict().items()}}
    online = dict(policy=online_state, wm={'dynamics.weight': torch.tensor([5.])},
                  config={'seed': 29}, standing_skill={'mode': 'expert'}, standing_reference={'sha256': 'ref'},
                  supermimic_stop=dict(algorithm='supermimic_bptt', actor_updates=2, iteration=20,
                                      source_sha256='initial', reference_sha256='ref', entry_sha256='entry',
                                      config={'ema_decay': .5}, source_commit='test'),
                  policy_ema=dict(ema.metadata(), variant='online'))
    averaged = copy.deepcopy(online)
    averaged['policy'] = ema.policy_state(online_state)
    averaged['policy_ema'] = dict(ema.metadata(), variant='ema', paired_online_sha256='raw-sha')
    return online, averaged


def test_export_preserves_base_and_serializes_independently(tmp_path):
    online, averaged = pair()
    validate_pair(online, averaged, 'raw-sha')
    assert torch.equal(online['policy']['decoder.weight'], averaged['policy']['decoder.weight'])
    assert not torch.equal(online['policy']['standing_skill.weight'], averaged['policy']['standing_skill.weight'])
    path = tmp_path / 'ema.pt'
    torch.save(averaged, path)
    loaded = torch.load(path, weights_only=False)
    validate_pair(online, loaded, 'raw-sha')
    loaded['policy']['decoder.weight'].zero_()
    assert online['policy']['decoder.weight'].item() == 17


@pytest.mark.parametrize('section,key', [('policy', 'decoder.weight'), ('wm', 'dynamics.weight')])
def test_pair_rejects_changed_shared_weights(section, key):
    online, averaged = pair()
    averaged[section][key].add_(1)
    with pytest.raises(ValueError, match='shared frozen tensor'):
        validate_pair(online, averaged, 'raw-sha')


def test_pair_rejects_wrong_origin_or_count():
    online, averaged = pair()
    with pytest.raises(ValueError, match='different online checkpoint'):
        validate_pair(online, averaged, 'other-sha')
    averaged['policy_ema']['updates'] = 1
    with pytest.raises(ValueError, match='update count'):
        validate_pair(online, averaged, 'raw-sha')


@pytest.mark.parametrize('decay', [-.1, 1., float('nan'), float('inf')])
def test_invalid_decay_is_rejected(decay):
    with pytest.raises(ValueError):
        StandingEMA(Actor(), decay)
