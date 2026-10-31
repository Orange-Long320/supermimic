import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec=importlib.util.spec_from_file_location('skill_training',Path(__file__).resolve().parents[1]/'scripts/experiments/train_skill_phase.py')
training=importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


def options(**overrides):
    args=dict(phase='train',compiled=True,share_gpu=True,torch_memory_gib=5.5,
              min_free_mib=10752,stop_free_mib=3072)
    args.update(overrides)
    return SimpleNamespace(**args)


def test_timed_resume_can_share_with_explicit_memory_budget():
    training.validate_memory_options(options())
    training.validate_memory_options(options(phase='train-steps'))
    training.validate_memory_options(options(share_gpu=False,torch_memory_gib=None))
    training.validate_memory_options(options(stop_free_mib=0,min_free_mib=8192))


@pytest.mark.parametrize('change',[
    {'min_free_mib':10751}, {'stop_free_mib':-1}, {'compiled':False},
    {'torch_memory_gib':None}, {'torch_memory_gib':8.5}, {'phase':'capacity'},
])
def test_shared_training_rejects_missing_or_inadequate_budget(change):
    with pytest.raises(ValueError):
        training.validate_memory_options(options(**change))
