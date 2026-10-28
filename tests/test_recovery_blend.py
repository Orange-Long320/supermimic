import torch
from supermimic.policy.recovery_blend import blend_joint_recovery


def test_disabled_early_stage_preserves_original_commands_exactly():
    generator=torch.Generator().manual_seed(7)
    main=torch.randn(3,31,generator=generator);recovery=torch.randn(3,31,generator=generator)
    residual=torch.randn(3,31,generator=generator);fraction=torch.tensor([0.,.5,1.])
    expected=torch.lerp(main,recovery+residual,fraction[:,None])
    actual=blend_joint_recovery(main,recovery,residual,fraction,torch.zeros(3),[10,11])
    assert torch.equal(actual,expected)


def test_early_stage_uses_only_selected_feedback_and_restores_late_control():
    main=torch.full((1,31),2.);recovery=torch.full((1,31),3.);residual=torch.full((1,31),.5)
    early=blend_joint_recovery(main,recovery,residual,torch.zeros(1),torch.ones(1),[10,11])
    assert torch.equal(early[:,[10,11]],recovery[:,[10,11]])
    assert torch.equal(early[:,:10],main[:,:10])
    assert torch.equal(early[:,12:],main[:,12:])
    late=blend_joint_recovery(main,recovery,residual,torch.ones(1),torch.ones(1),[10,11])
    assert torch.equal(late,recovery+residual)
