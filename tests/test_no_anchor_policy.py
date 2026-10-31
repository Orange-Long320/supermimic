import torch
import pytest

from supermimic.policy.nets import Policy, PolicyCfg, Critic, CriticCfg
from supermimic.policy.no_anchor import remove_anchor_weights
from supermimic.robots.dimensions import GOAL_FEAT_DIM, OBS_DIM


def test_transfer_preserves_zero_anchor_actions_and_values():
    torch.manual_seed(5)
    future = 2*GOAL_FEAT_DIM
    common = dict(encoder_hidden=(16,), decoder_hidden=(16,), history=1)
    old_cfg = PolicyCfg(**common, goal_dim=future+9)
    new_cfg = PolicyCfg(**common, goal_dim=future)
    old, new = Policy(old_cfg), Policy(new_cfg)
    new.load_state_dict(remove_anchor_weights(old.state_dict(), future_dim=future), strict=True)
    old_v = Critic(CriticCfg(hidden=(16,)), feat_dim=7, xi_dim=3, pcfg=old_cfg)
    new_v = Critic(CriticCfg(hidden=(16,)), feat_dim=7, xi_dim=3, pcfg=new_cfg)
    new_v.load_state_dict(remove_anchor_weights(old_v.state_dict(), future_dim=future, feature_dim=7), strict=True)
    goal, obs = torch.randn(4, future), torch.randn(4, OBS_DIM)
    padded = torch.cat((goal, torch.zeros(4, 9)), -1)
    feat, xi, root = torch.randn(4, 7), torch.randn(4, 3), torch.randn(4, 4)
    torch.testing.assert_close(new(obs, goal), old(obs, padded))
    torch.testing.assert_close(new_v(feat, goal, xi, root), old_v(feat, padded, xi, root))
    # The shared decoder, biases and all later layers are preserved exactly.
    for key, value in old.state_dict().items():
        if key not in ('encoders.motion.scale', 'encoders.motion.mlp.net.0.weight'):
            assert torch.equal(value, new.state_dict()[key])
    with pytest.raises(ValueError):
        remove_anchor_weights(new.state_dict(), future_dim=future)


def test_no_anchor_goal_never_reads_root_pose_or_integrates_position():
    import numpy as np
    from supermimic.deploy.skill_goal import SkillGoal
    class Scheduler:
        def select(self, mode): self.mode=mode
        def window(self): return np.arange(720, dtype=np.float32).reshape(10, 72)
        def advance(self): self.advanced=True
    obj=SkillGoal.__new__(SkillGoal)
    obj.anchor_obs=False; obj.scheduler=Scheduler()
    obj.has_motion_request=False; obj.idle_velocity_feedback=False
    obj.planner='baseline'; obj.foot_idle=None
    # Deliberately provide no FK, pose integration, or world anchor state.
    for mode in ('stand','walk','backward','turn_left_inplace','turn_right_inplace','jog'):
        a=obj.goal(None,None,None,mode)
        b=obj.goal([100.,-500.,2.],[0.,0.,0.,1.],None,mode)
        assert np.array_equal(a,b) and a.shape==(720,)
        obj.advance()
        assert obj.scheduler.advanced
