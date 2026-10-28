"""Fixed terrain must remain identifiable in real and imagined transitions.

Run on the simulation server; the scene test compiles native robot assets.
"""
from types import SimpleNamespace
from dataclasses import asdict

import torch

from supermimic.envs.box_scene import FixedBox, box_features
from supermimic.envs.mdp.observations import fixed_box_goal
from supermimic.robots.dimensions import STATE_DIM, FEAT_DIM
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg


def states():
    s = torch.zeros(3, STATE_DIM)
    s[:, 2] = .7
    s[:, 3] = 1
    s[1, 0] = .3
    s[2, 3:7] = torch.tensor([2**-.5, 0., 0., 2**-.5])
    return s


def test_box_preserves_position_and_heading_information():
    s = states()
    flat = StateSpec(StateSpecCfg(), 'cpu').canonical(s)
    # Existing planar dynamics cannot distinguish these states.
    torch.testing.assert_close(flat[0], flat[1])
    torch.testing.assert_close(flat[0], flat[2])
    spec = StateSpec(StateSpecCfg(box=FixedBox(.2)), 'cpu')
    features = spec.canonical(s)
    assert features.shape == (3, FEAT_DIM + 9)
    assert not torch.allclose(features[0], features[1])
    assert not torch.allclose(features[0], features[2])
    torch.testing.assert_close(features[:, :FEAT_DIM], flat)
    torch.testing.assert_close(features[0, FEAT_DIM:],
        torch.tensor([.53, 0., -.475, 1., 0., 0., .33, .375, .225]))
    restored = StateSpec(StateSpecCfg(**asdict(spec.cfg)), 'cpu')
    torch.testing.assert_close(restored.canonical(s), features)


def test_real_goal_matches_imagination_and_origin_normalization():
    s = states()
    origins = torch.tensor([[2., 1., 0.], [-2., 3., 0.], [0., 0., 0.]])
    qpos = s.clone()
    qpos[:, :3] += origins
    env = SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(qpos=qpos)),
                          scene=SimpleNamespace(env_origins=origins))
    box = FixedBox(.2, center_y=-.02)
    actual = fixed_box_goal(env, box)
    torch.testing.assert_close(actual, box_features(s[:, :3], s[:, 3:7], box))
    differentiable = s.requires_grad_()
    box_features(differentiable[:, :3], differentiable[:, 3:7], box).square().sum().backward()
    assert torch.isfinite(differentiable.grad).all()
    assert differentiable.grad[:, :2].abs().sum() > 0


def test_box_scene_uses_one_box_and_zero_origin_per_warp_world():
    import numpy as np
    from mjlab.scene import Scene
    from supermimic.envs.tracking_env_cfg import SuperMimicEnvCfgArgs, make_env_cfg

    plain = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=[], num_envs=4))
    cfg = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=[], num_envs=4, box=FixedBox(.2)))
    baseline = Scene(plain.scene, device='cpu').compile()
    scene = Scene(cfg.scene, device='cpu')
    model = scene.compile()
    torch.testing.assert_close(scene.env_origins, torch.zeros(4, 3))
    np.testing.assert_allclose(model.geom('box45_collision').pos, [.53, 0., .225])
    assert model.nq == baseline.nq and model.nv == baseline.nv
    for field in ('body_mass', 'body_inertia', 'jnt_range', 'actuator_forcerange',
                  'actuator_gainprm', 'actuator_biasprm'):
        np.testing.assert_array_equal(getattr(model, field), getattr(baseline, field))
