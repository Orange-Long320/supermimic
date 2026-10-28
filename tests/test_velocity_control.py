"""Contracts that can silently break a speed interface while tracking still works."""

from types import SimpleNamespace

import pytest
import torch

from supermimic.kinematics.rotations import quat_from_rotvec, quat_from_yaw
from supermimic.locomotion.config import VelocityCfg
from supermimic.locomotion.policy import VelocityPolicy
from supermimic.locomotion.task import fall_signals, measured_velocity, velocity_reward
from supermimic.policy.nets import PolicyCfg
from supermimic.robots.dimensions import NJ, NQ, OBS_DIM, STATE_DIM, XI_DIM


def standing(n=2):
    state = torch.zeros(n, STATE_DIM)
    state[:, 2] = 0.8
    state[:, 3] = 1
    return state


def test_velocity_coordinates_and_world_yaw_rate():
    state = standing()
    state[0, 3:7] = quat_from_yaw(torch.tensor(torch.pi / 2))
    state[0, NQ + 1] = 1
    state[1, 3:7] = quat_from_rotvec(torch.tensor([0., torch.pi / 2, 0.]))
    state[1, NQ + 3] = 1  # body x points along -world z
    result = measured_velocity(state)
    torch.testing.assert_close(result[0], torch.tensor([1., 0., 0.]), atol=1e-6, rtol=0)
    torch.testing.assert_close(result[1, 2], torch.tensor(-1.), atol=1e-6, rtol=0)


def test_task_scales_fall_height_and_penalizes_zero_command_drift():
    state = standing()
    state[1, NQ] = 0.5
    limits = torch.tensor([-2., 2.]).repeat(NJ, 1)
    action = torch.zeros(2, NJ)
    reward = velocity_reward(state, torch.zeros(2, 3), action, action, limits, 0.8, torch.zeros(NJ))
    assert reward[0] > reward[1]
    state[1, 2] = 0.4
    hard, soft = fall_signals(state, 0.8, VelocityCfg())
    assert hard.tolist() == [False, True]
    assert soft[1] > soft[0]


def test_frozen_decoder_preserves_encoder_gradients():
    cfg = PolicyCfg(encoder_hidden=(16,), decoder_hidden=(16,))
    policy = VelocityPolicy(cfg, VelocityCfg(encoder_hidden=(16,)))
    before = {key: value.clone() for key, value in policy.decoder.state_dict().items()}
    policy.decoder.requires_grad_(False)
    observation = torch.randn(3, 10 * OBS_DIM)
    action = policy.velocity_action(observation, torch.tensor([[0.3, 0., 0.2]]).repeat(3, 1))
    action.square().sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.encoders["velocity"].parameters())
    assert all(p.grad is None for p in policy.decoder.parameters())
    assert all(p.grad is None for p in policy.encoders["motion"].parameters())
    for key, value in before.items():
        torch.testing.assert_close(policy.decoder.state_dict()[key], value)


def test_velocity_replay_keeps_commands_and_masks_tracking_termination():
    from supermimic.data.replay_buffer import ReplayBuffer

    replay = ReplayBuffer(2, 16, STATE_DIM, NJ, XI_DIM, "cpu")
    replay.enable_velocity()
    replay.velocity_env_mask = torch.tensor([False, True])
    for i in range(12):
        replay.add(
            torch.arange(2), standing(), torch.zeros(2, NJ), torch.zeros(2, NJ), torch.zeros(2, XI_DIM),
            torch.zeros(2), torch.zeros(2), torch.full((2,), i), torch.zeros(2, dtype=torch.long),
            torch.zeros(2, dtype=torch.bool), torch.full((2,), i == 0),
            command=torch.tensor([[0., 0., 0.], [0.5, 0., 0.2]]), command_reference=torch.tensor([False, True]),
        )
    starts = replay.sample_starts(20, 3, env_mask=replay.velocity_env_mask)
    torch.testing.assert_close(starts.command, torch.tensor([[0.5, 0., 0.2]]).repeat(20, 1))
    assert starts.command_reference.all()
    windows = replay.sample_windows(100, 3)
    assert (~windows.termination_mask).any() and windows.termination_mask.any()
    assert (windows.termination_mask == windows.termination_mask[:1]).all()


def test_velocity_imagined_update_and_checkpoint_roundtrip():
    from supermimic.data.replay_buffer import StartBatch
    from supermimic.imagination.policy_trainer import PolicyTrainerCfg
    from supermimic.locomotion.trainer import JointPolicyTrainer
    from supermimic.policy.nets import Critic, CriticCfg
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    policy_cfg = PolicyCfg(encoder_hidden=(16,), decoder_hidden=(16,))
    velocity_cfg = VelocityCfg(encoder_hidden=(16,))
    policy = VelocityPolicy(policy_cfg, velocity_cfg)
    heads = SimpleNamespace(soft_limits=torch.tensor([-2., 2.]).repeat(NJ, 1), anchor_index=0)
    wm = WorldModel(WorldModelCfg(hidden=(16,), term_hidden=(8,)), heads, "cpu")
    critic_cfg = CriticCfg(hidden=(16,))
    critic = Critic(critic_cfg, pcfg=policy_cfg)
    n = 2
    state = standing(n)
    xi = torch.zeros(n, XI_DIM)
    xi[:, 0] = 0.75

    class Library:
        default_joint_pos = torch.zeros(NJ)

        def last_frame_of(self, frames):
            return torch.ones_like(frames) * 20

        def ref(self, frames):
            count = frames.numel()
            return SimpleNamespace(root_quat_w=state[:1, 3:7].repeat(count, 1), root_lin_vel_w=torch.zeros(count, 3), root_ang_vel_w=torch.zeros(count, 3))

        def goal_window(self, frames, future):
            return torch.zeros(frames.numel(), policy_cfg.goal_dim)

    start = StartBatch(state, xi, torch.zeros(n, dtype=torch.long), torch.zeros(n, NJ), state[None].repeat(10, 1, 1), torch.zeros(10, n, NJ), torch.zeros(n, 3), torch.ones(n, dtype=torch.bool))
    buffer = SimpleNamespace(velocity_env_mask=torch.tensor([True, True]), sample_starts=lambda *args, **kwargs: start)
    cfg = PolicyTrainerCfg(batch_size=4, horizon=3, critic_epochs=1, critic_minibatches=1, compile=None, obs_noise=False)
    trainer = JointPolicyTrainer(wm, policy, critic, Library(), cfg, velocity_cfg, critic_cfg, 0.8, torch.zeros(21, 3))
    decoder = {k: v.clone() for k, v in policy.decoder.state_dict().items()}
    metrics = trainer._velocity_update(buffer)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert metrics["grad_norm"] > 0
    for key, value in decoder.items():
        torch.testing.assert_close(policy.decoder.state_dict()[key], value)
    trainer.load_checkpoint_state(trainer.checkpoint_state())
    assert all(p.requires_grad for p in wm.parameters())


@pytest.mark.parametrize("config", [VelocityCfg(forward=(1., 2.)), VelocityCfg(fraction=1.), VelocityCfg(resample_seconds=0)])
def test_invalid_velocity_settings_fail_early(config):
    with pytest.raises(ValueError):
        config.validate()


def test_keyboard_release_lease_and_ramp():
    from supermimic.locomotion.keyboard import CommandMailbox

    mailbox = CommandMailbox(timeout=0.5)
    mailbox.put((0.5, 0., 0.6), now=0.)
    assert mailbox.read(0.1, now=0.1) == pytest.approx((0.1, 0., 0.15))
    assert mailbox.read(1., now=0.2) == pytest.approx((0.5, 0., 0.6))
    assert mailbox.read(1., now=0.8) == (0., 0., 0.)
    with pytest.raises(ValueError):
        mailbox.put((float("nan"), 0., 0.))


def test_browser_keyboard_transport():
    from http.client import HTTPConnection

    from supermimic.locomotion.keyboard import CommandMailbox, start_keyboard_server

    mailbox = CommandMailbox()
    server = start_keyboard_server(mailbox, 0, 0.5, 0.4)
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    try:
        connection.request("GET", "/")
        response = connection.getresponse()
        assert response.status == 200
        assert b"keys.has('w')?0.5" in response.read()
        connection.request("POST", "/command", body="[0.5, 0, -0.4]", headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 204
        response.read()
        assert mailbox.target == (0.5, 0., -0.4)
        connection.request("POST", "/command", body='["NaN", 0, 0]')
        response = connection.getresponse()
        assert response.status == 400
        response.read()
        assert mailbox.target == (0.5, 0., -0.4)
    finally:
        connection.close()
        server.shutdown()
        server.server_close()


def test_tracking_termination_head_excludes_velocity_labels():
    from supermimic.world_model.heads import ReferenceFrame
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    def termination_features(position, quaternion, reference):
        return torch.stack((position[:, 2] - reference.root_pos_w[:, 2], position[:, 2] * 0), -1)

    heads = SimpleNamespace(termination_features=termination_features)
    wm = WorldModel(WorldModelCfg(hidden=(16,), term_hidden=(8,)), heads, "cpu")
    state = standing()
    reference = ReferenceFrame(None, None, None, None, state[:, :3], state[:, 3:7])
    states = torch.stack((state, state + 0.01))
    loss, metrics = wm.window_loss(
        states, torch.zeros(1, 2, NJ), torch.zeros(2, XI_DIM), torch.ones(1, 2),
        [reference], termination_mask=torch.zeros(1, 2, dtype=torch.bool),
    )
    loss.backward()
    assert metrics["term_bce"] == 0
    assert all(p.grad is None or not p.grad.any() for p in wm.term_head.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in wm.dynamics.parameters())


def command_fixture():
    """Exercise the real command methods without constructing a simulator."""
    from supermimic.locomotion.env import VelocityMotionCommand

    command = object.__new__(VelocityMotionCommand)
    command.cfg = SimpleNamespace(velocity=VelocityCfg(standing_probability=0., reference_probability=1.))
    command.is_velocity = torch.tensor([False, True, True])
    command.velocity_command = torch.zeros(3, 3)
    command.fallback_command = torch.tensor([[0., 0., 0.], [0.7, 0., -0.2], [0.6, 0., 0.3]])
    command.reference_mode = command.is_velocity.clone()
    command.reference_exhausted = torch.zeros(3, dtype=torch.bool)
    command.command_from_reference = torch.zeros(3, dtype=torch.bool)
    command.command_age = torch.zeros(3, dtype=torch.long)
    command.manual_command = None
    command._env = SimpleNamespace(device="cpu")
    reference = torch.tensor([[0.2, 0., 0.], [0.3, 0., 0.1], [0.4, 0., -0.1]])
    valid = torch.ones(3, dtype=torch.bool)
    command._bounded_reference = lambda: (reference, valid)
    return command, reference, valid


def test_reference_recovers_and_preserves_independent_fallback():
    command, reference, valid = command_fixture()
    fallback = command.fallback_command.clone()
    command._refresh_velocity()
    assert command.command_from_reference.tolist() == [False, True, True]
    torch.testing.assert_close(command.velocity_command[1:], reference[1:])
    valid[1] = False
    command._refresh_velocity()
    torch.testing.assert_close(command.velocity_command[1], fallback[1])
    assert not command.command_from_reference[1]
    reference[1, 0] = 0.5
    valid[1] = True
    command._refresh_velocity()
    assert command.command_from_reference[1]
    torch.testing.assert_close(command.velocity_command[1], reference[1])
    torch.testing.assert_close(command.fallback_command, fallback)


def test_reference_resampling_and_manual_override_are_independent():
    command, reference, valid = command_fixture()
    valid[1] = False
    command._sample_velocity(torch.tensor([1]))
    assert command.reference_mode[1] and not command.command_from_reference[1]
    valid[1] = True
    command._refresh_velocity()
    assert command.command_from_reference[1]
    fallback = command.fallback_command.clone()
    command.set_manual_command((0., 0., 0.))
    assert not command.command_from_reference.any()
    assert not command.velocity_command.any()
    torch.testing.assert_close(command.fallback_command, fallback)
    command.manual_command = None
    command._refresh_velocity()
    torch.testing.assert_close(command.velocity_command[1:], reference[1:])
    command.cfg.velocity.standing_probability = 1.
    untouched = command.velocity_command[2].clone()
    command._sample_velocity(torch.tensor([1]))
    assert not command.reference_mode[1]
    assert not command.velocity_command[1].any()
    assert not command.fallback_command[1].any()
    torch.testing.assert_close(command.velocity_command[2], untouched)


def test_model_validation_uses_executed_actions_and_never_crosses_episodes():
    from supermimic.data.replay_buffer import ReplayBuffer
    from supermimic.locomotion.validation import validate_velocity_model

    buffer = ReplayBuffer(2, 64, STATE_DIM, NJ, XI_DIM, "cpu")
    buffer.enable_velocity()
    buffer.velocity_env_mask = torch.tensor([False, True])
    limits = torch.tensor([-2., 2.]).repeat(NJ, 1)
    clean = torch.zeros(2, NJ)
    executed = clean.clone()
    executed[:, 0] = 0.01
    for episode in range(2):
        for time in range(21):
            state = standing()
            state[:, NQ] = episode * 0.5 + time * 0.01
            next_state = state.clone()
            next_state[:, NQ] += 0.01
            reward = velocity_reward(next_state, torch.zeros(2, 3), clean, clean, limits, 0.8, torch.zeros(NJ)) * 0.02
            buffer.add(torch.arange(2), state, clean, executed, torch.zeros(2, XI_DIM), reward,
                       torch.zeros(2), torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long),
                       torch.full((2,), time == 20), torch.full((2,), time == 0),
                       command=torch.zeros(2, 3), command_reference=torch.zeros(2, dtype=torch.bool))
        buffer.end_episode(torch.arange(2))

    def step(state, action, xi):
        next_state = state.clone()
        next_state[:, NQ] += action[:, 0]
        return next_state

    wm = SimpleNamespace(step=step, heads=SimpleNamespace(soft_limits=limits))
    result = validate_velocity_model(wm, buffer, 0.8, torch.zeros(NJ), 0.02, samples=64, horizons=(1, 4, 8))
    assert result["available_windows"] == 26
    for check in result["checks"]:
        assert max(check["velocity_rmse"]) < 1e-6
        assert check["model_reward_rate"] == pytest.approx(check["native_reward_rate"], abs=1e-6)


@pytest.mark.gpu
@pytest.mark.dataset
def test_velocity_clip_end_does_not_reset_physical_state():
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.robots.paths import lafan_npz_dir

    motion = lafan_npz_dir() / "walk1_subject1.npz"
    if not torch.cuda.is_available() or not motion.exists():
        pytest.skip("Needs server GPU and prepared walking motion")
    env = make_env(SuperMimicEnvCfgArgs(
        npz_files=[str(motion)], num_envs=4, record_transitions=False,
        domain_randomization=False, reset_noise=False, obs_noise=False,
        velocity=VelocityCfg(enabled=True),
    ))
    try:
        cmd = env.command_manager.get_term("motion")
        env.reset()
        cmd.reference_mode[cmd.is_velocity] = True
        cmd.frame[:] = cmd.library.clip_last[0]
        before = cmd.state_io.read()
        cmd._update_command()
        torch.testing.assert_close(cmd.state_io.read(), before, rtol=0, atol=0)
        assert not cmd.clip_end[cmd.is_velocity].any()
        assert cmd.clip_end[~cmd.is_velocity].all()
        assert cmd.reference_exhausted[cmd.is_velocity].all()
        cmd._sample_velocity(cmd.is_velocity.nonzero().squeeze(-1))
        assert not cmd.command_from_reference[cmd.is_velocity].any()
        ids = cmd.is_velocity.nonzero().squeeze(-1)[:1]
        cmd._resample_command(ids)
        assert not cmd.reference_exhausted[ids].any()
        assert cmd.reference_exhausted[cmd.is_velocity.nonzero().squeeze(-1)[1:]].all()
    finally:
        env.close()
