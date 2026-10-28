"""Joint tracking and velocity updates through one shared dynamics model.

Written against this repository's interfaces. Velocity actor updates freeze the
decoder weights while retaining the derivative from its output to its input.
"""

from dataclasses import replace

import torch

from supermimic.imagination.objective import gae_targets, policy_objective
from supermimic.imagination.policy_trainer import PolicyTrainer
from supermimic.imagination.rollout import OBS_NOISE_HALF, ImagRollout
from supermimic.locomotion.policy import VelocityCritic
from supermimic.locomotion.task import (
    bounded_reference,
    fall_signals,
    reference_command_table,
    velocity_reward,
)
from supermimic.training.distributed import allreduce_grads, broadcast_params
from supermimic.world_model.heads import (
    analytic_observation,
    anchor_relative_obs,
    flatten_history_term_major,
)
from supermimic.world_model.wm_trainer import reference_at


def relabel_stop_starts(start, fraction):
    """Counterfactual goal learning; keep physical history and replay immutable.

    Replay starts are already randomly ordered. Taking their first fraction
    avoids introducing additional RNG draws into paired training comparisons.
    """
    count = round(len(start.state) * fraction)
    if count == 0:
        return start
    command = start.command.clone()
    reference = start.command_reference.clone()
    command[:count] = 0
    reference[:count] = False
    return replace(start, command=command, command_reference=reference)


class JointPolicyTrainer(PolicyTrainer):
    def __init__(self, wm, policy, critic, lib, cfg, velocity_cfg, critic_cfg, nominal_height, reference_velocity=None):
        if cfg.batch_size < 2:
            raise ValueError("Joint updates require a batch size of at least two")
        super().__init__(wm, policy, critic, lib, replace(cfg, batch_size=max(1, cfg.batch_size // 2)))
        self.velocity_batch = cfg.batch_size - self.cfg.batch_size
        self.velocity_cfg = velocity_cfg
        self.nominal_height = nominal_height
        self.reference_velocity = reference_command_table(lib) if reference_velocity is None else reference_velocity
        device = next(policy.parameters()).device
        self.velocity_critic = VelocityCritic(critic_cfg, velocity_cfg).to(device)
        broadcast_params(self.velocity_critic)
        self.velocity_opt = torch.optim.Adam(policy.encoders["velocity"].parameters(), lr=cfg.lr, betas=cfg.betas)
        self.velocity_critic_opt = torch.optim.Adam(self.velocity_critic.parameters(), lr=cfg.critic_lr)

    def set_lr(self, lr):
        super().set_lr(lr)
        for opt in (self.velocity_opt, self.velocity_critic_opt):
            for group in opt.param_groups:
                group["lr"] = lr

    def checkpoint_state(self):
        return {
            "critic": self.velocity_critic.state_dict(),
            "actor_optimizer": self.velocity_opt.state_dict(),
            "critic_optimizer": self.velocity_critic_opt.state_dict(),
        }

    def load_checkpoint_state(self, state):
        self.velocity_critic.load_state_dict(state["critic"])
        self.velocity_opt.load_state_dict(state["actor_optimizer"])
        self.velocity_critic_opt.load_state_dict(state["critic_optimizer"])

    def update(self, buffer, xi_norm_fn=None):
        motion_metrics = super().update(buffer, env_mask=~buffer.velocity_env_mask)
        velocity_metrics = self._velocity_update(buffer)
        motion_metrics.update({f"velocity/{key}": value for key, value in velocity_metrics.items()})
        return motion_metrics

    def _imagine_velocity(self, start):
        cfg, vc = self.cfg, self.velocity_cfg
        wm, policy, lib = self.wm, self.policy, self.lib
        state, previous = start.state, start.prev_action
        noise = OBS_NOISE_HALF.to(state.device)

        def observe(s, a):
            obs = analytic_observation(s, a, start.xi, lib.default_joint_pos)
            return obs + (2 * torch.rand_like(obs) - 1) * noise if cfg.obs_noise else obs

        history = [observe(start.hist_states[i], start.hist_actions[i]) for i in range(cfg.history)]
        states, actions, rewards = [state], [], []
        survival = [torch.ones(state.shape[0], device=state.device)]
        termination = [torch.zeros_like(survival[0])]
        commands, frames = [], []
        last = lib.last_frame_of(start.frame)
        align = torch.zeros((), device=state.device)

        for t in range(cfg.horizon + 1):
            frame = torch.minimum(start.frame + t, last)
            ref_cmd, valid = bounded_reference(self.reference_velocity, frame, vc)
            reference = start.command_reference & valid & (start.frame + t <= last)
            command = torch.where(reference[:, None], ref_cmd, start.command)
            commands.append(command)
            frames.append(frame)
            if t == cfg.horizon:
                break
            observations = flatten_history_term_major(torch.stack(history))
            latent = policy.encoders["velocity"](observations, command)
            action = policy.decoder(latent, observations)
            # Initial-state alignment supplies a motion prior without letting
            # a mismatched/clipped command supervise the velocity encoder.
            if t == 0 and vc.alignment_weight:
                with torch.no_grad():
                    goal = lib.goal_window(frame, cfg.future_frames)
                    if cfg.anchor_obs:
                        bodies = wm.heads.robot_bodies(state)
                        ref = reference_at(lib, frame)
                        ai = wm.heads.anchor_index
                        relative = anchor_relative_obs(bodies["pos"][:, ai], bodies["quat"][:, ai], ref.body_pos_w[:, ai], ref.body_quat_w[:, ai])
                        goal = torch.cat((goal, relative), -1)
                    teacher = policy.encoders["motion"](goal)
                error = (latent - teacher).square().mean(-1)
                align = (error * reference).sum() / reference.sum().clamp_min(1)
            state = wm.step(state, action, start.xi)
            reward = velocity_reward(state, command, action, previous, wm.heads.soft_limits, self.nominal_height, lib.default_joint_pos)
            _, soft = fall_signals(state, self.nominal_height, vc)
            actions.append(action)
            rewards.append(reward * wm.spec.dt)
            states.append(state)
            termination.append(soft)
            survival.append(survival[-1] * (1 - soft))
            history = history[1:] + [observe(state, action)]
            previous = action
        stacked_states = torch.stack(states)
        rollout = ImagRollout(
            states=stacked_states, actions=torch.stack(actions), rewards=torch.stack(rewards),
            dhat=torch.stack(termination), survival=torch.stack(survival), goals=torch.stack(commands),
            e_root=torch.zeros(*stacked_states.shape[:2], 4, device=state.device), frames=torch.stack(frames), reward_terms={},
        )
        return rollout, align

    def _velocity_update(self, buffer):
        cfg = self.cfg
        start = buffer.sample_starts(self.velocity_batch, cfg.history, env_mask=buffer.velocity_env_mask)
        start = relabel_stop_starts(start, self.velocity_cfg.stop_relabel_fraction)
        xi = self.wm.spec.normalize_xi(start.xi)
        encoder = self.policy.encoders["velocity"]
        modules = (self.wm, self.policy.decoder, self.velocity_critic)
        parameters = [(p, p.requires_grad) for module in modules for p in module.parameters()]
        for p, _ in parameters:
            p.requires_grad_(False)
        try:
            rollout, align = self._imagine_velocity(start)
            H1, n, _ = rollout.states.shape
            features = self.wm.spec.canonical(rollout.states.reshape(H1 * n, -1)).reshape(H1, n, -1)
            tail = self.velocity_critic(features[-1], rollout.goals[-1], xi) if cfg.use_critic else None
            objective = policy_objective(rollout, tail, cfg.gamma)
            loss = -objective / cfg.horizon + self.velocity_cfg.alignment_weight * align
            self.velocity_opt.zero_grad(set_to_none=True)
            loss.backward()
            allreduce_grads(encoder)
            norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), cfg.grad_clip_norm)
            torch.nn.utils.clip_grad_value_(encoder.parameters(), cfg.grad_clip_value)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise FloatingPointError("Non-finite velocity actor update")
            self.velocity_opt.step()
        finally:
            for p, requires_grad in parameters:
                p.requires_grad_(requires_grad)
        metrics = {
            "objective": float(objective.detach()), "alignment": float(align.detach()),
            "grad_norm": float(norm), "survival": float(rollout.survival[-1].mean().detach()),
            "reward_mean": float(rollout.rewards.mean().detach()),
        }
        if cfg.use_critic:
            with torch.no_grad():
                features = features.detach()
                commands = rollout.goals.detach()
                context = xi[None].expand(H1, n, -1)
                values = self.velocity_critic(features.reshape(-1, features.shape[-1]), commands.reshape(-1, 3), context.reshape(-1, context.shape[-1])).reshape(H1, n)
                targets = gae_targets(rollout.rewards.detach(), rollout.dhat[1:].detach(), values, cfg.gamma, cfg.gae_lambda).reshape(-1)
                features = features[:-1].reshape(-1, features.shape[-1])
                commands = commands[:-1].reshape(-1, 3)
                context = context[:-1].reshape(-1, context.shape[-1])
            for _ in range(cfg.critic_epochs):
                permutation = torch.randperm(len(targets), device=targets.device)
                for indices in torch.tensor_split(permutation, min(cfg.critic_minibatches, len(targets))):
                    prediction = self.velocity_critic(features[indices], commands[indices], context[indices])
                    critic_loss = (prediction - targets[indices]).square().mean()
                    self.velocity_critic_opt.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    allreduce_grads(self.velocity_critic)
                    torch.nn.utils.clip_grad_norm_(self.velocity_critic.parameters(), cfg.critic_grad_clip)
                    self.velocity_critic_opt.step()
            metrics["critic_loss"] = float(critic_loss.detach())
        return metrics
