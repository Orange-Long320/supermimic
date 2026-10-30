"""Calibrate a velocity feedback projection through a frozen world model.

Real trajectory collection, world-model validation, checkpoint persistence and
physical acceptance belong to the caller. This module uses deterministic actions.
"""

import copy
import warnings

import torch
from torch import nn

from supermimic.imagination.objective import gae_targets, policy_objective
from supermimic.imagination.policy_trainer import PolicyTrainerCfg
from supermimic.imagination.rollout import ImagRollout
from supermimic.locomotion.task import fall_signals, measured_velocity, velocity_reward
from supermimic.policy.standing import add_velocity_feedback, task_latent
from supermimic.robots.dimensions import STATE_DIM
from supermimic.world_model.heads import analytic_observation, flatten_history_term_major


class VelocityFeedbackImaginationTrainer:
    """Optimize only a new six-input velocity projection; retain a fixed teacher."""

    def __init__(self, policy, wm, default_joint_pos, soft_joint_limits, nominal_height,
                 velocity_cfg, history, dt, lr=3e-5, horizon=32):
        if history <= 0 or horizon <= 0 or dt <= 0 or lr <= 0:
            raise ValueError('history, horizon, dt and lr must be positive')
        if abs(float(wm.spec.dt) - float(dt)) > 1e-8:
            raise ValueError('World-model and collection time steps differ')
        if not hasattr(policy, 'standing_skill') or policy.standing_skill.mode != 'expert' or policy.standing_skill.feedback_dim != 6:
            raise ValueError('A six-input standing expert is required')
        if hasattr(policy, 'velocity_feedback_weight'):
            raise ValueError('Expected no existing velocity feedback projection')
        self.policy, self.wm = policy, wm
        policy.requires_grad_(False)
        wm.requires_grad_(False)
        add_velocity_feedback(policy)
        self.projection = policy.velocity_feedback_weight
        self.projection.requires_grad_(True)
        self.device, self.dtype = self.projection.device, self.projection.dtype
        self.default_joint_pos = torch.as_tensor(default_joint_pos, device=self.device, dtype=self.dtype).detach().clone()
        self.soft_joint_limits = torch.as_tensor(soft_joint_limits, device=self.device, dtype=self.dtype).detach().clone()
        if self.default_joint_pos.ndim != 1 or self.soft_joint_limits.shape != (len(self.default_joint_pos), 2):
            raise ValueError('Invalid joint positions or soft joint limits')
        self.nominal_height = float(nominal_height)
        self.velocity_cfg, self.history, self.dt, self.horizon, self.lr = velocity_cfg, history, float(dt), horizon, float(lr)
        self.cfg = PolicyTrainerCfg(horizon=horizon, history=history, lr=lr)
        prototype = torch.zeros(1, STATE_DIM, device=self.device, dtype=self.dtype)
        prototype[:, 3] = 1.
        critic_dim = wm.spec.canonical(prototype).shape[-1] + 3 + wm.spec.xi_offset.numel()
        self.value = nn.Sequential(nn.Linear(critic_dim, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1)).to(device=self.device, dtype=self.dtype)
        nn.init.zeros_(self.value[-1].weight)
        nn.init.zeros_(self.value[-1].bias)
        self.projection_opt = torch.optim.Adam([self.projection], lr=lr, betas=self.cfg.betas)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=self.cfg.critic_lr)

    def _feedback(self, state):
        return torch.cat((measured_velocity(state), state.new_zeros((*state.shape[:-1], 3))), -1)

    def _critic_inputs(self, state, command, xi):
        return torch.cat((self.wm.spec.canonical(state), command, xi), -1)

    def _teacher(self, obs, command):
        with torch.no_grad():
            latent = self.policy.encoders['velocity'](obs, command)
            action = self.policy.decoder(latent, obs)
        return latent, action

    def _new_latent(self, obs, command, feedback):
        error = torch.cat((feedback[..., :3] - command, feedback[..., 3:]), -1)
        return task_latent(self.policy.encoders['velocity'], obs, command, error,
                           self.projection, self.policy.standing_skill.feedback_scale)

    def _validate(self, tensors):
        for tensor in tensors:
            if tensor.device != self.device or not torch.isfinite(tensor).all():
                raise ValueError('Inputs must be finite tensors on the actor device')

    def update(self, batch, retention_batch):
        """Update from real histories and a fixed real-state retention set.

        Histories are oldest first; their last action is the previous action for
        the final state. Each command is the actual command at that state.
        """
        if any(k not in batch for k in ('hist_states', 'hist_actions', 'xi', 'command')):
            raise ValueError('batch requires hist_states, hist_actions, xi and command')
        if any(k not in retention_batch for k in ('obs', 'command', 'feedback')):
            raise ValueError('retention_batch requires obs, command and feedback')
        states, history_actions, xi, command = (batch[k].detach() for k in ('hist_states', 'hist_actions', 'xi', 'command'))
        retained_obs, retained_command, retained_feedback = (retention_batch[k].detach() for k in ('obs', 'command', 'feedback'))
        if states.ndim != 3 or states.shape[0] != self.history or history_actions.ndim != 3 or history_actions.shape[:2] != states.shape[:2]:
            raise ValueError('Histories must have matching (history, batch, feature) dimensions')
        n = states.shape[1]
        if not n or xi.shape != (n, self.wm.spec.xi_offset.numel()) or command.shape != (n, 3):
            raise ValueError('Invalid imagined batch context')
        if retained_obs.ndim != 2 or retained_command.shape != (len(retained_obs), 3) or retained_feedback.shape != (len(retained_obs), 6):
            raise ValueError('Invalid retention set dimensions')
        self._validate((states, history_actions, xi, command, retained_obs, retained_command, retained_feedback))
        if (command.square().sum(-1) < .1 ** 2).any() or (retained_command.square().sum(-1) < .1 ** 2).any():
            raise ValueError('All calibration and retention commands must have norm >= 0.1')
        groups, inverse = torch.unique(retained_command, dim=0, return_inverse=True)
        expected = retained_command.new_tensor([[.3,0,0],[-.2,0,0],[0,0,.4],[0,0,-.4],[.6,0,0]])
        if any(not torch.isclose(retained_command, c, atol=1e-6, rtol=0).all(-1).any() for c in expected):
            raise ValueError('Retention set must cover forward, backward, both turns and jogging')
        if not torch.allclose(retained_feedback[:, 3:], torch.zeros_like(retained_feedback[:, 3:]), atol=1e-6, rtol=0):
            raise ValueError('Movement retention feedback must have zero anchor errors')
        xi_normalized = self.wm.spec.normalize_xi(xi)
        obs_history = [analytic_observation(states[i], history_actions[i], xi, self.default_joint_pos) for i in range(self.history)]
        self.value.requires_grad_(False)
        try:
            state, previous = states[-1], history_actions[-1]
            imagined_states, imagined_actions, rewards = [state], [], []
            survival = [torch.ones(n, device=self.device, dtype=self.dtype)]
            terminations = [torch.zeros(n, device=self.device, dtype=self.dtype)]
            teacher_costs = []
            for _ in range(self.horizon):
                obs = flatten_history_term_major(torch.stack(obs_history))
                feedback = self._feedback(state)
                latent, target = self._teacher(obs, command)
                new_latent = self._new_latent(obs, command, feedback)
                action = self.policy.velocity_action(obs, command, feedback)
                teacher_costs.append((new_latent - latent).square().mean(-1) + (action - target).square().mean(-1))
                state = self.wm.step(state, action, xi)
                reward = velocity_reward(state, command, action, previous, self.soft_joint_limits,
                                         self.nominal_height, self.default_joint_pos) * self.dt
                _, soft = fall_signals(state, self.nominal_height, self.velocity_cfg)
                imagined_states.append(state)
                imagined_actions.append(action)
                rewards.append(reward)
                terminations.append(soft)
                survival.append(survival[-1] * (1. - soft))
                obs_history = obs_history[1:] + [analytic_observation(state, action, xi, self.default_joint_pos)]
                previous = action
            ss, aa, rr = torch.stack(imagined_states), torch.stack(imagined_actions), torch.stack(rewards)
            dhat, alive = torch.stack(terminations), torch.stack(survival)
            command_sequence = command.expand(self.horizon + 1, -1, -1)
            rollout = ImagRollout(states=ss, actions=aa, rewards=rr, dhat=dhat, survival=alive,
                                  goals=command_sequence, e_root=ss.new_zeros((self.horizon + 1, n, 4)),
                                  frames=ss.new_zeros((self.horizon + 1, n)), reward_terms={})
            value_inputs = self._critic_inputs(ss, command_sequence, xi_normalized.expand(self.horizon + 1, -1, -1))
            terminal_value = self.value(value_inputs[-1]).squeeze(-1)
            objective = policy_objective(rollout, terminal_value, self.cfg.gamma)
            retained_latent, retained_teacher = self._teacher(retained_obs, retained_command)
            retained_new_latent = self._new_latent(retained_obs, retained_command, retained_feedback)
            retained_action = self.policy.velocity_action(retained_obs, retained_command, retained_feedback)
            if not hasattr(self, '_zero_projection_verified'):
                if not torch.equal(retained_action.detach(), retained_teacher):
                    raise AssertionError('Zero projection changed original moving actions')
                self._zero_projection_verified = True
            imagined_cost = torch.stack(teacher_costs).mean()
            retained_cost = ((retained_new_latent - retained_latent).square().mean(-1)
                             + (retained_action - retained_teacher).square().mean(-1)).mean()
            loss = -objective / self.horizon + 10. * (imagined_cost + retained_cost)
            if not torch.isfinite(loss) or not torch.isfinite(value_inputs).all() or not torch.isfinite(rr).all() or not torch.isfinite(dhat).all():
                raise FloatingPointError('Non-finite imagination or actor objective')
            self.projection_opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([self.projection], self.cfg.grad_clip_norm)
            torch.nn.utils.clip_grad_value_([self.projection], self.cfg.grad_clip_value)
            if self.projection.grad is None or not torch.isfinite(grad_norm) or not torch.isfinite(self.projection.grad).all():
                raise FloatingPointError('Missing or non-finite velocity projection gradient')
            prior_projection = self.projection.detach().clone()
            prior_optimizer = copy.deepcopy(self.projection_opt.state_dict())
            accepted, used_lr, parameter_change = False, 0., 0.
            group_metrics = []
            for attempt in range(8):
                if attempt:
                    with torch.no_grad():
                        self.projection.copy_(prior_projection)
                    self.projection_opt.load_state_dict(copy.deepcopy(prior_optimizer))
                rate = self.lr / (2 ** attempt)
                for group in self.projection_opt.param_groups:
                    group['lr'] = rate
                self.projection_opt.step()
                with torch.no_grad():
                    current = self.policy.velocity_action(retained_obs, retained_command, retained_feedback)
                    diff = current - retained_teacher
                    group_metrics = []
                    for i, group_command in enumerate(groups):
                        selected = diff[inverse == i]
                        group_metrics.append({'command': group_command.cpu().tolist(), 'count': len(selected),
                                              'rms': float(selected.square().mean().sqrt()),
                                              'max_abs': float(selected.abs().max())})
                    finite = torch.isfinite(self.projection).all() and torch.isfinite(current).all()
                if finite and all(g['rms'] <= .03 and g['max_abs'] <= .15 for g in group_metrics):
                    accepted, used_lr = True, rate
                    parameter_change = float((self.projection.detach() - prior_projection).norm())
                    break
            if not accepted:
                with torch.no_grad():
                    self.projection.copy_(prior_projection)
                self.projection_opt.load_state_dict(prior_optimizer)
                warnings.warn('All eight velocity projection steps rejected; parameters and optimizer restored', RuntimeWarning)
            for group in self.projection_opt.param_groups:
                group['lr'] = self.lr
        finally:
            self.value.requires_grad_(True)
        with torch.no_grad():
            critic_inputs = value_inputs.detach()
            values = self.value(critic_inputs).squeeze(-1)
            targets = gae_targets(rr.detach(), dhat[1:].detach(), values, self.cfg.gamma, self.cfg.gae_lambda)
            features, labels = critic_inputs[:-1].reshape(-1, critic_inputs.shape[-1]), targets.reshape(-1)
            if not torch.isfinite(values).all() or not torch.isfinite(labels).all():
                raise FloatingPointError('Non-finite critic values or targets')
        critic_loss = None
        for _ in range(self.cfg.critic_epochs):
            permutation = torch.randperm(len(labels), device=self.device)
            for indices in torch.tensor_split(permutation, min(self.cfg.critic_minibatches, len(labels))):
                critic_loss = (self.value(features[indices]).squeeze(-1) - labels[indices]).square().mean()
                if not torch.isfinite(critic_loss):
                    raise FloatingPointError('Non-finite critic loss')
                self.value_opt.zero_grad(set_to_none=True)
                critic_loss.backward()
                critic_norm = torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.cfg.critic_grad_clip)
                torch.nn.utils.clip_grad_value_(self.value.parameters(), self.cfg.grad_clip_value)
                if not torch.isfinite(critic_norm):
                    raise FloatingPointError('Non-finite critic gradients')
                self.value_opt.step()
        return {'objective': float(objective.detach()), 'reward_mean': float(rr.detach().mean()),
                'survival_final': float(alive[-1].detach().mean()), 'projection_grad_norm': float(grad_norm),
                'imagined_teacher_loss': float(imagined_cost.detach()), 'retention_teacher_loss': float(retained_cost.detach()),
                'step_accepted': accepted, 'step_attempts': attempt + 1, 'step_lr': used_lr,
                'projection_change_l2': parameter_change, 'retention_groups': group_metrics,
                'retention_metrics_are_rejected_candidate': not accepted,
                'critic_loss': float(critic_loss.detach()), 'critic_value_mean': float(values.mean()),
                'critic_target_mean': float(targets.mean()), 'physical_validation': 'pending'}

    def save_state(self):
        """Return independent projection and optimizer/value snapshots."""
        return copy.deepcopy({'velocity_feedback_weight': self.projection.detach().clone(),
                              'projection_optimizer': self.projection_opt.state_dict(),
                              'value': self.value.state_dict(), 'value_optimizer': self.value_opt.state_dict()})
