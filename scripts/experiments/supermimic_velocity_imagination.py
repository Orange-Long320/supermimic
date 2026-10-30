"""Calibrate a velocity feedback projection through a frozen world model.

Real trajectory collection, world-model validation, checkpoint persistence and
physical acceptance belong to the caller. This module uses deterministic actions.
"""

import copy
import math
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
                 velocity_cfg, history, dt, lr=3e-5, horizon=32, precision_weight=0.0):
        if history <= 0 or horizon <= 0 or dt <= 0 or lr <= 0:
            raise ValueError('history, horizon, dt and lr must be positive')
        if not math.isfinite(precision_weight) or precision_weight < 0:
            raise ValueError('precision_weight must be finite and nonnegative')
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
        self.precision_weight = float(precision_weight)
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
        self.retention = None

    def set_retention(self, obs, command, feedback, source_action):
        """Bind an immutable copy of real rollout states and original full-policy actions."""
        if self.retention is not None:
            raise ValueError('Retention is already bound; create a new trainer for another set')
        if obs.ndim != 2 or len(obs) == 0 or command.shape != (len(obs), 3) or feedback.shape != (len(obs), 6):
            raise ValueError('Invalid retention shapes')
        for tensor in (obs, command, feedback, source_action):
            if tensor.device != self.device or tensor.dtype != self.dtype or not torch.isfinite(tensor).all():
                raise ValueError('Retention tensors must be finite and match actor device and dtype')
        norms = command.square().sum(-1).sqrt()
        expected = command.new_tensor([[.3, 0, 0], [-.2, 0, 0], [0, 0, .4], [0, 0, -.4], [.6, 0, 0]])
        if not (norms == 0).any() or not ((norms > 0) & (norms < .1)).any():
            raise ValueError('Retention must cover zero and near-zero commands')
        if any(not torch.isclose(command, c, atol=1e-6, rtol=0).all(-1).any() for c in expected):
            raise ValueError('Retention must cover all five full commands')
        with torch.no_grad():
            teacher = self.policy.velocity_action(obs, command, feedback, disable_moving_feedback=True)
        if teacher.shape != source_action.shape or not torch.isfinite(teacher).all():
            raise ValueError('Invalid source action shape or teacher output')
        source_difference = float((teacher - source_action).abs().max())
        if source_difference > 2e-5:
            raise ValueError(f'Retention source actions differ from fixed teacher: {source_difference}')
        self.retention = dict(obs=obs.detach().clone(), command=command.detach().clone(),
                              feedback=feedback.detach().clone(), teacher=teacher.detach().clone(),
                              source_action_max_abs_diff=source_difference)

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

    def update(self, batch):
        """Update from real histories and a fixed real-state retention set.

        Histories are oldest first; their last action is the previous action for
        the final state. Each command is the actual command at that state.
        """
        if any(k not in batch for k in ('hist_states', 'hist_actions', 'xi', 'command')):
            raise ValueError('batch requires hist_states, hist_actions, xi and command')
        if self.retention is None:
            raise ValueError('Call set_retention before updating')
        states, history_actions, xi, command = (batch[k].detach() for k in ('hist_states', 'hist_actions', 'xi', 'command'))
        retained_obs, retained_command, retained_feedback = (self.retention[k] for k in ('obs', 'command', 'feedback'))
        if states.ndim != 3 or states.shape[0] != self.history or history_actions.ndim != 3 or history_actions.shape[:2] != states.shape[:2]:
            raise ValueError('Histories must have matching (history, batch, feature) dimensions')
        n = states.shape[1]
        if not n or xi.shape != (n, self.wm.spec.xi_offset.numel()) or command.shape != (n, 3):
            raise ValueError('Invalid imagined batch context')
        if retained_obs.ndim != 2 or retained_command.shape != (len(retained_obs), 3) or retained_feedback.shape != (len(retained_obs), 6):
            raise ValueError('Invalid retention set dimensions')
        self._validate((states, history_actions, xi, command, retained_obs, retained_command, retained_feedback))
        if (command.square().sum(-1) < .1 ** 2).any():
            raise ValueError('Imagined calibration commands must have norm >= 0.1')
        norms = retained_command.square().sum(-1).sqrt()
        expected = retained_command.new_tensor([[.3,0,0],[-.2,0,0],[0,0,.4],[0,0,-.4],[.6,0,0]])
        masks = [('zero', norms == 0), ('near_zero', (norms > 0) & (norms < .1))]
        masks += [(name, torch.isclose(retained_command, c, atol=1e-6, rtol=0).all(-1))
                  for name, c in zip(('forward', 'backward', 'left', 'right', 'jog'), expected)]
        covered = torch.stack([mask for _, mask in masks]).any(0)
        masks.append(('other_ramp', ~covered))
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
                                         self.nominal_height, self.default_joint_pos)
                if self.precision_weight:
                    scale = state.new_tensor([.05, .05, .08])
                    reward = reward - self.precision_weight * ((measured_velocity(state) - command) / scale).square().sum(-1)
                reward = reward * self.dt
                _, soft = fall_signals(state, self.nominal_height, self.velocity_cfg)
                imagined_states.append(state)
                imagined_actions.append(action)
                rewards.append(reward)
                terminations.append(soft)
                survival.append(survival[-1] * (1. - soft))
                obs_history = obs_history[1:] + [analytic_observation(state, action, xi, self.default_joint_pos)]
                previous = action
            ss, aa, rr = torch.stack(imagined_states), torch.stack(imagined_actions), torch.stack(rewards)
            imagined_velocity_rmse = (measured_velocity(ss[1:]) - command).square().mean((0, 1)).sqrt().detach()
            dhat, alive = torch.stack(terminations), torch.stack(survival)
            command_sequence = command.expand(self.horizon + 1, -1, -1)
            rollout = ImagRollout(states=ss, actions=aa, rewards=rr, dhat=dhat, survival=alive,
                                  goals=command_sequence, e_root=ss.new_zeros((self.horizon + 1, n, 4)),
                                  frames=ss.new_zeros((self.horizon + 1, n)), reward_terms={})
            value_inputs = self._critic_inputs(ss, command_sequence, xi_normalized.expand(self.horizon + 1, -1, -1))
            terminal_value = self.value(value_inputs[-1]).squeeze(-1)
            objective = policy_objective(rollout, terminal_value, self.cfg.gamma)
            retained_latent, _ = self._teacher(retained_obs, retained_command)
            retained_teacher = self.retention['teacher']
            retained_new_latent = self._new_latent(retained_obs, retained_command, retained_feedback)
            retained_action = self.policy.velocity_action(retained_obs, retained_command, retained_feedback)
            if not hasattr(self, '_zero_projection_verified'):
                if not torch.equal(retained_action.detach(), retained_teacher):
                    raise AssertionError('Zero projection changed original moving actions')
                self._zero_projection_verified = True
            imagined_cost = torch.stack(teacher_costs).mean()
            full_movement = norms >= .1
            retained_cost = (retained_action - retained_teacher).square().mean()
            if full_movement.any():
                retained_cost = retained_cost + (retained_new_latent[full_movement] - retained_latent[full_movement]).square().mean()
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
            initial_obs = flatten_history_term_major(torch.stack([analytic_observation(states[i], history_actions[i], xi, self.default_joint_pos) for i in range(self.history)]))
            initial_feedback = self._feedback(states[-1])
            with torch.no_grad():
                initial_action = self.policy.velocity_action(initial_obs, command, initial_feedback).clone()
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
                    for name, mask in masks:
                        selected = diff[mask]
                        if not len(selected):
                            continue
                        group_metrics.append({'group': name, 'count': len(selected),
                                              'max_sample_rms': float(selected.square().mean(-1).sqrt().max()),
                                              'max_abs': float(selected.abs().max())})
                    initial_new = self.policy.velocity_action(initial_obs, command, initial_feedback)
                    update_rms = float((initial_new - initial_action).square().mean().sqrt())
                    zero_diff = diff[norms == 0]
                    finite = torch.isfinite(self.projection).all() and torch.isfinite(current).all()
                if (finite and update_rms <= .02 and torch.equal(zero_diff, torch.zeros_like(zero_diff))
                        and all(g['max_sample_rms'] <= .03 and g['max_abs'] <= .15 for g in group_metrics)):
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
                'precision_weight': self.precision_weight,
                'imagined_velocity_rmse_before_actor_step': imagined_velocity_rmse.cpu().tolist(),
                'survival_final': float(alive[-1].detach().mean()), 'projection_grad_norm': float(grad_norm),
                'imagined_teacher_loss': float(imagined_cost.detach()), 'retention_teacher_loss': float(retained_cost.detach()),
                'step_accepted': accepted, 'step_attempts': attempt + 1, 'step_lr': used_lr,
                'projection_change_l2': parameter_change, 'retention_groups': group_metrics,
                'update_action_rms': update_rms,
                'retention_metrics_are_rejected_candidate': not accepted,
                'critic_loss': float(critic_loss.detach()), 'critic_value_mean': float(values.mean()),
                'critic_target_mean': float(targets.mean()), 'physical_validation': 'pending'}

    def save_state(self):
        """Return independent projection and optimizer/value snapshots."""
        return copy.deepcopy({'velocity_feedback_weight': self.projection.detach().clone(),
                              'projection_optimizer': self.projection_opt.state_dict(),
                              'value': self.value.state_dict(), 'value_optimizer': self.value_opt.state_dict(),
                              'retention': self.retention, 'precision_weight': self.precision_weight})
