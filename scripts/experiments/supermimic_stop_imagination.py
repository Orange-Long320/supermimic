"""Differentiable stop-skill updates through a frozen SuperMimic world model.

The caller supplies real, contiguous observation histories and is responsible for
collection, world-model validation, checkpointing, and physical acceptance.
"""

import copy
import warnings

import torch
from torch import nn

from supermimic.imagination.objective import gae_targets, policy_objective
from supermimic.imagination.policy_trainer import PolicyTrainerCfg
from supermimic.imagination.rollout import ImagRollout
from supermimic.kinematics.rotations import heading_yaw, projected_gravity
from supermimic.locomotion.task import fall_signals, measured_velocity
from supermimic.robots.dimensions import NQ, STATE_DIM
from supermimic.world_model.heads import analytic_observation, flatten_history_term_major


class StopImaginationTrainer:
    """Train only the standing expert; input and output tensors stay on its device."""

    def __init__(self, policy, wm, default_joint_pos, qref, arm_ids, arm_target,
                 nominal_height, velocity_cfg, history, dt, lr=3e-6, horizon=32, train_base=False):
        if history <= 0 or horizon <= 0 or dt <= 0 or lr <= 0:
            raise ValueError('history, horizon, dt and lr must be positive')
        if abs(float(wm.spec.dt) - float(dt)) > 1e-8:
            raise ValueError('World-model time step does not match the collection time step')
        if not hasattr(policy, 'standing_skill') or policy.standing_skill.mode != 'expert' or policy.standing_skill.feedback_dim != 6:
            raise ValueError('A six-input standing expert is required')
        self.policy, self.wm = policy, wm
        self.actor = policy.standing_skill
        policy.requires_grad_(False)
        self.actor.requires_grad_(False)
        self.train_base = bool(train_base)
        if self.train_base:
            self.actor.requires_grad_(True)
        else:
            self.actor.feedback_weight.requires_grad_(True)
        self.wm.requires_grad_(False)
        self.teacher = copy.deepcopy(self.actor).requires_grad_(False)
        self.actor_parameter_names = [name for name, param in self.actor.named_parameters() if param.requires_grad]
        self.actor_params = [param for param in self.actor.parameters() if param.requires_grad]
        self.cold_retention = None
        self.device = next(self.actor.parameters()).device
        self.dtype = next(self.actor.parameters()).dtype
        self.default_joint_pos = torch.as_tensor(default_joint_pos, device=self.device, dtype=self.dtype).detach().clone()
        self.qref = torch.as_tensor(qref, device=self.device, dtype=self.dtype).detach().clone()
        self.arm_ids = torch.as_tensor(arm_ids, device=self.device, dtype=torch.long).detach().clone()
        self.arm_target = torch.as_tensor(arm_target, device=self.device, dtype=self.dtype).detach().clone()
        if self.default_joint_pos.shape != (NQ - 7,) or self.qref.shape != (NQ - 7,) or self.arm_ids.ndim != 1 or not self.arm_ids.numel() or self.arm_target.shape != self.arm_ids.shape:
            raise ValueError('Invalid joint target dimensions')
        if (self.arm_ids < 0).any() or (self.arm_ids >= NQ - 7).any():
            raise ValueError('Arm joint index out of range')
        self.nominal_height = float(nominal_height)
        self.velocity_cfg, self.history, self.dt, self.horizon, self.lr = velocity_cfg, history, float(dt), horizon, float(lr)
        self.cfg = PolicyTrainerCfg(horizon=horizon, history=history, lr=lr)
        # canonical state + six task inputs + normalized context + normalized stop age.
        prototype = torch.zeros(1, STATE_DIM, device=self.device, dtype=self.dtype)
        prototype[:, 3] = 1.
        feature_dim = wm.spec.canonical(prototype).shape[-1]
        critic_dim = feature_dim + 6 + wm.spec.xi_offset.numel() + 1
        self.value = nn.Sequential(nn.Linear(critic_dim, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1)).to(device=self.device, dtype=self.dtype)
        # A fresh task critic starts at zero instead of injecting an arbitrary
        # terminal-state gradient into the first model-based actor update.
        nn.init.zeros_(self.value[-1].weight)
        nn.init.zeros_(self.value[-1].bias)
        self.actor_opt = torch.optim.Adam(self.actor_params, lr=lr, betas=self.cfg.betas)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=self.cfg.critic_lr)

    def set_cold_retention(self, obs, feedback, source_action=None):
        """Bind verified real standing observations and their fixed source actions."""
        if obs.ndim != 2 or len(obs) == 0 or feedback.shape != (len(obs), 6):
            raise ValueError('Retention observations and feedback must be nonempty and aligned')
        tensors = (obs, feedback) if source_action is None else (obs, feedback, source_action)
        if any(t.device != self.device or t.dtype != self.dtype or not torch.isfinite(t).all() for t in tensors):
            raise ValueError('Retention tensors must be finite and match the actor device and dtype')
        with torch.no_grad():
            teacher_action = self.teacher(obs, feedback)
        if not torch.isfinite(teacher_action).all():
            raise ValueError('Teacher produced non-finite retention actions')
        difference = 0.
        if source_action is not None:
            if source_action.shape != teacher_action.shape:
                raise ValueError('Retention source action shape does not match teacher output')
            difference = float((source_action - teacher_action).abs().max())
            if difference > 2e-5:
                raise ValueError(f'Retention source action differs from teacher by {difference:.8g} (limit 2e-5)')
        self.cold_retention = {'obs': obs.detach().clone(), 'feedback': feedback.detach().clone(),
                               'teacher_action': teacher_action.detach().clone(),
                               'source_action_max_abs_diff': difference,
                               'source_action_supplied': source_action is not None}

    def _feedback(self, state, anchor_xy, anchor_yaw):
        yaw = heading_yaw(state[..., 3:7])
        delta = anchor_xy - state[..., :2]
        c, s = torch.cos(yaw), torch.sin(yaw)
        position = torch.stack((c * delta[..., 0] + s * delta[..., 1], -s * delta[..., 0] + c * delta[..., 1]), -1)
        heading = torch.atan2(torch.sin(anchor_yaw - yaw), torch.cos(anchor_yaw - yaw))[..., None]
        return torch.cat((measured_velocity(state), position, heading), -1)

    def _value_inputs(self, state, feedback, xi_normalized, stop_age):
        return torch.cat((self.wm.spec.canonical(state), feedback, xi_normalized, (stop_age / 66.)[..., None]), -1)

    def _reward(self, state, feedback, stop_age, soft_fall):
        speed = feedback[..., :3]
        q, dq = state[..., 7:NQ], state[..., NQ + 6:]
        entry = stop_age < .8
        motion_weight = torch.where(entry, .1, 10.)
        pose_weight = torch.where(entry, 10., .5)
        full_motion_weight = torch.where(entry, .01, .1)
        reward = (2. - 30. * speed[..., :2].square().sum(-1) - 40. * speed[..., 2].square()
                  - 200. * feedback[..., 3:5].square().sum(-1) - 40. * feedback[..., 5].square()
                  - 2. * projected_gravity(state[..., 3:7])[..., :2].square().sum(-1)
                  - 10. * (state[..., 2] - self.nominal_height).square()
                  - motion_weight * dq[..., self.arm_ids].square().mean(-1)
                  - pose_weight * (q[..., self.arm_ids] - self.arm_target).square().mean(-1)
                  - 2. * (q - self.qref).square().mean(-1)
                  - full_motion_weight * dq.square().mean(-1) - 20. * soft_fall)
        # Match the existing SuperMimic velocity-imagination reward convention.
        return reward * self.dt

    def update(self, batch):
        """Update from a dict of histories and frozen stop anchors; return diagnostics.

        hist_states/history actions are oldest first, with the final state used as
        the rollout start and the final action as its previous action.
        """
        required = ('hist_states', 'hist_actions', 'xi', 'anchor_xy', 'anchor_yaw', 'stop_age', 'is_cold')
        if self.cold_retention is None:
            raise ValueError('Call set_cold_retention with the verified standing dataset before updating')
        if any(key not in batch for key in required):
            raise ValueError(f'Batch must contain {required}')
        states = batch['hist_states'].detach()
        actions = batch['hist_actions'].detach()
        xi = batch['xi'].detach()
        anchor_xy = batch['anchor_xy'].detach()
        anchor_yaw = batch['anchor_yaw'].detach()
        stop_age = batch['stop_age'].detach()
        cold = batch['is_cold'].detach()
        if states.ndim != 3 or states.shape[0] != self.history or actions.ndim != 3 or actions.shape[:2] != states.shape[:2]:
            raise ValueError('Histories must have matching (history, batch, feature) dimensions')
        n = states.shape[1]
        if not n or xi.shape != (n, self.wm.spec.xi_offset.numel()) or anchor_xy.shape != (n, 2) or anchor_yaw.shape != (n,) or stop_age.shape != (n,) or cold.shape != (n,) or cold.dtype != torch.bool:
            raise ValueError('Invalid batch context dimensions')
        if any(t.device != self.device for t in (states, actions, xi, anchor_xy, anchor_yaw, stop_age, cold)):
            raise ValueError('Batch tensors must already be on the actor device')
        if any(not torch.isfinite(t).all() for t in (states, actions, xi, anchor_xy, anchor_yaw, stop_age)) or (stop_age < 0).any():
            raise ValueError('Batch contains invalid or non-finite values')
        xi_normalized = self.wm.spec.normalize_xi(xi)
        obs_history = [analytic_observation(states[i], actions[i], xi, self.default_joint_pos) for i in range(self.history)]
        initial_obs = flatten_history_term_major(torch.stack(obs_history))
        initial_feedback = self._feedback(states[-1], anchor_xy, anchor_yaw)
        with torch.no_grad():
            old_action = self.actor(initial_obs, initial_feedback).detach().clone()
            teacher_action = self.teacher(initial_obs[cold], initial_feedback[cold]) if cold.any() else None
        self.value.requires_grad_(False)
        try:
            state = states[-1]
            imagined_states, imagined_actions, rewards = [state], [], []
            feedbacks = [initial_feedback]
            terminations = [torch.zeros(n, device=self.device, dtype=self.dtype)]
            survival = [torch.ones(n, device=self.device, dtype=self.dtype)]
            ages = [stop_age]
            for t in range(self.horizon):
                obs = flatten_history_term_major(torch.stack(obs_history))
                action = self.actor(obs, feedbacks[-1])
                state = self.wm.step(state, action, xi)
                feedback = self._feedback(state, anchor_xy, anchor_yaw)
                age = stop_age + (t + 1) * self.dt
                _, soft = fall_signals(state, self.nominal_height, self.velocity_cfg)
                reward = self._reward(state, feedback, age, soft)
                imagined_states.append(state)
                imagined_actions.append(action)
                feedbacks.append(feedback)
                ages.append(age)
                rewards.append(reward)
                terminations.append(soft)
                survival.append(survival[-1] * (1. - soft))
                obs_history = obs_history[1:] + [analytic_observation(state, action, xi, self.default_joint_pos)]
            ss, aa, rr = torch.stack(imagined_states), torch.stack(imagined_actions), torch.stack(rewards)
            ff, ages = torch.stack(feedbacks), torch.stack(ages)
            dhat, alive = torch.stack(terminations), torch.stack(survival)
            # The objective uses rewards and survival; placeholders satisfy its shared rollout interface.
            rollout = ImagRollout(states=ss, actions=aa, rewards=rr, dhat=dhat, survival=alive,
                                  goals=ff, e_root=ff[..., 2:6], frames=ages, reward_terms={})
            value_inputs = self._value_inputs(ss, ff, xi_normalized.expand(self.horizon + 1, -1, -1), ages)
            terminal_value = self.value(value_inputs[-1]).squeeze(-1)
            objective = policy_objective(rollout, terminal_value, self.cfg.gamma)
            cold_prior = ((self.actor(initial_obs[cold], initial_feedback[cold]) - teacher_action).square().mean()
                          if cold.any() else objective.new_zeros(()))
            retained = self.cold_retention
            retention_prior = (self.actor(retained['obs'], retained['feedback']) - retained['teacher_action']).square().mean()
            loss = -objective / self.horizon + 10. * cold_prior + 10. * retention_prior
            if not torch.isfinite(loss) or not torch.isfinite(value_inputs).all() or not torch.isfinite(rr).all() or not torch.isfinite(dhat).all():
                raise FloatingPointError('Non-finite imagination or actor objective')
            self.actor_opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_params, self.cfg.grad_clip_norm)
            torch.nn.utils.clip_grad_value_(self.actor_params, self.cfg.grad_clip_value)
            if not torch.isfinite(grad_norm) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.actor_params):
                raise FloatingPointError('Non-finite standing actor gradients')
            backup = [p.detach().clone() for p in self.actor_params]
            opt_backup = copy.deepcopy(self.actor_opt.state_dict())
            accepted, action_change, parameter_change, used_lr = False, float('nan'), 0., 0.
            retention_max_rms = retention_max_abs = cold_max_rms = cold_max_abs = float('nan')
            for attempt in range(8):
                if attempt:
                    with torch.no_grad():
                        for p, before in zip(self.actor_params, backup):
                            p.copy_(before)
                    self.actor_opt.load_state_dict(copy.deepcopy(opt_backup))
                rate = self.lr / (2 ** attempt)
                for group in self.actor_opt.param_groups:
                    group['lr'] = rate
                self.actor_opt.step()
                with torch.no_grad():
                    new_action = self.actor(initial_obs, initial_feedback)
                    change = (new_action - old_action).square().mean().sqrt()
                    retained_diff = self.actor(retained['obs'], retained['feedback']) - retained['teacher_action']
                    retention_max_rms = float(retained_diff.square().mean(-1).sqrt().max())
                    retention_max_abs = float(retained_diff.abs().max())
                    if cold.any():
                        cold_diff = self.actor(initial_obs[cold], initial_feedback[cold]) - teacher_action
                        cold_max_rms = float(cold_diff.square().mean(-1).sqrt().max())
                        cold_max_abs = float(cold_diff.abs().max())
                    else:
                        cold_max_rms = cold_max_abs = 0.
                    finite = (torch.isfinite(new_action).all() and torch.isfinite(retained_diff).all()
                              and all(torch.isfinite(p).all() for p in self.actor_params))
                    action_change = float(change)
                if (finite and torch.isfinite(change) and change <= .02
                        and retention_max_rms <= .03 and retention_max_abs <= .15
                        and cold_max_rms <= .03 and cold_max_abs <= .15):
                    accepted, used_lr = True, rate
                    with torch.no_grad():
                        parameter_change = float(torch.sqrt(sum((p - before).square().sum() for p, before in zip(self.actor_params, backup))))
                    break
            if not accepted:
                with torch.no_grad():
                    for p, before in zip(self.actor_params, backup):
                        p.copy_(before)
                self.actor_opt.load_state_dict(opt_backup)
                warnings.warn('All eight standing actor step sizes were rejected; parameters and optimizer restored', RuntimeWarning)
            for group in self.actor_opt.param_groups:
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
                predicted = self.value(features[indices]).squeeze(-1)
                critic_loss = (predicted - labels[indices]).square().mean()
                if not torch.isfinite(critic_loss):
                    raise FloatingPointError('Non-finite critic loss')
                self.value_opt.zero_grad(set_to_none=True)
                critic_loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.cfg.critic_grad_clip)
                torch.nn.utils.clip_grad_value_(self.value.parameters(), self.cfg.grad_clip_value)
                if not torch.isfinite(norm):
                    raise FloatingPointError('Non-finite critic gradients')
                self.value_opt.step()
        return {'objective': float(objective.detach()), 'reward_mean': float(rr.detach().mean()),
                'survival_final': float(alive[-1].detach().mean()), 'actor_grad_norm': float(grad_norm),
                'cold_prior': float(cold_prior.detach()), 'retention_prior': float(retention_prior.detach()),
                'train_base': self.train_base, 'actor_parameter_names': self.actor_parameter_names,
                'actor_step_accepted': accepted,
                'actor_step_attempts': attempt + 1, 'actor_step_lr': used_lr,
                'initial_action_change_rms': action_change, 'actor_parameter_change_l2': parameter_change,
                'cold_max_sample_action_rms': cold_max_rms, 'cold_max_abs_action_diff': cold_max_abs,
                'retention_max_sample_action_rms': retention_max_rms, 'retention_max_abs_action_diff': retention_max_abs,
                'action_metrics_are_rejected_candidate': not accepted,
                'critic_loss': float(critic_loss.detach()), 'critic_value_mean': float(values.mean()),
                'critic_target_mean': float(targets.mean()), 'cold_count': int(cold.sum())}

    def save_state(self):
        """Return independent optimizer and value snapshots; the caller saves actor weights."""
        return copy.deepcopy({'actor_optimizer': self.actor_opt.state_dict(), 'value': self.value.state_dict(),
                              'value_optimizer': self.value_opt.state_dict(), 'train_base': self.train_base,
                              'actor_parameter_names': self.actor_parameter_names,
                              'cold_retention': self.cold_retention})
