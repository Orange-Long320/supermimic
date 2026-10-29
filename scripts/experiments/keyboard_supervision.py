"""Install persistent paired tracking supervision in the pinned keyboard trainer.

Import/install before constructing Trainer. No change to deployed policy inputs.
"""

def install():
    import torch
    import supermimic.locomotion.trainer as module
    from supermimic.locomotion.task import bounded_reference
    from supermimic.world_model.heads import analytic_observation, flatten_history_term_major, anchor_relative_obs
    from supermimic.world_model.wm_trainer import reference_at

    class PairedJointPolicyTrainer(module.JointPolicyTrainer):
        def _velocity_update(self, buffer):
            # The motion half resets at clip end and retains paired histories.
            # Velocity episodes intentionally outlive clips; their reference flag
            # must not control whether the new encoder gets teacher supervision.
            self._teacher_start = buffer.sample_starts(self.velocity_batch, self.cfg.history,
                                                       env_mask=~buffer.velocity_env_mask)
            try:
                metrics = super()._velocity_update(buffer)
                metrics['teacher_valid_fraction'] = self._teacher_valid_fraction
                return metrics
            finally:
                self._teacher_start = None

        def _imagine_velocity(self, start):
            rollout, _ = super()._imagine_velocity(start)
            s = self._teacher_start
            command, valid = bounded_reference(self.reference_velocity, s.frame, self.velocity_cfg)
            self._teacher_valid_fraction = float(valid.float().mean())
            with torch.no_grad():
                history = torch.stack([analytic_observation(s.hist_states[i], s.hist_actions[i], s.xi,
                    self.lib.default_joint_pos) for i in range(self.cfg.history)])
                observations = flatten_history_term_major(history)
                goal = self.lib.goal_window(s.frame, self.cfg.future_frames)
                if self.cfg.anchor_obs:
                    bodies = self.wm.heads.robot_bodies(s.state)
                    ref = reference_at(self.lib, s.frame)
                    ai = self.wm.heads.anchor_index
                    relative = anchor_relative_obs(bodies['pos'][:, ai], bodies['quat'][:, ai],
                        ref.body_pos_w[:, ai], ref.body_quat_w[:, ai])
                    goal = torch.cat((goal, relative), -1)
                teacher = self.policy.encoders['motion'](goal)
            latent = self.policy.encoders['velocity'](observations, command)
            error = (latent - teacher).square().mean(-1)
            align = (error * valid).sum() / valid.sum().clamp_min(1)
            return rollout, align

    module.JointPolicyTrainer = PairedJointPolicyTrainer


def install_quiet_stop_cost():
    """Carry the physical quiet-stop joint-motion costs into imagined updates.

    Only zero commands receive these costs. Deployment and moving rewards stay
    unchanged. The extension still needs physical regression after training.
    """
    import torch
    import supermimic.locomotion.trainer as module
    from supermimic.locomotion.task import measured_velocity
    from supermimic.robots.dimensions import PROFILE, NQ

    base = module.velocity_reward
    arms = [i for i, name in enumerate(PROFILE['joint_names'])
            if any(key in name for key in ('shoulder', 'elbow', 'wrist'))]
    if len(arms) != 14:
        raise ValueError('Expected X2 shoulder/elbow/wrist joints')

    def reward(state, command, action, previous_action, joint_limits, nominal_height, default_joints):
        original = base(state, command, action, previous_action, joint_limits, nominal_height, default_joints)
        standing = command.square().sum(-1) < 1e-6
        speed = measured_velocity(state)
        dq, q = state[..., NQ + 6:], state[..., 7:NQ]
        cost = (2 * dq[..., arms].square().mean(-1)
                + .1 * (q[..., arms] - default_joints[arms]).square().mean(-1)
                + .05 * dq.square().mean(-1)
                + 10 * speed[..., :2].square().sum(-1) + 2 * speed[..., 2].square())
        return original - cost * standing

    module.velocity_reward = reward
