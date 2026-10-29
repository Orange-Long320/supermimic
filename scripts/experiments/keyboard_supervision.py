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
