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


def install_velocity_horizon(horizon=8):
    """Keep velocity actor/critic updates within measured model accuracy.

    Override the whole velocity update so actor normalization, rollout, and GAE
    share the same horizon. Motion tracking retains its original configuration.
    Install after the paired-supervision extension.
    """
    from dataclasses import replace
    import supermimic.locomotion.trainer as module
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError('Velocity horizon must be a positive integer')
    base = module.JointPolicyTrainer

    class BoundedVelocityTrainer(base):
        def checkpoint_state(self):
            state = super().checkpoint_state()
            state['imagination_horizon'] = horizon
            return state

        def _velocity_update(self, buffer):
            previous = self.cfg
            self.cfg = replace(previous, horizon=horizon)
            try:
                return super()._velocity_update(buffer)
            finally:
                self.cfg = previous

    module.JointPolicyTrainer = BoundedVelocityTrainer


def install_control_prior(checkpoint, weight=10.):
    """Retain validated command responses on the SAME evolving physical history.

    A frozen single-encoder checkpoint supplies training labels only. Action
    labels use its own decoder, so tracking updates cannot silently move the
    velocity interface's target when the shared decoder changes.
    """
    import copy
    import hashlib
    from pathlib import Path
    import torch
    import supermimic.locomotion.trainer as module
    from supermimic.world_model.heads import analytic_observation, flatten_history_term_major
    path=Path(checkpoint)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    base=module.JointPolicyTrainer

    class RetainedControlTrainer(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            source=torch.load(path,map_location=next(self.policy.parameters()).device,weights_only=False)
            self.control_prior=copy.deepcopy(self.policy).requires_grad_(False).eval()
            self.control_prior.load_state_dict(source['policy'],strict=True)
            self.control_prior_metrics={}
            if self.velocity_cfg.alignment_weight<=0:
                raise ValueError('Control retention requires positive alignment weight')

        def _imagine_velocity(self,start):
            rollout,align=super()._imagine_velocity(start)
            with torch.no_grad():
                history=torch.stack([analytic_observation(start.hist_states[i],start.hist_actions[i],start.xi,
                    self.lib.default_joint_pos) for i in range(self.cfg.history)])
                obs=flatten_history_term_major(history)
                choices=torch.tensor([[.3,0,0],[-.2,0,0],[0,0,.4],[0,0,-.4],[.6,0,0]],device=obs.device)
                command=torch.zeros(len(obs),3,device=obs.device)
                count=len(obs)//2
                command[count:]=choices[torch.arange(len(obs)-count,device=obs.device)%5]
                target=self.control_prior.encoders['velocity'](obs,command)
                target_action=self.control_prior.decoder(target,obs)
            latent=self.policy.encoders['velocity'](obs,command)
            latent_mse=(latent-target).square().mean()
            action_mse=(self.policy.decoder(latent,obs)-target_action).square().mean()
            self.control_prior_metrics=dict(control_prior_latent_mse=float(latent_mse.detach()),control_prior_action_mse=float(action_mse.detach()))
            return rollout,align+weight/self.velocity_cfg.alignment_weight*(latent_mse+action_mse)

        def _velocity_update(self,buffer):
            metrics=super()._velocity_update(buffer)
            metrics.update(self.control_prior_metrics)
            return metrics

        def checkpoint_state(self):
            state=super().checkpoint_state()
            state['control_prior']=dict(checkpoint=str(path),sha256=digest,weight=weight)
            return state

        def load_checkpoint_state(self,state):
            if 'control_prior' in state and state['control_prior']['sha256']!=digest:
                raise ValueError('Control prior changed across resume')
            return super().load_checkpoint_state(state)

    module.JointPolicyTrainer=RetainedControlTrainer
