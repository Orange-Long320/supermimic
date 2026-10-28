"""Short CPU MuJoCo previews with the frozen policy's actual input history.

This is a local predictor. Executed trajectories still require independent
native Warp simulation and the unchanged full physical acceptance gates.
"""
import mujoco
import numpy as np
import torch

from supermimic.data.command_basis import early_contact_command_basis, preparation_command_basis
from supermimic.data.design_metrics import BatchedDesignMetrics
from supermimic.robots.dimensions import PROFILE, XI_DIM
from supermimic.world_model.heads import OBS_TERMS, analytic_observation, anchor_relative_obs, flatten_history_term_major


class NativePolicyPreview:
    def __init__(self, model, policy, library, groups, parameters, takeoff_frame, landing_frame,
                 initial_frame, anchor_obs=True, future_frames=10, contact_pulses=False):
        self.model = model
        self.data = mujoco.MjData(model)
        self.policy = policy
        self.library = library
        self.parameters = torch.tensor(parameters, dtype=torch.float32)
        self.takeoff = int(takeoff_frame); self.landing = int(landing_frame)
        self.initial_frame = int(initial_frame)
        self.future_frames = int(future_frames); self.anchor_obs = bool(anchor_obs)
        self.contact_pulses = bool(contact_pulses)
        self.count = len(groups)
        expected = 1+(6 if contact_pulses else 4)*self.count
        if self.parameters.shape != (expected,):
            raise ValueError('Preview requires explicit preparation parameters with an initial knot')
        names = PROFILE['joint_names']
        if [model.joint(int(j)).name.rsplit('/', 1)[-1] for j in model.actuator_trnid[:, 0]] != list(names):
            raise ValueError('Preview actuator order differs from robot profile')
        self.q0 = torch.tensor([PROFILE['default_joint_pos'][n] for n in names], dtype=torch.float32)
        scale = PROFILE['action_scale']
        self.scale = np.array([scale[n] for n in names]) if isinstance(scale, dict) else np.full(len(names), scale)
        mapping = np.zeros((len(groups), len(names)))
        for k, group in enumerate(groups):
            for name, sign in group:
                j = names.index(name)
                mapping[k, j] = sign*PROFILE['effort_limit'][name]/PROFILE['stiffness'][name]/self.scale[j]
        self.mapping = torch.tensor(mapping, dtype=torch.float32)
        self.anchor_native = model.body('robot/'+PROFILE['anchor_body']).id
        self.anchor_reference = tuple(PROFILE['tracked_bodies']).index(PROFILE['anchor_body'])
        self.ratio = round(.02/model.opt.timestep)
        if abs(self.ratio*model.opt.timestep-.02) > 1e-8:
            raise ValueError('Preview requires the native 50 Hz controller')
        geometry = BatchedDesignMetrics(model, 'cpu')
        self.vertices = geometry.vertices.numpy(); self.geom_ids = geometry.geom_ids.numpy()
        self.geometry_groups = [g.numpy() for g in geometry.groups]

    def residual(self, time):
        p = self.parameters; count = self.count
        initial = p[1+3*count:1+4*count]
        knots = torch.cat([initial[None], p[1:1+3*count].reshape(3, count)], 0)
        weights = preparation_command_basis(torch.tensor(time, dtype=torch.float32), p[0])
        command = weights@knots@self.mapping
        if self.contact_pulses:
            pulse = early_contact_command_basis(torch.tensor(time, dtype=torch.float32))
            command += pulse@p[1+4*count:].reshape(2, count)@self.mapping
        return command

    def frame_at(self, time):
        # Match the native discrete retimed clock, including its reset lead.
        prep = self.parameters[0]
        t = torch.tensor(time, dtype=torch.float32)
        frame = torch.where(t <= prep, t/prep*self.takeoff, self.takeoff+(t-prep)*50).round().long()
        return torch.minimum(frame+self.initial_frame, self.library.clip_last[0]).reshape(1)

    def goal(self, time):
        frame = self.frame_at(time)
        goal = self.library.goal_window(frame, self.future_frames)
        if self.anchor_obs:
            d = self.data
            pos = torch.tensor(d.xpos[self.anchor_native], dtype=torch.float32)[None]
            quat = torch.tensor(d.xquat[self.anchor_native], dtype=torch.float32)[None]
            reference = self.library.bodies(frame)
            rel = anchor_relative_obs(pos, quat, reference.pos_w[:, self.anchor_reference], reference.quat_w[:, self.anchor_reference])
            goal = torch.cat([goal, rel], -1)
        return goal

    def geometry_heights(self):
        d = self.data
        matrix = d.geom_xmat.reshape(-1, 3, 3)
        z = (matrix[self.geom_ids, 2]*self.vertices).sum(-1)+d.geom_xpos[self.geom_ids, 2]
        return np.array([z[g].min() for g in self.geometry_groups])

    @staticmethod
    def history_from_observation(observation):
        observation = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        width = sum(size for _, size in OBS_TERMS)
        if observation.shape[-1] % width:
            raise ValueError('Observation history does not match robot dimensions')
        length = observation.shape[-1]//width
        chunks = []; offset = 0
        for _, size in OBS_TERMS:
            chunks.append(observation[:, offset:offset+length*size].reshape(1, length, size).permute(1, 0, 2))
            offset += length*size
        return torch.cat(chunks, -1)

    @torch.no_grad()
    def predict(self, qpos, qvel, observation, time, horizon=2, correction=None, initial_goal=None):
        """Predict H control frames, correcting only the first position action.

        The next policy input includes the predicted state and applied action.
        Native state is untouched; MuJoCo owns a separate MjData.
        """
        d = self.data; m = self.model
        mujoco.mj_resetData(m, d)
        d.qpos[:] = qpos; d.qvel[:] = qvel
        mujoco.mj_forward(m, d)
        history = self.history_from_observation(observation)
        states = []; actions = []; heights = []; goals = []
        for h in range(horizon):
            current_time = time+h*.02
            goal = (torch.as_tensor(initial_goal, dtype=torch.float32).reshape(1, -1)
                    if h == 0 and initial_goal is not None else self.goal(current_time))
            action = self.policy(flatten_history_term_major(history), goal)[0]+self.residual(current_time)
            if h == 0 and correction is not None:
                action += torch.as_tensor(correction, dtype=torch.float32)
            actions.append(action.numpy().copy()); goals.append(goal[0].numpy().copy())
            d.ctrl[:] = self.q0.numpy()+action.numpy()*self.scale
            for _ in range(self.ratio):
                mujoco.mj_step(m, d); mujoco.mj_forward(m, d)
                states.append(np.r_[d.qpos, d.qvel]); heights.append(self.geometry_heights())
            state = torch.tensor(np.r_[d.qpos, d.qvel], dtype=torch.float32)[None]
            obs = analytic_observation(state, action[None], torch.zeros(1, XI_DIM), self.q0)
            history = torch.cat([history[1:], obs[None]], 0)
        return {'states': np.asarray(states), 'actions': np.asarray(actions),
                'heights': np.asarray(heights), 'goals': np.asarray(goals)}
