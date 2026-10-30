"""Convert phase-scheduled reference goals to SuperMimic tracker inputs."""
import json

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from mjlab.utils.lab_api.math import matrix_from_quat, subtract_frame_transforms

from supermimic.control.skill_phase import SkillPhase
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.robots.dimensions import NJ


class SkillTracker:
    def __init__(self, env, dataset, cfg):
        self.env = env
        self.cmd = env.command_manager.get_term('motion')
        names = ['walk', 'backward', 'turn_left_inplace', 'turn_right_inplace', 'jog']
        files = [dataset / (n + '.npz') for n in names] + [dataset / (n + '_stop.npz') for n in names]
        meta = json.loads((dataset / 'manifest.json').read_text())
        brake = {c['name'].removesuffix('_stop'): c['release_frame'] + 10 for c in meta['clips'] if c['type'] == 'exploratory_stop'}
        if set(brake) != set(names):
            raise ValueError('Five stop phases required')
        lib = MotionLibrary(files, self.cmd.cfg.body_names, self.cmd.library.default_joint_pos, device=env.device)
        clips = {c.name: lib.goal_feat[c.start:c.end].cpu().numpy() for c in lib.clips}
        self.scheduler = SkillPhase({n: clips[n] for n in names}, {n: clips[n+'_stop'] for n in names}, brake, NJ, horizon=cfg['env']['future_frames'])
        self.model = env.sim.mj_model
        self.ref = mujoco.MjData(self.model)
        self.anchor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'robot/' + self.cmd.cfg.anchor_body_name)
        if self.anchor < 0:
            raise ValueError('Anchor body not found')
        self.joint_default = self.cmd.library.default_joint_pos.cpu().numpy()
        self.origin = env.scene.env_origins[0].cpu().numpy()
        q = env.sim.data.qpos[0].cpu().numpy()
        self.pos = q[:3].copy() - self.origin
        self.yaw = Rotation.from_quat(q[[4, 5, 6, 3]]).as_euler('xyz')[2]

    def goal(self, mode):
        self.scheduler.select(mode)
        window = self.scheduler.window()
        g = window[0]
        grav = g[2*NJ:2*NJ+3]
        roll = np.arctan2(-grav[1], -grav[2])
        pitch = np.arcsin(np.clip(grav[0], -1., 1.))
        quat = Rotation.from_euler('xyz', [roll, pitch, self.yaw]).as_quat()[[3, 0, 1, 2]]
        self.ref.qpos[:] = np.r_[self.pos[:2], float(g[-1]), quat, g[:NJ] + self.joint_default]
        mujoco.mj_kinematics(self.model, self.ref)
        device = self.cmd.robot_anchor_pos_w.device
        target_pos = torch.as_tensor(self.ref.xpos[self.anchor] + self.origin, dtype=torch.float32, device=device)[None]
        target_quat = torch.as_tensor(self.ref.xquat[self.anchor], dtype=torch.float32, device=device)[None]
        pos, quat = subtract_frame_transforms(self.cmd.robot_anchor_pos_w, self.cmd.robot_anchor_quat_w, target_pos, target_quat)
        anchor = torch.cat([pos, matrix_from_quat(quat)[..., :2].reshape(1, -1)], dim=-1)
        return torch.cat([torch.as_tensor(window.reshape(1, -1), device=device), anchor], dim=-1)

    def advance(self):
        g = self.scheduler.window()[0]
        lin = g[2*NJ+6:2*NJ+9]
        self.pos[:2] += Rotation.from_euler('z', self.yaw).apply(lin)[:2] * self.scheduler.dt
        self.yaw += float(g[2*NJ+5]) * self.scheduler.dt
        self.scheduler.advance()
