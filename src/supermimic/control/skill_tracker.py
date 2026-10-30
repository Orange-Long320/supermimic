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


def choose_cycle(path, minimum_speed):
    """Rank sustained moving segments by pose, velocity and foot-height seams."""
    with np.load(path, allow_pickle=False) as z:
        q, v = z['joint_pos'], z['joint_vel']
        names = list(z['body_names'])
        root = names.index('pelvis')
        feet = [names.index(s+'_ankle_roll_link') for s in ('left', 'right')]
        root_vel = z['body_lin_vel_w'][:, root, :2]
        foot_height = z['body_pos_w'][:, feet, 2]
    speed = np.linalg.norm(root_vel, axis=1)
    candidates = []
    for start in range(20, len(q)-25):
        for end in range(start+25, min(start+100, len(q)-1)):
            if min(speed[start], speed[end]) < minimum_speed or np.min(speed[start:end+1]) < .7*minimum_speed:
                continue
            qerr = np.sqrt(np.mean((q[start]-q[end])**2))
            verr = np.sqrt(np.mean((v[start]-v[end])**2))
            ferr = np.max(np.abs(foot_height[start]-foot_height[end]))
            rerr = np.linalg.norm(root_vel[start]-root_vel[end])
            candidates.append((float(qerr+.05*verr+2*ferr+.1*rerr), start, end))
    if not candidates:
        raise ValueError('No sustained cycle in '+str(path))
    _, start, end = min(candidates)
    return start, end


class SkillTracker:
    def __init__(self, env, dataset, cfg, loop_profile='baseline'):
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
        if loop_profile not in ('baseline', 'sustained'):
            raise ValueError('Unknown loop profile')
        loops = {n: choose_cycle(dataset/(n+'.npz'), minimum) for n, minimum in [('walk', .22), ('jog', .5)]} if loop_profile == 'sustained' else {}
        self.scheduler = SkillPhase({n: clips[n] for n in names}, {n: clips[n+'_stop'] for n in names}, brake, NJ, horizon=cfg['env']['future_frames'], loops=loops)
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
