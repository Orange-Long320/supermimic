"""Reference and anchor computation without a simulation environment.

The caller supplies an estimated pelvis pose and named joint positions. The
model is used only for kinematics; it is not a physics simulation.
"""
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from supermimic.control.skill_phase import SkillPhase
from supermimic.control.skill_tracker import choose_cycle
from supermimic.envs.mdp.motion_library import MotionLibrary


class SkillGoal:
    def __init__(self, dataset, profile, root_position, root_quat_wxyz,
                 *, sustained=True, straight=True, horizon=10, jog_brake_fade=None, stand_anchor_gain=1.,
                 stand_gain_quiet_time=0., stand_gain_speed_limit=.08):
        dataset, profile = Path(dataset), Path(profile)
        spec = json.loads(profile.read_text())
        self.names = tuple(spec['joint_names'])
        self.nj = len(self.names)
        self.default = np.array([spec['default_joint_pos'][n] for n in self.names], dtype=np.float32)
        xml = Path(spec['xml_path'])
        self.model = mujoco.MjModel.from_xml_path(str(xml if xml.is_absolute() else profile.parent / xml))
        self.actual = mujoco.MjData(self.model)
        self.reference = mujoco.MjData(self.model)
        self.anchor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, spec['anchor_body'])
        self.qaddr = []
        for name in self.names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError('Missing model joint: ' + name)
            self.qaddr.append(int(self.model.jnt_qposadr[jid]))
        if self.anchor < 0 or len(set(self.qaddr)) != self.nj:
            raise ValueError('Invalid model anchor or joint mapping')
        names = ['walk', 'backward', 'turn_left_inplace', 'turn_right_inplace', 'jog']
        paths = [dataset/(n+'.npz') for n in names] + [dataset/(n+'_stop.npz') for n in names]
        meta = json.loads((dataset/'manifest.json').read_text())
        brake = {c['name'].removesuffix('_stop'): c['release_frame']+10
                 for c in meta['clips'] if c['type'] == 'exploratory_stop'}
        lib = MotionLibrary(paths, tuple(spec['tracked_bodies']), self.default, device='cpu')
        clips = {c.name: lib.goal_feat[c.start:c.end].numpy().copy() for c in lib.clips}
        if straight:
            for name in ('walk', 'jog'):
                clips[name][:, 2*self.nj+5] = 0.
                clips[name+'_stop'][:, 2*self.nj+5] = 0.
        loops = {n: choose_cycle(dataset/(n+'.npz'), speed) for n, speed in [('walk', .22), ('jog', .5)]} if sustained else {}
        self.scheduler = SkillPhase({n: clips[n] for n in names}, {n: clips[n+'_stop'] for n in names}, brake, self.nj, horizon=horizon, loops=loops, jog_brake_fade=jog_brake_fade)
        self.position = np.asarray(root_position, dtype=float).copy()
        self.yaw = self.rotation(root_quat_wxyz).as_euler('xyz')[2]
        if not 1. <= stand_anchor_gain <= 2.:
            raise ValueError('stand anchor gain must be between 1 and 2')
        self.stand_anchor_gain = stand_anchor_gain
        self.stand_gain = 1.
        if not 0. <= stand_gain_quiet_time <= 5. or not 0. < stand_gain_speed_limit <= .5:
            raise ValueError('invalid stand gain quiet gate')
        self.stand_gain_quiet_time = stand_gain_quiet_time
        self.stand_gain_speed_limit = stand_gain_speed_limit
        self.stand_quiet_steps = 0
        self.last_position = None

    @staticmethod
    def rotation(wxyz):
        w, x, y, z = wxyz
        return Rotation.from_quat([x, y, z, w])

    def fk(self, data, pos, quat, joints):
        data.qpos[:7] = np.r_[pos, quat]
        data.qpos[self.qaddr] = joints
        mujoco.mj_kinematics(self.model, data)
        return data.xpos[self.anchor].copy(), self.rotation(data.xquat[self.anchor])

    def goal(self, position, quat_wxyz, joints, mode):
        self.scheduler.select(mode)
        window = self.scheduler.window()
        g = window[0]
        grav = g[2*self.nj:2*self.nj+3]
        roll = np.arctan2(-grav[1], -grav[2])
        pitch = np.arcsin(np.clip(grav[0], -1., 1.))
        quat = Rotation.from_euler('xyz', [roll, pitch, self.yaw]).as_quat()[[3, 0, 1, 2]]
        target_pos, target_rot = self.fk(self.reference, [*self.position[:2], g[-1]], quat, g[:self.nj]+self.default)
        actual_pos, actual_rot = self.fk(self.actual, position, quat_wxyz, joints)
        delta_pos = actual_rot.inv().apply(target_pos-actual_pos)
        # Experimental feedback scaling once the complete committed window is
        # standing. Ramped so an immediate mode change cannot jump the anchor.
        stable_reference = mode == 'stand' and np.allclose(window, self.scheduler.stand, atol=1e-7)
        position = np.asarray(position, dtype=float)
        speed = np.linalg.norm(position[:2]-self.last_position[:2])/self.scheduler.dt if self.last_position is not None else float('inf')
        self.last_position = position.copy()
        quiet = stable_reference and (self.stand_gain_quiet_time == 0. or speed <= self.stand_gain_speed_limit)
        self.stand_quiet_steps = self.stand_quiet_steps+1 if quiet else 0
        target_gain = self.stand_anchor_gain if quiet and self.stand_quiet_steps*self.scheduler.dt >= self.stand_gain_quiet_time else 1.
        self.stand_gain += np.clip(target_gain-self.stand_gain, -.02, .02)
        delta_pos[:2] *= self.stand_gain
        delta_rot = (actual_rot.inv()*target_rot).as_matrix()[:, :2].reshape(-1)
        return np.r_[window.reshape(-1), delta_pos, delta_rot].astype(np.float32)

    def advance(self):
        g = self.scheduler.window()[0]
        velocity = g[2*self.nj+6:2*self.nj+9]
        self.position[:2] += Rotation.from_euler('z', self.yaw).apply(velocity)[:2]*self.scheduler.dt
        self.yaw += float(g[2*self.nj+5])*self.scheduler.dt
        self.scheduler.advance()
