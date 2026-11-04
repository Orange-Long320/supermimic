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
                 stand_gain_quiet_time=0., stand_gain_speed_limit=.08, planner='baseline',
                 planner_ablation='none', idle_join_frames=None, idle_entry_offset=None,
                 idle_anchor_mode='fixed', idle_velocity_feedback=False, foot_aligned_scene=None,
                 foot_registration_only=False, anchor_obs=True, zero_horizontal_anchor_modes=()):
        if not anchor_obs and (planner != 'baseline' or idle_anchor_mode != 'fixed' or idle_velocity_feedback
                               or stand_anchor_gain != 1. or stand_gain_quiet_time != 0. or foot_aligned_scene is not None):
            raise ValueError('No-anchor candidate requires the unchanged baseline skill scheduler')
        allowed_zero_modes = {'stand', 'walk', 'backward', 'jog', 'turn_left_inplace', 'turn_right_inplace'}
        if (not isinstance(zero_horizontal_anchor_modes, (tuple, list)) or
                any(m not in allowed_zero_modes for m in zero_horizontal_anchor_modes) or
                len(set(zero_horizontal_anchor_modes)) != len(zero_horizontal_anchor_modes) or
                (zero_horizontal_anchor_modes and (not anchor_obs or idle_velocity_feedback))):
            raise ValueError('Invalid horizontal anchor intervention')
        self.zero_horizontal_anchor_modes = tuple(zero_horizontal_anchor_modes)
        self.anchor_obs = anchor_obs
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
        cycle_specs = [('walk', .22)] if planner == 'context-idle' else [('walk', .22), ('jog', .5)]
        loops = {n: choose_cycle(dataset/(n+'.npz'), speed) for n, speed in cycle_specs} if sustained else {}
        self.scheduler = SkillPhase({n: clips[n] for n in names}, {n: clips[n+'_stop'] for n in names}, brake, self.nj, horizon=horizon, loops=loops, jog_brake_fade=jog_brake_fade)
        if planner not in ('baseline', 'context-idle'):
            raise ValueError('Unknown reference planner')
        if planner_ablation not in ('none', 'baseline-idle', 'baseline-motion'):
            raise ValueError('Unknown planner diagnostic ablation')
        if planner == 'baseline' and (planner_ablation != 'none' or idle_join_frames is not None or idle_entry_offset is not None):
            raise ValueError('Planner diagnostics require the experimental context planner')
        self.planner = planner
        if idle_anchor_mode not in ('fixed', 'zero-xy'):
            raise ValueError('Unknown Idle anchor diagnostic')
        self.idle_anchor_mode = idle_anchor_mode
        self.has_motion_request = False
        if idle_velocity_feedback and idle_anchor_mode != 'fixed':
            raise ValueError('Idle damping requires the original fixed root target')
        self.idle_velocity_feedback = idle_velocity_feedback
        self.idle_root_velocity = np.zeros(3)
        self.idle_lookahead = 0.
        if idle_velocity_feedback:
            from supermimic.control.idle_velocity_feedback import RootVelocityEstimate
            self.root_velocity_estimate = RootVelocityEstimate()
        if planner == 'context-idle':
            from supermimic.control.idle_planner import ContextIdlePlanner
            self.foot_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, side+'_ankle_roll_link')
                             for side in ('left', 'right')]
            if min(self.foot_ids) < 0:
                raise ValueError('Missing planner foot bodies')
            limits = np.asarray([self.model.jnt_range[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                                 for n in self.names])-self.default[:, None]
            velocity = np.asarray([spec['velocity_limit'][n] for n in self.names])
            self.scheduler = ContextIdlePlanner({n: clips[n] for n in names}, {n: clips[n+'_stop'] for n in names},
                brake, self.nj, horizon=horizon, loops=loops, joint_limits=limits, velocity_limits=velocity,
                foot_features=self.planner_feet, baseline_motion=planner_ablation == 'baseline-motion',
                baseline_idle=planner_ablation == 'baseline-idle', idle_join_frames=idle_join_frames,
                idle_entry_offset=idle_entry_offset)
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
        self.foot_idle = None
        if foot_aligned_scene is not None:
            if planner != 'context-idle' or idle_velocity_feedback or idle_anchor_mode != 'fixed' or stand_anchor_gain != 1. or planner_ablation != 'none':
                raise ValueError('Foot-aligned Idle requires the unmodified context planner and anchor')
            from supermimic.deploy.foot_idle_session import FootIdleSession
            self.foot_idle = FootIdleSession(foot_aligned_scene,spec,self.scheduler,foot_registration_only)
        elif foot_registration_only:
            raise ValueError('Registration diagnostic requires the foot planner scene')

    @staticmethod
    def rotation(wxyz):
        w, x, y, z = wxyz
        return Rotation.from_quat([x, y, z, w])

    def fk(self, data, pos, quat, joints):
        data.qpos[:7] = np.r_[pos, quat]
        data.qpos[self.qaddr] = joints
        mujoco.mj_kinematics(self.model, data)
        return data.xpos[self.anchor].copy(), self.rotation(data.xquat[self.anchor])

    def planner_feet(self, row):
        """Heading-frame ankle geometry; this does not measure support."""
        grav = row[2*self.nj:2*self.nj+3]
        quat = Rotation.from_euler('xyz', [np.arctan2(-grav[1], -grav[2]),
                                           np.arcsin(np.clip(grav[0], -1., 1.)), 0.]).as_quat()[[3,0,1,2]]
        self.fk(self.reference, [0., 0., row[-1]], quat, row[:self.nj]+self.default)
        return self.reference.xpos[self.foot_ids].copy()

    def goal(self, position, quat_wxyz, joints, mode, joint_velocity=None, sample_time=None, base_angular_velocity=None):
        self.has_motion_request = self.has_motion_request or mode != 'stand'
        if self.idle_velocity_feedback:
            if sample_time is None:
                raise ValueError('Idle damping requires timestamped measured odometry')
            self.idle_root_velocity = self.root_velocity_estimate.update(sample_time, position)
        if self.planner == 'context-idle':
            if joint_velocity is None:
                raise ValueError('Context planner requires measured joint velocity')
            self.scheduler.set_context(np.asarray(joints)-self.default, joint_velocity)
        if self.foot_idle is not None:
            self.foot_idle.prepare(self,position,quat_wxyz,joints,joint_velocity,base_angular_velocity,sample_time,mode)
        if (self.anchor_obs and mode in ('turn_left_inplace', 'turn_right_inplace')
                and mode != self.scheduler.requested):
            # A new in-place turn holds its entry location, rather than chasing
            # translation error accumulated by earlier walking. Register once:
            # following measured XY every tick would remove position feedback.
            # Keep heading and the committed transition window unchanged.
            self.position[:2] = np.asarray(position, dtype=float)[:2]
        self.scheduler.select(mode)
        window = self.scheduler.window()
        if not self.anchor_obs:
            return window.reshape(-1).astype(np.float32)
        g = window[0]
        grav = g[2*self.nj:2*self.nj+3]
        roll = np.arctan2(-grav[1], -grav[2])
        pitch = np.arcsin(np.clip(grav[0], -1., 1.))
        quat = Rotation.from_euler('xyz', [roll, pitch, self.yaw]).as_quat()[[3, 0, 1, 2]]
        target_pos, target_rot = self.fk(self.reference, [*self.position[:2], g[-1]], quat, g[:self.nj]+self.default)
        actual_pos, actual_rot = self.fk(self.actual, position, quat_wxyz, joints)
        stable_reference = mode == 'stand' and np.allclose(window, self.scheduler.stand, atol=1e-7)
        if self.idle_anchor_mode == 'zero-xy' and stable_reference and self.has_motion_request:
            # Causal diagnostic only: remove the global horizontal root error
            # while preserving pose, height, heading and all future features.
            # This follows the actual root and cannot qualify as position hold.
            target_pos[:2] += np.asarray(position)[:2]-self.position[:2]
        world_delta = target_pos-actual_pos
        self.raw_world_anchor_translation = world_delta.copy()
        if mode in getattr(self, 'zero_horizontal_anchor_modes', ()):
            # Remove world-horizontal feedback before rotating into the anchor
            # frame: a tilted body otherwise leaks XY drift into local Z.
            world_delta[:2] = 0.
        delta_pos = actual_rot.inv().apply(world_delta)
        self.raw_anchor_translation = delta_pos.copy()
        self.idle_lookahead = self.scheduler.horizon*self.scheduler.dt if (
            self.idle_velocity_feedback and stable_reference and self.has_motion_request) else 0.
        if self.idle_lookahead:
            # Feed predicted position error over the existing committed horizon.
            # The world target, joint trajectory and reference velocities stay
            # fixed; at zero measured speed this reduces to original feedback.
            delta_pos -= actual_rot.inv().apply(self.idle_root_velocity*self.idle_lookahead)
        # Experimental feedback scaling once the complete committed window is
        # standing. Ramped so an immediate mode change cannot jump the anchor.
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
        if not self.anchor_obs:
            self.scheduler.advance()
            return
        if self.foot_idle is not None and self.foot_idle.advance(self):
            return
        g = self.scheduler.window()[0]
        velocity = g[2*self.nj+6:2*self.nj+9]
        self.position[:2] += Rotation.from_euler('z', self.yaw).apply(velocity)[:2]*self.scheduler.dt
        self.yaw += float(g[2*self.nj+5])*self.scheduler.dt
        self.scheduler.advance()

    def close(self):
        if self.foot_idle is not None:
            self.foot_idle.close()
