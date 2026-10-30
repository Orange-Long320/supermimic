"""Reference tracking goal for a single Webster clip, for isolated simulation only."""
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from supermimic.envs.mdp.motion_library import MotionLibrary

class WebsterGoal:
    def __init__(self, reference, profile):
        import json
        spec = json.loads(Path(profile).read_text())
        self.names = tuple(spec['joint_names'])
        self.default = np.array([spec['default_joint_pos'][n] for n in self.names], dtype=np.float32)
        with np.load(reference, allow_pickle=False) as a:
            if tuple(a['joint_names']) != self.names: raise ValueError('joint order mismatch')
            if abs(float(a['fps']) - 50.) > 1e-5: raise ValueError('expected 50Hz reference')
            anchor = list(a['body_names']).index(spec['anchor_body'])
            root = list(a['body_names']).index('pelvis')
            self.target_pos = a['body_pos_w'][:, anchor].copy()
            self.target_quat = a['body_quat_w'][:, anchor].copy()
            self.root0 = a['body_pos_w'][0, root].copy()
            self.initial_joints = a['joint_pos'][0].copy()
        lib = MotionLibrary([reference], tuple(spec['tracked_bodies']), self.default, device='cpu')
        self.features = lib.goal_feat.numpy().copy()
        xml = Path(spec['xml_path'])
        if not xml.is_absolute(): xml = Path(profile).parent / xml
        self.model = mujoco.MjModel.from_xml_path(str(xml)); self.data = mujoco.MjData(self.model)
        self.anchor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, spec['anchor_body'])
        self.addresses = []
        for name in self.names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0: raise ValueError('missing joint '+name)
            self.addresses.append(int(self.model.jnt_qposadr[jid]))
        if self.anchor < 0 or len(set(self.addresses)) != len(self.names): raise ValueError('invalid FK')
        self.frame = 0; self.offset = None

    @staticmethod
    def rotation(q):
        return Rotation.from_quat(np.asarray(q)[[1,2,3,0]])

    def goal(self, pos, quat, joints):
        if self.offset is None:
            self.offset = np.array([pos[0]-self.root0[0], pos[1]-self.root0[1], 0.])
        i = min(self.frame, len(self.features)-1)
        idx = np.minimum(np.arange(i, i+10), len(self.features)-1)
        self.data.qpos[:7] = np.r_[pos, quat]; self.data.qpos[self.addresses] = joints
        mujoco.mj_kinematics(self.model, self.data)
        actual_pos = self.data.xpos[self.anchor].copy()
        actual_rot = self.rotation(self.data.xquat[self.anchor])
        ref_pos = self.target_pos[i] + self.offset
        ref_rot = self.rotation(self.target_quat[i])
        delta_pos = actual_rot.inv().apply(ref_pos - actual_pos)
        delta_rot = (actual_rot.inv()*ref_rot).as_matrix()[:, :2].reshape(-1)
        return np.r_[self.features[idx].reshape(-1), delta_pos, delta_rot].astype(np.float32)

    def advance(self): self.frame += 1
