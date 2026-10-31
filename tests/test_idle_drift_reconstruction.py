import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def test_anchor_reconstruction_recovers_known_root_pose(tmp_path):
    path = Path(__file__).parents[1]/'scripts/evaluation/diagnose_idle_drift.py'
    spec = importlib.util.spec_from_file_location('idle_drift_audit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    xml = tmp_path/'robot.xml'
    xml.write_text('''<mujoco><worldbody><body name="pelvis"><freejoint/>
      <geom type="sphere" size=".1" mass="1"/>
      <body name="torso" pos=".03 0 .2"><joint name="waist" axis="0 1 0"/>
      <geom type="sphere" size=".05" mass="1"/></body></body></worldbody></mujoco>''')
    profile = tmp_path/'profile.json'
    profile.write_text(json.dumps(dict(xml_path='robot.xml', anchor_body='torso',
                                      joint_names=['waist'], default_joint_pos={'waist': .02})))
    model = mujoco.MjModel.from_xml_path(str(xml))
    actual, target = mujoco.MjData(model), mujoco.MjData(model)
    ra = Rotation.from_euler('xyz', [.04, -.06, -.4])
    rt = Rotation.from_euler('xyz', [-.02, .05, .3])
    actual.qpos[:] = np.r_[-.2, .4, .65, ra.as_quat()[[3,0,1,2]], -.05]
    target.qpos[:] = np.r_[1.2, -.1, .67, rt.as_quat()[[3,0,1,2]], .12]
    mujoco.mj_kinematics(model, actual)
    mujoco.mj_kinematics(model, target)
    body = model.body('torso').id
    a = Rotation.from_quat(actual.xquat[body][[1,2,3,0]])
    b = Rotation.from_quat(target.xquat[body][[1,2,3,0]])
    anchor = np.r_[a.inv().apply(target.xpos[body]-actual.xpos[body]),
                   (a.inv()*b).as_matrix()[:,:2].reshape(-1)]
    goal = np.r_[.12-.02, 0., rt.inv().apply([0.,0.,-1.]), np.zeros(6), .67]
    row = dict(root=actual.qpos[:3].tolist(), q=actual.qpos[3:7].tolist(),
               joint_pos=[-.05], goal_current=goal.tolist(), anchor=anchor.tolist(), stand_gain=1.)
    pos, quat, error, _ = module.reconstruct([row], profile)
    np.testing.assert_allclose(pos[0], target.qpos[:3], atol=1e-12)
    np.testing.assert_allclose(Rotation.from_quat(quat[0,[1,2,3,0]]).as_matrix(), rt.as_matrix(), atol=1e-12)
    assert error < 1e-12
