"""An actual gyro sensor anchors the native diagnostic's angular frame."""
import importlib.util
from pathlib import Path
import unittest

import mujoco
import numpy as np

path = Path(__file__).resolve().parents[1]/'scripts/evaluation/native_stand_entry_compare.py'
spec = importlib.util.spec_from_file_location('native_compare', path)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class BodyVelocityFrameTests(unittest.TestCase):
    def test_link_velocity_matches_sensor_with_rotated_principal_inertia(self):
        xml = '''<mujoco><worldbody><body name="base" quat="0.9238795 0 0 0.3826834">
          <freejoint/><inertial pos=".03 .02 .01" mass="3" diaginertia=".2 .3 .4"
          quat="0.70710678 0.70710678 0 0"/><site name="imu"/>
          </body></worldbody><sensor><gyro name="gyro" site="imu"/></sensor></mujoco>'''
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base')
        for omega in ([.1, .2, .3], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]):
            data.qvel[3:6] = omega; mujoco.mj_forward(model, data)
            actual = module.body_frame_velocity(model, data, body)[:3]
            np.testing.assert_allclose(actual, data.sensordata[:3], atol=1e-12)
        wrong = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, wrong, 1)
        self.assertGreater(float(np.linalg.norm(wrong[:3]-data.sensordata[:3])), .5)


if __name__ == '__main__': unittest.main()
