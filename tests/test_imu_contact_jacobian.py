"""Check the relative contact-point velocity used by the offline estimator."""
import mujoco
import numpy as np


def test_contact_point_jacobian_matches_motion():
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <worldbody><body name="base"><freejoint/>
        <geom type="sphere" size=".1" mass="1"/>
        <body name="foot" pos="0 0 -.4"><joint name="ankle" axis="0 1 0"/>
          <geom type="sphere" size=".1" mass="1"/>
        </body></body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    data.qpos[3:7] = np.array([.98, .05, -.1, .15])/np.linalg.norm([.98, .05, -.1, .15])
    data.qpos[-1] = .25
    data.qvel[:] = [0., 0., 0., .4, -.3, .2, .7]
    mujoco.mj_kinematics(model, data); mujoco.mj_comPos(model, data)
    body = model.body('foot').id; local = np.array([.06, .01, -.09])
    point = data.xpos[body]+data.xmat[body].reshape(3,3)@local
    jac = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jac, None, point, body)
    expected = jac@data.qvel
    eps = 1e-6
    qpos = data.qpos.copy()
    mujoco.mj_integratePos(model, qpos, data.qvel, eps)
    data.qpos[:] = qpos; mujoco.mj_kinematics(model, data)
    next_point = data.xpos[body]+data.xmat[body].reshape(3,3)@local
    np.testing.assert_allclose((next_point-point)/eps, expected, atol=2e-6)
