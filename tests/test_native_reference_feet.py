"""A soft recovery target must not invent support while both feet are airborne."""
import mujoco
import numpy as np

from supermimic.policy.native_inverse import NativeContactController


def test_airborne_foot_recovery_preserves_free_fall_and_motor_caps():
    model=mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep="0.005" integrator="implicitfast"/>
      <worldbody>
        <geom type="plane" size="2 2 .1"/>
        <body name="pelvis" pos="0 0 .4999">
          <freejoint/>
          <geom type="sphere" size=".1" mass="10" contype="0" conaffinity="0"/>
          <body name="left" pos="0 .2 -.4">
            <joint name="left_slide" type="slide" axis="0 0 1" range="-.3 .3"/>
            <geom type="sphere" size=".1" mass="1"/>
          </body>
          <body name="right" pos="0 -.2 -.4">
            <joint name="right_slide" type="slide" axis="0 0 1" range="-.3 .3"/>
            <geom type="sphere" size=".1" mass="1"/>
          </body>
        </body>
      </worldbody>
      <actuator>
        <position joint="left_slide" kp="100" kv="10" forcerange="-20 20"/>
        <position joint="right_slide" kp="100" kv="10" forcerange="-20 20"/>
      </actuator>
    </mujoco>''')
    data=mujoco.MjData(model)
    controller=NativeContactController(model,data.qpos.copy(),np.full(2,20.),np.full(2,5.),model.body('left').id)
    controller.integration_aware=True
    controller.reference_foot_targets_valid=True
    controller.stabilize_support_normal=True
    controller.foot_task_weight=1.
    data.qpos[2]+=.2
    mujoco.mj_forward(model,data)
    assert data.ncon==0
    torque,report=controller.solve(data.qpos.copy(),data.qvel.copy())
    assert report['reference_foot_tracking_without_contact']
    assert not report['actual_primary_foot_contact']
    assert np.max(np.abs(torque))<=20.0002
    data.ctrl[:]=data.qpos[7:]+torque/100.
    mujoco.mj_forward(model,data)
    jac=np.zeros((3,model.nv))
    mujoco.mj_jacSubtreeCom(model,data,jac,model.body('pelvis').id)
    np.testing.assert_allclose(jac@data.qacc,model.opt.gravity,atol=1e-6)
    np.testing.assert_array_equal(data.qfrc_applied,0.)
    np.testing.assert_array_equal(data.xfrc_applied,0.)
