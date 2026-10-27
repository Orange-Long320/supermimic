"""Check integrated acceleration against the actual implicitfast step."""
import mujoco
import numpy as np
from supermimic.policy.native_inverse import implicitfast_acceleration_map


def test_unsaturated_position_servo_map_predicts_the_actual_velocity_update():
    m=mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".005" integrator="implicitfast"/>
    <worldbody><body pos="0 0 2"><freejoint/><geom type="sphere" size=".1" mass="2"/>
    <body pos=".2 0 0"><joint name="hinge" type="hinge" axis="0 1 0" damping=".03"/>
    <geom type="capsule" fromto="0 0 0 .3 0 0" size=".04" mass=".5"/></body></body></worldbody>
    <actuator><position joint="hinge" kp="20" kv="1.2" forcerange="-30 30"/></actuator></mujoco>''')
    d=mujoco.MjData(m);d.qpos[-1]=.2;d.qvel[:]=np.linspace(-.3,.4,m.nv);d.ctrl[:]=.6
    mujoco.mj_forward(m,d);v=d.qvel.copy();instantaneous=d.qacc.copy()
    integrated=np.linalg.solve(implicitfast_acceleration_map(m,d),instantaneous)
    assert np.max(abs(integrated-instantaneous))>1.
    mujoco.mj_step(m,d)
    np.testing.assert_allclose((d.qvel-v)/m.opt.timestep,integrated,atol=1e-9)
