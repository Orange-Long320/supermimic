import numpy as np
import pytest
from supermimic.deploy.velocity_observation import DeclaredRestGyroBias, VelocityObservationCorrection, FootVelocityObservation


def test_bias_requires_external_rest_and_never_treats_quiet_gyro_as_rest():
    b=DeclaredRestGyroBias()
    for i in range(2001):assert b.update(i*.002,[.002,.001,0]) is None
    assert b.count==0
    for i in range(2001):value=b.update(i*.002,[.002,.001,0],stationary_declared=True)
    np.testing.assert_allclose(value,[.002,.001,0])
    assert b.update(4.002,[.002,.001,0],stationary_declared=False) is None
    assert b.count==0


def test_no_observation_exactly_preserves_native_trajectory():
    c=VelocityObservationCorrection()
    for t in np.arange(0,10,.02):
        x=np.array([t*t,.3*t,2*t,.3])
        np.testing.assert_allclose(c.update(t,x),x,atol=1e-12)
    assert not c.observation_fresh()


def test_velocity_observation_limits_drift_without_position_reset():
    c=VelocityObservationCorrection()
    for t in np.arange(0,60,.02):
        x=c.update(t,[3+.015*t*t,-2,.03*t,0],[0,0])
    assert abs(x[0]-3)<.05 and x[1]==-2
    assert c.observation_fresh()
    c.update(60,[3+.015*60**2,-2,.03*60,0])
    for t in np.arange(60.02,60.3,.02):c.update(t,[3+.015*t*t,-2,.03*t,0])
    assert not c.observation_fresh()


def test_nonzero_velocity_observation_is_not_zero_velocity_update():
    c=VelocityObservationCorrection()
    for t in np.arange(0,10,.02):x=c.update(t,[0,0,0,0],[.4,0])
    assert 3.9<x[0]<4.01
    with pytest.raises(ValueError,match='interval'):c.update(t,[0,0,0,0])


def test_suspension_disables_foot_constraint_and_disagreement_is_rejected(tmp_path):
    # Static FK only: no physics stepping or simulated contact state.
    model=tmp_path/'model.xml'
    model.write_text('''<mujoco><worldbody><body name="pelvis"><freejoint/><geom size=".1" mass="1"/>
      <body name="left_ankle_roll_link" pos="0 .1 -.5"><joint name="left" axis="0 1 0"/><geom size=".05" mass="1"/></body>
      <body name="right_ankle_roll_link" pos="0 -.1 -.5"><joint name="right" axis="0 1 0"/><geom size=".05" mass="1"/></body>
    </body></worldbody></mujoco>''')
    f=FootVelocityObservation(model,['left','right'],[0,0,0])
    args=([1,0,0,0],[0,0,0],[0,0],[0,0])
    for t in [0,.02,.04]:assert f.observe(t,*args)==(None,'support-not-confirmed')
    assert f.observe(.06,*args,support_confirmed=True)[1]=='hold'
    v,reason=f.observe(.12,*args,support_confirmed=True)
    assert reason=='accepted';np.testing.assert_array_equal(v,[0,0])


def test_support_velocity_matches_finite_difference_and_imu_lever_arm(tmp_path):
    import mujoco
    model=tmp_path/'chain.xml'
    model.write_text('''<mujoco><worldbody><body name="pelvis"><freejoint/><geom size=".1" mass="1"/>
      <body pos="0 .1 0"><joint name="left" axis="0 1 0"/><geom size=".05" mass="1"/>
        <body name="left_ankle_roll_link" pos="0 0 -.5"><geom size=".05" mass="1"/></body></body>
      <body name="right_ankle_roll_link" pos="0 -.1 -.4"><joint name="right" axis="0 1 0"/><geom size=".05" mass="1"/></body>
    </body></worldbody></mujoco>''')
    f=FootVelocityObservation(model,['left','right'],[.03,0,.04])
    args=([1,0,0,0],[0,0,.02],[0,0],[.1,0])
    f.observe(0,*args,support_confirmed=True)
    v,reason=f.observe(.06,*args,support_confirmed=True)
    assert reason=='accepted'
    before=f.data.xpos[f.feet[0]].copy();rotation=f.data.xmat[f.root].reshape(3,3).copy()
    q=f.data.qpos.copy();mujoco.mj_integratePos(f.model,q,f.data.qvel,1e-6)
    f.data.qpos[:]=q;mujoco.mj_kinematics(f.model,f.data)
    expected=-(f.data.xpos[f.feet[0]]-before)/1e-6+rotation@np.cross(args[1],f.offset)
    np.testing.assert_allclose(v,expected[:2],atol=1e-7)
