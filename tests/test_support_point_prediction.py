"""Verify the foot-floor predictor independently against FK at accelerated states."""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from supermimic.data.motion_convert import load_g1_model
from supermimic.policy.native_inverse import support_point_height_kinematics


def test_point_acceleration_includes_rotational_and_centripetal_terms():
    m=load_g1_model();d=mujoco.MjData(m);q=m.qpos0.copy()
    q[3:7]=Rotation.from_rotvec([.3,-.2,.1]).as_quat()[[3,0,1,2]]
    rng=np.random.default_rng(81);v=rng.normal(0,.8,m.nv);a=rng.normal(0,2,m.nv)
    body=m.body('left_ankle_roll_link').id;points=np.array([[.13,.04,-.03],[-.09,-.05,-.02]])
    def read(t):
        d.qpos[:]=q;mujoco.mj_integratePos(m,d.qpos,v*t+.5*a*t*t,1.)
        d.qvel[:]=v+a*t;mujoco.mj_forward(m,d)
        jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jacBody(m,d,jp,jr,body)
        h=d.xpos[body,2]+(points@d.xmat[body].reshape(3,3).T)[:,2]
        return h,np.r_[jp,jr]
    h0,jac=read(0.);hplus,jplus=read(1e-5);hminus,jminus=read(-1e-5)
    drift=(jplus-jminus)@v/(2e-5);read(0.)
    height,point_jac,point_drift=support_point_height_kinematics(d,body,points,jac,drift,v)
    np.testing.assert_allclose(height,h0,atol=1e-12)
    np.testing.assert_allclose(point_jac@v,(hplus-hminus)/(2e-5),atol=1e-8)
    np.testing.assert_allclose(point_jac@a+point_drift,(hplus-2*h0+hminus)/1e-10,atol=3e-5)
