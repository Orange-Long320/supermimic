"""A rigid rotation about a fixed foot must not invent initial foot slip."""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from supermimic.data.motion_convert import load_g1_model
from supermimic.data.physics_motion import differentiate_qpos,export_qpos_motion


def rotating_poses():
    m=load_g1_model();d=mujoco.MjData(m);q=m.qpos0.copy()
    d.qpos[:]=q;mujoco.mj_forward(m,d)
    foot=m.body('left_ankle_roll_link').id
    anchor=d.xpos[foot].copy()
    t=np.arange(17)*.01;rotvec=np.zeros((len(t),3));rotvec[:,1]=5*t*t
    r=Rotation.from_rotvec(rotvec)
    poses=np.tile(q,(len(t),1))
    poses[:,:3]=anchor+r.apply(np.tile(q[:3]-anchor,(len(t),1)))
    poses[:,3:7]=(r*Rotation.from_quat(q[[4,5,6,3]])).as_quat()[:,[3,0,1,2]]
    omega=np.zeros((len(t),3));omega[:,1]=10*t
    return m,poses,anchor,foot,omega


def test_accelerating_rotation_preserves_endpoint_velocity_and_fixed_point():
    m,q,anchor,foot,omega=rotating_poses()
    v=differentiate_qpos(m,q,100.)
    r=Rotation.from_quat(q[:,[4,5,6,3]])
    np.testing.assert_allclose(r.apply(v[:,3:6]),omega,atol=1e-9)
    d=mujoco.MjData(m);jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv))
    speeds=[]
    for row,velocity in zip(q,v):
        d.qpos[:]=row;mujoco.mj_forward(m,d)
        np.testing.assert_allclose(d.xpos[foot],anchor,atol=1e-9)
        mujoco.mj_jac(m,d,jp,jr,d.xpos[foot],foot)
        speeds.append(np.linalg.norm(jp@velocity))
    assert speeds[0]<.001
    assert max(speeds)<.003


def test_export_preserves_supplied_physical_velocity(tmp_path):
    m,q,anchor,foot,omega=rotating_poses()
    v=np.zeros((len(q),m.nv));r=Rotation.from_quat(q[:,[4,5,6,3]])
    v[:,:3]=np.cross(omega,q[:,:3]-anchor)
    v[:,3:6]=r.inv().apply(omega)
    path=tmp_path/'actual-state.npz'
    export_qpos_motion(m,q,path,100.,qvel=v,source='analytic rigid rotation')
    with np.load(path) as z:
        root=z['body_names'].tolist().index(m.body(int(m.jnt_bodyid[0])).name)
        f=z['body_names'].tolist().index(m.body(foot).name)
        np.testing.assert_allclose(z['body_lin_vel_w'][:,root],v[:,:3],atol=1e-6)
        np.testing.assert_allclose(z['body_lin_vel_w'][:,f],0.,atol=1e-6)
        assert str(z['velocity_source'])=='provided_generalized_velocity'
