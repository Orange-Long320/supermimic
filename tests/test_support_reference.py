"""Independent FK checks for the coordinated support reference."""
import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from supermimic.data.motion_convert import load_g1_model
from supermimic.kinematics.support_reference import SupportPoseIK,SupportReference,root_rate_for_momentum


def test_ik_keeps_the_foot_planted_while_raising_the_whole_body():
    m=load_g1_model();d=mujoco.MjData(m);q=m.qpos0.copy()
    for side in ['left','right']:
        q[m.jnt_qposadr[m.joint(side+'_elbow_joint').id]]=-.6
        for suffix,value in [('hip_pitch',-.6),('knee',1.2),('ankle_pitch',-.6)]:
            q[m.jnt_qposadr[m.joint(side+'_'+suffix+'_joint').id]]=value
    foot=m.body('left_ankle_roll_link').id;root=int(m.jnt_bodyid[0]);d.qpos[:]=q;mujoco.mj_forward(m,d)
    position=d.xpos[foot].copy();rotation=Rotation.from_matrix(d.xmat[foot].reshape(3,3))
    target=d.subtree_com[root].copy()+[.01,0,.02];root_rotation=Rotation.from_quat(q[3:7][[1,2,3,0]])
    ik=SupportPoseIK(m,q,foot);pose,report=ik.fit(q,target,position,rotation,root_rotation)
    d.qpos[:]=pose;mujoco.mj_forward(m,d)
    np.testing.assert_allclose(d.xpos[foot],position,atol=1e-9)
    np.testing.assert_allclose(d.xmat[foot].reshape(3,3),rotation.as_matrix(),atol=1e-9)
    np.testing.assert_allclose(d.subtree_com[root],target,atol=1e-5)
    assert report['root_orientation_error_rad']<1e-5
    lo,hi=m.jnt_range[1:].T
    assert np.all(pose[7:]>=lo+.025-1e-9) and np.all(pose[7:]<=hi-.025+1e-9)


def make_reference(path,times):
    n=len(times);q=np.zeros((n,38));q[:,3]=1;q[:,0]=np.asarray(times)*2
    arrays={'time':np.asarray(times),'qpos':q,'qvel':np.ones((n,37))*2,'qacc':np.ones((n,37)),
        'foot_quat':np.tile([0,0,0,1.],(n,1))}
    for key in ['com','com_velocity','com_acceleration','momentum','momentum_rate','foot_position']:
        arrays[key]=np.ones((n,3))*2
    for key in ['foot_velocity','foot_acceleration']:arrays[key]=np.ones((n,6))*2
    np.savez(path,**arrays)


def test_endpoint_momentum_preserves_translation_and_the_planted_foot():
    m=load_g1_model();d=mujoco.MjData(m);q=m.qpos0.copy()
    q[3:7]=Rotation.from_rotvec([.15,.3,-.2]).as_quat()[[3,0,1,2]]
    for side in ['left','right']:
        q[m.jnt_qposadr[m.joint(side+'_knee_joint').id]]=1.2
    joint_rates=np.zeros(m.nv-6)
    joint_rates[m.jnt_dofadr[m.joint('right_hip_pitch_joint').id]-6]=3.
    foot=m.body('left_ankle_roll_link').id;translation=np.array([.5,0,.3]);momentum=np.array([0,12.,0])
    rates,velocity,residual=root_rate_for_momentum(m,q,foot,joint_rates,translation,np.zeros(6),np.array([.2,.4,-.1]),momentum)
    assert np.isfinite(rates).all() and residual<1e-9
    d.qpos[:]=q;d.qvel[:]=velocity;mujoco.mj_forward(m,d)
    com=np.zeros((3,m.nv));jp=np.zeros_like(com);jr=np.zeros_like(com);angular=np.zeros_like(com)
    mujoco.mj_jacSubtreeCom(m,d,com,int(m.jnt_bodyid[0]));mujoco.mj_jacBody(m,d,jp,jr,foot)
    mujoco.mj_angmomMat(m,d,angular,int(m.jnt_bodyid[0]))
    np.testing.assert_allclose(com@velocity,translation,atol=1e-9)
    np.testing.assert_allclose(np.r_[jp,jr]@velocity,0,atol=1e-9)
    np.testing.assert_allclose(angular@velocity,momentum,atol=1e-9)


def test_sampler_preserves_the_launch_velocity_until_the_terminal_boundary(tmp_path):
    path=tmp_path/'ref.npz';make_reference(path,[0.,.1,.2]);ref=SupportReference(path)
    sample=ref.sample(.15)
    assert sample['qpos'][0]==pytest.approx(.3)
    assert np.linalg.norm(sample['qpos'][3:7])==pytest.approx(1.)
    assert (ref.sample(.2)['qvel']==2).all()
    held=ref.sample(.3)
    assert held['qpos'][0]==pytest.approx(.4)
    assert (held['qvel']==0).all() and (held['momentum_rate']==0).all()


def test_reference_rejects_a_nonincreasing_clock(tmp_path):
    path=tmp_path/'bad.npz';make_reference(path,[0.,.1,.1])
    with pytest.raises(ValueError,match='time must increase'):SupportReference(path)
