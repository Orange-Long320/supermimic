"""Incoming velocity experiments preserve momentum and planted contacts."""
import mujoco
import numpy as np
import pytest
from scripts.data.optimize_webster_policy_cpu import entry_velocity_state
from supermimic.data.motion_convert import load_g1_model
from supermimic.robots.dimensions import PROFILE


def test_entry_velocity_has_no_added_spin_or_support_slip():
    m=load_g1_model();d=mujoco.MjData(m);q=m.qpos0.copy()
    d.qpos[:]=q;mujoco.mj_forward(m,d)
    low=[]
    for g in range(m.ngeom):
        if 'foot_' not in m.geom(g).name:continue
        k=m.geom_dataid[g];a,n=m.mesh_vertadr[k],m.mesh_vertnum[k]
        low.append((m.mesh_vert[a:a+n]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])[:,2].min())
    q[2]+=.005-min(low)
    limits=np.array([PROFILE['velocity_limit'][n] for n in PROFILE['joint_names']])
    velocity,report=entry_velocity_state(m,q,np.zeros(m.nv),.5,limits)
    assert len(report['stationary_support_bodies'])>=1
    np.testing.assert_allclose(report['initial_com_velocity'],[.5,0,0],atol=1e-9)
    np.testing.assert_allclose(report['initial_momentum'],0,atol=1e-9)
    assert report['maximum_constraint_error']<1e-9
    assert report['maximum_joint_speed_ratio']<=1
    assert report['initial_kinetic_energy_j']>0
    q[2]+=1
    with pytest.raises(ValueError,match='planted support'):
        entry_velocity_state(m,q,velocity,.5,limits)


def test_full_entry_target_matches_descent_momentum_and_stationary_feet():
    m=load_g1_model();d=mujoco.MjData(m);q=m.qpos0.copy()
    for side in ['left','right']:
        for suffix,value in [('hip_pitch',-.6),('knee',1.2),('ankle_pitch',-.6)]:
            q[m.jnt_qposadr[m.joint(side+'_'+suffix+'_joint').id]]=value
    d.qpos[:]=q;mujoco.mj_forward(m,d);heights=[]
    for g in range(m.ngeom):
        if 'foot_' not in m.geom(g).name:continue
        k=m.geom_dataid[g];a,n=m.mesh_vertadr[k],m.mesh_vertnum[k]
        heights.append((m.mesh_vert[a:a+n]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])[:,2].min())
    q[2]+=.005-min(heights)
    limits=np.array([PROFILE['velocity_limit'][n] for n in PROFILE['joint_names']])
    velocity,report=entry_velocity_state(m,q,np.zeros(m.nv),1.5,limits,
        com_velocity=[1.5,0,-1.2],angular_momentum=[0,8.86,0])
    np.testing.assert_allclose(report['initial_com_velocity'],[1.5,0,-1.2],atol=1e-9)
    np.testing.assert_allclose(report['initial_momentum'],[0,8.86,0],atol=1e-9)
    assert report['maximum_joint_speed_ratio']<=1
    d.qpos[:]=q;d.qvel[:]=velocity;mujoco.mj_forward(m,d)
    for name in report['stationary_support_bodies']:
        jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jacBody(m,d,jp,jr,m.body(name).id)
        np.testing.assert_allclose(np.r_[jp,jr]@velocity,0,atol=1e-9)
    with pytest.raises(ValueError,match='Conflicting'):
        entry_velocity_state(m,q,np.zeros(m.nv),.5,limits,com_velocity=[1.,0,0])
