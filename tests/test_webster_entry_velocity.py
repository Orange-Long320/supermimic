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
