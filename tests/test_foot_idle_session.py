from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from supermimic.control.foot_aligned_idle import FootPlan, polygon_margin, project_inside
from supermimic.control.skill_phase import SkillPhase
from supermimic.deploy.foot_idle_session import FootIdleSession


def fixture():
    g = np.zeros(72,dtype=np.float32)
    g[64], g[-1] = -1., .65
    clip = np.repeat(g[None],32,axis=0)
    scheduler = SkillPhase({'walk':clip},{'walk':clip},{'walk':0},31,horizon=3)
    session = FootIdleSession.__new__(FootIdleSession)
    session.scheduler = scheduler
    session.poses = None
    session.events = []
    session.planner = SimpleNamespace(ready=lambda context:True)
    tracker = SimpleNamespace(position=np.array([4.,5.,.65]),yaw=.2,nj=31,scheduler=scheduler)
    context = dict(feet=np.array([[0.,.1,.073],[0.,-.1,.073]]),foot_rotation=np.repeat(np.eye(3)[None],2,axis=0))
    features = np.repeat(g[None],6,axis=0)
    features[:,0] = np.linspace(0,.1,6)
    positions = np.c_[np.linspace(0,.03,6),np.zeros(6),np.full(6,.65)]
    plan = FootPlan(features,positions,np.zeros(6),context,dict(support_side='left',measured_foot_targets=context['feet'].tolist()))
    return session,tracker,plan,context


def test_published_window_preserved_and_terminal_world_target_stays_fixed():
    session,tracker,plan,context = fixture()
    published = tracker.scheduler.window().copy()
    session.install(tracker,plan,context)
    np.testing.assert_array_equal(tracker.scheduler.window(),published)
    for _ in range(40):
        before = tracker.scheduler.window()
        session.advance(tracker)
        np.testing.assert_array_equal(before[1:],tracker.scheduler.window()[:-1])
    np.testing.assert_allclose(tracker.position,plan.root_positions[-1])
    np.testing.assert_allclose(tracker.scheduler.window(),np.repeat(plan.features[-1][None],3,axis=0))
    terminal = tracker.position.copy()
    for _ in range(1000):
        session.advance(tracker)
    np.testing.assert_array_equal(tracker.position,terminal)


def test_stale_relative_foot_motion_is_rejected_without_mutating_window():
    session,tracker,plan,context = fixture()
    measured = dict(context,feet=context['feet'].copy())
    measured['feet'][1,0] += .02
    before = tracker.scheduler.window()
    with pytest.raises(ValueError,match='stale'):
        session.install(tracker,plan,measured)
    np.testing.assert_array_equal(before,tracker.scheduler.window())
    assert session.poses is None


def test_contact_hull_projection_and_degenerate_support():
    square = np.array([[-.1,-.05],[.1,-.05],[.1,.05],[-.1,.05]])
    inside = project_inside(square,[.3,.2],margin=.01)
    assert polygon_margin(square,inside) >= .01-1e-8
    assert polygon_margin([[0,0]], [0,0]) == -np.inf


def test_registration_diagnostic_preserves_original_joint_and_velocity_goals():
    session,tracker,plan,context = fixture()
    session.registration_only = True
    session.original_stand = tracker.scheduler.stand.copy()
    plan.report['total_planning_seconds'] = 0.
    original = tracker.scheduler.window().copy()
    session.install(tracker,plan,context)
    for _ in range(20):
        session.advance(tracker)
        np.testing.assert_array_equal(tracker.scheduler.window(),original)
    np.testing.assert_allclose(tracker.position,plan.root_positions[0])


def test_registration_control_matches_measured_foot_by_rigid_transform():
    import mujoco
    from scipy.spatial.transform import Rotation
    from supermimic.control.foot_aligned_idle import FootAlignedIdle
    planner = FootAlignedIdle.__new__(FootAlignedIdle)
    planner.model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="pelvis"><freejoint/>
        <geom type="sphere" size=".05"/><body name="left" pos=".1 .1 -.6"><joint name="hinge"/>
        <geom type="sphere" size=".03"/></body><body name="right" pos="0 -.1 -.6">
        <geom type="sphere" size=".03"/></body></body></worldbody></mujoco>''')
    planner.n = 1; planner.default = np.zeros(1); planner.qadr = np.array([7])
    planner.feet = [planner.model.body(s).id for s in ('left','right')]
    data = mujoco.MjData(planner.model)
    rotation = Rotation.from_euler('z',.4)
    planner._fk(data,[2.,-1.,.65],rotation.as_quat()[[3,0,1,2]],[0.])
    context = dict(feet=data.xpos[planner.feet].copy(),foot_rotation=data.xmat[planner.feet].reshape(2,3,3).copy())
    source = np.zeros(12); source[4] = -1.; source[-1] = .65
    plan = planner.register_only(source,context)
    np.testing.assert_allclose(plan.root_positions,[[2,-1,.65],[2,-1,.65]],atol=1e-12)
    np.testing.assert_allclose(plan.yaw,.4,atol=1e-12)
    np.testing.assert_array_equal(plan.features,np.repeat(source[None],2,axis=0))
