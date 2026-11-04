import numpy as np
from scipy.spatial.transform import Rotation

from supermimic.deploy.skill_goal import SkillGoal
from supermimic.control.skill_phase import SkillPhase


def test_standing_gain_ramps_without_altering_future_window():
    class Scheduler:
        dt = .02
        stand = np.array([0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, .65], dtype=np.float32)
        def select(self, mode): pass
        def window(self): return np.tile(self.stand, (10, 1))
    obj = SkillGoal.__new__(SkillGoal)
    obj.anchor_obs = True
    obj.scheduler = Scheduler(); obj.nj = 1; obj.default = np.zeros(1)
    obj.has_motion_request = False; obj.idle_velocity_feedback = False; obj.planner = 'baseline'
    obj.idle_anchor_mode = 'fixed'; obj.foot_idle = None; obj.idle_root_velocity = np.zeros(3)
    obj.position = np.zeros(3); obj.yaw = 0.; obj.stand_anchor_gain = 2.; obj.stand_gain = 1.
    obj.stand_gain_quiet_time=0.; obj.stand_gain_speed_limit=.08; obj.stand_quiet_steps=0; obj.last_position=None
    obj.reference, obj.actual = object(), object()
    obj.fk = lambda data, pos, quat, joints: (np.array([1., 0., 0.]) if data is obj.reference else np.zeros(3), Rotation.identity())
    goals = [obj.goal([0,0,0], [1,0,0,0], [0], 'stand') for _ in range(50)]
    assert abs(goals[-1][-9]-2.) < 1e-6
    assert max(abs(np.diff([g[-9] for g in goals]))) < .021
    assert all(np.array_equal(g[:-9], goals[0][:-9]) for g in goals)
    after = obj.goal([0,0,0], [1,0,0,0], [0], 'jog')
    assert abs(after[-9]-1.98) < 1e-6
    obj.stand_gain=1.; obj.stand_gain_quiet_time=.1; obj.stand_quiet_steps=0; obj.last_position=None
    for i in range(10):
        goal = obj.goal([i*.01,0,0], [1,0,0,0], [0], 'stand')
    assert abs(goal[-9]-1.) < 1e-6  # moving faster than the quiet threshold
    for i in range(4): obj.goal([.09,0,0], [1,0,0,0], [0], 'stand')
    assert obj.stand_gain == 1.
    obj.goal([.09,0,0], [1,0,0,0], [0], 'stand')
    assert obj.stand_gain > 1.


def turn_goal_fixture():
    """Use the real phase queue and simple pelvis FK to isolate registration."""
    obj = SkillGoal.__new__(SkillGoal)
    obj.anchor_obs = True
    obj.nj = 1
    obj.default = np.zeros(1)
    row = np.zeros(12, dtype=np.float32)
    row[4], row[-1] = -1., .65
    names = ('walk', 'turn_left_inplace', 'turn_right_inplace')
    clips = {n: np.tile(row, (50, 1)) for n in names}
    obj.scheduler = SkillPhase(clips, clips, {n: 0 for n in names}, 1)
    obj.has_motion_request = False
    obj.idle_velocity_feedback = False
    obj.planner = 'baseline'
    obj.idle_anchor_mode = 'fixed'
    obj.foot_idle = None
    obj.position = np.array([8., -4., .65])
    obj.yaw = .7
    obj.stand_anchor_gain = obj.stand_gain = 1.
    obj.stand_gain_quiet_time = 0.
    obj.stand_gain_speed_limit = .08
    obj.stand_quiet_steps = 0
    obj.last_position = None
    obj.reference, obj.actual = object(), object()
    obj.fk = lambda data, pos, quat, joints: (np.asarray(pos, dtype=float), obj.rotation(quat))
    return obj


def test_new_turn_holds_entry_xy_and_preserves_heading_and_committed_window():
    obj = turn_goal_fixture()
    before = obj.scheduler.window().copy()
    goal = obj.goal([2., 3., .65], [1, 0, 0, 0], [0], 'turn_left_inplace')
    np.testing.assert_array_equal(obj.position, [2., 3., .65])
    np.testing.assert_array_equal(goal[-9:-7], [0., 0.])
    np.testing.assert_array_equal(obj.scheduler.window(), before)
    assert obj.yaw == .7
    # A displaced body must see restoring feedback during the same held turn.
    goal = obj.goal([2.2, 2.9, .65], [1, 0, 0, 0], [0], 'turn_left_inplace')
    np.testing.assert_allclose(goal[-9:-7], [-.2, .1], atol=1e-7)
    np.testing.assert_array_equal(obj.position[:2], [2., 3.])


def test_turn_reentry_registers_again_but_stand_and_walk_keep_position_feedback():
    obj = turn_goal_fixture()
    for mode in ('walk', 'stand'):
        obj.goal([2., 3., .65], [1, 0, 0, 0], [0], mode)
        np.testing.assert_array_equal(obj.position[:2], [8., -4.])
    for mode, xy in (('turn_left_inplace', [2., 3.]),
                     ('turn_right_inplace', [3., 4.])):
        obj.goal([*xy, .65], [1, 0, 0, 0], [0], mode)
        np.testing.assert_array_equal(obj.position[:2], xy)
    obj.goal([5., 6., .65], [1, 0, 0, 0], [0], 'stand')
    np.testing.assert_array_equal(obj.position[:2], [3., 4.])
    obj.goal([5., 6., .65], [1, 0, 0, 0], [0], 'turn_right_inplace')
    np.testing.assert_array_equal(obj.position[:2], [5., 6.])


def test_no_anchor_turn_keeps_reference_registration_unchanged():
    obj = turn_goal_fixture()
    obj.anchor_obs = False
    goal = obj.goal([2., 3., .65], [1, 0, 0, 0], [0], 'turn_left_inplace')
    assert goal.shape == (120,)
    np.testing.assert_array_equal(obj.position[:2], [8., -4.])


def test_world_horizontal_intervention_removes_drift_even_when_body_is_tilted():
    modes=('stand','turn_left_inplace','turn_right_inplace')
    for mode in modes:
        obj=turn_goal_fixture();obj.zero_horizontal_anchor_modes=modes
        quat=Rotation.from_euler('xyz',[.1,.34,-.2]).as_quat()[[3,0,1,2]]
        position=np.array([2.,3.,.75])
        goal=obj.goal(position,quat,[0],mode)
        moved=obj.goal(position+[11.,-7.,0.],quat,[0],mode)
        np.testing.assert_allclose(moved,goal,atol=1e-7)
        expected=Rotation.from_quat(quat[[1,2,3,0]]).inv().apply([0,0,obj.scheduler.window()[0,-1]-.75])
        np.testing.assert_allclose(goal[-9:-6],expected,atol=1e-7)
        # Nonzero body-frame XY from a height difference must be preserved.
        assert np.linalg.norm(goal[-9:-7])>0
        plain=turn_goal_fixture();baseline=plain.goal(position,quat,[0],mode)
        np.testing.assert_array_equal(goal[:-9],baseline[:-9])
        np.testing.assert_allclose(goal[-6:],baseline[-6:],atol=1e-7)


def test_horizontal_intervention_keeps_walking_feedback_and_mode_transition():
    obj=turn_goal_fixture();obj.zero_horizontal_anchor_modes=('stand','turn_left_inplace','turn_right_inplace')
    obj.goal([2,3,.65],[1,0,0,0],[0],'turn_left_inplace')
    stand=obj.goal([4,5,.65],[1,0,0,0],[0],'stand')
    np.testing.assert_array_equal(stand[-9:-7],[0,0])
    walking=obj.goal([4,5,.65],[1,0,0,0],[0],'walk')
    np.testing.assert_allclose(walking[-9:-7],[-2,-2],atol=1e-7)
