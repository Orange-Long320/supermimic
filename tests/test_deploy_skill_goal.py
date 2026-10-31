import numpy as np
from scipy.spatial.transform import Rotation

from supermimic.deploy.skill_goal import SkillGoal


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
