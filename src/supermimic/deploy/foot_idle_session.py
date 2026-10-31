"""Experimental asynchronous foot planning around the fixed published window."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.spatial.transform import Rotation

from supermimic.control.foot_aligned_idle import FootAlignedIdle
from supermimic.control.idle_velocity_feedback import RootVelocityEstimate


class FootIdleSession:
    def __init__(self, scene, spec, scheduler):
        self.planner = FootAlignedIdle(scene,spec,scheduler.dt)
        self.scheduler = scheduler
        self.original_stand = scheduler.stand.copy()
        self.velocity = RootVelocityEstimate()
        self.worker = ThreadPoolExecutor(max_workers=1,thread_name_prefix='foot-idle')
        self.future = None
        self.poses = None
        self.pending = deque()
        self.previous = 'stand'
        self.attempted = False
        self.quiet = 0
        self.events = []
        self.status = 'initial'
        self.applied_step = None
        self.duration_steps = None

    def prepare(self, tracker, position, quat, joints, qvel, angular, stamp, mode):
        if stamp is None or angular is None:
            raise ValueError('Foot Idle requires timestamped odometry and angular velocity')
        v = self.velocity.update(stamp,position)
        if mode != self.previous:
            self.scheduler.stand = self.original_stand.copy()
            self.pending.clear()
            self.quiet = 0
            self.attempted = False
            if self.future is not None:
                # In-flight stale work is allowed to finish but never applied.
                self.future.cancel()
                self.future = None
            self.status = 'braking' if mode == 'stand' else 'motion'
            self.previous = mode
        self.scheduler.select(mode)
        if mode != 'stand' or not tracker.has_motion_request:
            return
        context = self.planner.observe(position,quat,joints,qvel,v,angular)
        if self.future is not None and self.future.done():
            future, self.future = self.future, None
            try:
                plan = future.result()
                self.install(tracker,plan,context)
            except Exception as exc:
                self.status = 'rejected'
                self.events.append(dict(kind='rejected',step=self.scheduler.steps,error=str(exc)))
        if self.applied_step is not None and self.status == 'transition' and self.scheduler.steps >= self.applied_step+self.scheduler.horizon+self.duration_steps:
            self.status = 'hold'
            self.events.append(dict(kind='hold',step=self.scheduler.steps))
        stable = np.allclose(self.scheduler.window(),self.original_stand,atol=1e-7) and not self.scheduler.plan
        self.quiet = self.quiet+1 if stable and self.planner.ready(context) else 0
        if not self.attempted and self.quiet >= 10:
            self.future = self.worker.submit(self.planner.build,self.scheduler.queue[-1].copy(),context)
            self.attempted = True
            self.status = 'planning'
            self.events.append(dict(kind='submit',step=self.scheduler.steps))

    def install(self, tracker, plan, context):
        support = ('left','right').index(plan.report['support_side'])
        old = plan.context
        old_yaw = Rotation.from_matrix(old['foot_rotation'][support]).as_euler('xyz')[2]
        new_yaw = Rotation.from_matrix(context['foot_rotation'][support]).as_euler('xyz')[2]
        angle = float(np.arctan2(np.sin(new_yaw-old_yaw),np.cos(new_yaw-old_yaw)))
        rz = Rotation.from_euler('z',angle)
        translation = context['feet'][support]-rz.apply(old['feet'][support])
        translation[2] = 0.
        moved = rz.apply(old['feet'])+translation-context['feet']
        rotation_error = max(np.linalg.norm(Rotation.from_matrix(context['foot_rotation'][s]@(rz.as_matrix()@old['foot_rotation'][s]).T).as_rotvec()) for s in (0,1))
        if np.max(np.linalg.norm(moved[:,:2],axis=1)) > .015 or rotation_error > .09 or not self.planner.ready(context):
            raise ValueError('Planning context stale: relative feet or readiness changed')
        positions = rz.apply(plan.root_positions)+translation
        yaw = plan.yaw+angle
        # Exactly preserve all already-published rows. Their world registration
        # remains old until the first new row reaches the front of this queue.
        self.poses = deque([(tracker.position.copy(),tracker.yaw) for _ in range(self.scheduler.horizon)])
        self.pending = deque(zip(positions[1:].copy(),yaw[1:].copy()))
        self.scheduler.plan = deque(np.asarray(plan.features[1:],dtype=np.float32))
        self.scheduler.stand = np.asarray(plan.features[-1],dtype=np.float32)
        self.applied_step = self.scheduler.steps
        self.duration_steps = len(plan.features)-1
        self.status = 'transition'
        self.events.append(dict(kind='installed',step=self.scheduler.steps,report=plan.report,
                                registration_delta=(positions[0]-tracker.position).tolist(),
                                registration_yaw_delta=float(yaw[0]-tracker.yaw),
                                terminal_root=positions[-1].tolist(),terminal_yaw=float(yaw[-1]),
                                foot_targets=(rz.apply(np.asarray(plan.report['measured_foot_targets']))+translation).tolist()))

    def advance(self, tracker):
        if self.poses is None:
            return False
        previous_tail, previous_yaw = self.poses[-1]
        tail = self.scheduler.queue[-1].copy()
        self.poses.popleft()
        self.scheduler.advance()
        if self.pending:
            pose = self.pending.popleft()
        else:
            velocity = tail[2*tracker.nj+6:2*tracker.nj+9]
            pos = previous_tail+Rotation.from_euler('z',previous_yaw).apply(velocity)*self.scheduler.dt
            pose = (pos,previous_yaw+float(tail[2*tracker.nj+5])*self.scheduler.dt)
        self.poses.append(pose)
        tracker.position, tracker.yaw = self.poses[0][0].copy(),float(self.poses[0][1])
        return True

    def close(self):
        self.worker.shutdown(wait=True,cancel_futures=True)
