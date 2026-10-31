"""Bounded passive IMU and joint sampling. No commands and no actuation."""
import json
import math
from pathlib import Path
import statistics
import sys
import time


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import Imu
    from aimdk_msgs.msg import JointStateArray

    out = Path(sys.argv[1])
    if out.exists():
        raise ValueError('New output path required')

    class Observer(Node):
        def __init__(self):
            super().__init__('supermimic_imu_joint_passive')
            self.imu = []; self.groups = {}; self.times = {}
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE)
            self.subs = [self.create_subscription(Imu, '/aima/hal/imu/torso/state', self.on_imu, qos)]
            for group in ('leg', 'waist', 'head', 'arm'):
                self.subs.append(self.create_subscription(JointStateArray, f'/aima/hal/joint/{group}/state',
                                                          lambda msg, g=group:self.on_joint(g,msg), qos))

        def on_imu(self, msg):
            now = time.monotonic(); a = msg.linear_acceleration; w = msg.angular_velocity; q = msg.orientation
            self.imu.append({'time':now, 'stamp':msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9,
                             'accel':[a.x,a.y,a.z], 'omega':[w.x,w.y,w.z],
                             'orientation':[q.w,q.x,q.y,q.z], 'frame':msg.header.frame_id})

        def on_joint(self, group, msg):
            self.times.setdefault(group, []).append(time.monotonic())
            if group not in self.groups:
                self.groups[group] = {'joint_count':len(msg.joints), 'names':[j.name for j in msg.joints],
                                      'first_error_codes':[j.error_code for j in msg.joints]}

    rclpy.init(); node = Observer(); start = time.monotonic()
    try:
        while time.monotonic()-start < 10.:
            rclpy.spin_once(node, timeout_sec=.05)
        samples = node.imu
        if not samples:
            raise RuntimeError('No IMU samples received')
        def intervals(times):
            diffs = [b-a for a,b in zip(times,times[1:])]
            return {'count':len(times), 'max_gap_s':max(diffs, default=None),
                    'median_gap_s':statistics.median(diffs) if diffs else None}
        acc_norm = [math.sqrt(sum(x*x for x in s['accel'])) for s in samples]
        omega_norm = [math.sqrt(sum(x*x for x in s['omega'])) for s in samples]
        quat_norm = [math.sqrt(sum(x*x for x in s['orientation'])) for s in samples]
        report = {'seconds':time.monotonic()-start,
                  'imu':{**intervals([s['time'] for s in samples]),
                         'frame_ids':sorted(set(s['frame'] for s in samples)),
                         'all_finite':all(math.isfinite(v) for s in samples for v in s['accel']+s['omega']+s['orientation']),
                         'orientation_norm_min':min(quat_norm), 'orientation_norm_max':max(quat_norm),
                         'acceleration_norm_median':statistics.median(acc_norm),
                         'acceleration_norm_min':min(acc_norm), 'acceleration_norm_max':max(acc_norm),
                         'angular_speed_median':statistics.median(omega_norm),
                         'stamp_nonzero_fraction':sum(s['stamp']>0 for s in samples)/len(samples)},
                  'groups':{g:{**data, **intervals(node.times[g])} for g,data in node.groups.items()},
                  'scope':'passive first-sample joint checks and 10-second topic timing; no pose ground truth, no motor commands'}
        out.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
    finally:
        node.destroy_node(); rclpy.try_shutdown()


if __name__ == '__main__':
    main()
