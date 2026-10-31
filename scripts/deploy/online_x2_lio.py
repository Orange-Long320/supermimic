"""Live diagnostic X2 lidar/IMU -> bounded propagated pelvis odometry.

Run only inside lxh. Publishes adapted sensors, diagnostic odometry and status;
there are no actuator publishers, control clients, or policy execution paths.
"""
import argparse
from bisect import bisect_left
from collections import Counter, deque
import json
from pathlib import Path
import subprocess
import time
import uuid

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from prepare_x2_fastlio_bag import convert_cloud, lidar_pose_in_imu
from supermimic.deploy.online_inertial import parse_filter_state, CachedImuPropagation, fit_level_floor
from supermimic.deploy.pelvis_state import UrdfPelvisMount, pelvis_from_imu, pelvis_angular_velocity, gravity_level_rotation, latest_causal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('binary', 'urdf', 'profile', 'extrinsics', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=0., help='0 keeps the diagnostic running until stopped')
    parser.add_argument('--state-clock', choices=('latest-imu', 'imu-grid'), default='latest-imu',
                        help='imu-grid estimates on a 20 ms grid bracketed by received IMU; no extrapolation')
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from rclpy.executors import ExternalShutdownException
    from sensor_msgs.msg import PointCloud2, Imu
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Float64MultiArray, String
    from tf2_msgs.msg import TFMessage
    from aimdk_msgs.msg import JointStateArray

    spec = json.loads(args.profile.read_text()); names = spec['joint_names']
    if len(names) != 31 or spec['robot'] != 'x2_ultra': raise ValueError('Wrong robot profile')
    transforms = json.loads(args.extrinsics.read_text())['transforms']
    def transform(row): return np.asarray(row['xyz']), Rotation.from_quat(row['xyzw']).as_matrix()
    ptl, rtl = transform(transforms['torso_link -> lidar_chest_front'])
    pli, rli = transform(transforms['lidar_chest_front -> lidar_imu_chest_front'])
    mount = UrdfPelvisMount(args.urdf, names, ptl+rtl@pli, rtl@rli)
    tli, rli_inverse = lidar_pose_in_imu(pli, transforms['lidar_chest_front -> lidar_imu_chest_front']['xyzw'])
    imu_from_lidar_r = np.asarray(rli_inverse).reshape(3, 3)
    imu_from_lidar_p = np.asarray(tli)
    prefix = '/supermimic/x2_online_lio'
    params = {'feature_extract_enable': False, 'point_filter_num': 2, 'max_iteration': 3,
              'filter_size_surf': .3, 'filter_size_map': .3, 'cube_side_length': 100.,
              'runtime_pos_log_enable': False,
              'common': {'lid_topic': prefix+'/cloud', 'imu_topic': prefix+'/imu',
                         'time_sync_en': False, 'time_offset_lidar_to_imu': 0.},
              'preprocess': {'lidar_type': 2, 'scan_line': 96, 'scan_rate': 10, 'timestamp_unit': 0, 'blind': .3},
              'mapping': {'acc_cov': .1, 'gyr_cov': .1, 'b_acc_cov': .0001, 'b_gyr_cov': .0001,
                          'fov_degree': 120., 'det_range': 30., 'extrinsic_est_en': False,
                          'extrinsic_T': tli, 'extrinsic_R': rli_inverse},
              'publish': {'path_en': False, 'scan_publish_en': False, 'dense_publish_en': False,
                          'scan_bodyframe_pub_en': False}, 'pcd_save': {'pcd_save_en': False, 'interval': -1}}
    config = args.out/'fastlio-live.yaml'
    config.write_text(yaml.safe_dump({'/**': {'ros__parameters': params}}))

    class Bridge(Node):
        def __init__(self):
            super().__init__('supermimic_online_pelvis_bridge', enable_rosout=False, start_parameter_services=False)
            self.started = time.monotonic(); self.session = uuid.uuid4().hex
            self.running = True
            self.imu_rows = deque(maxlen=256); self.imu_stamps_ns = deque(maxlen=256)
            self.groups = {g: deque(maxlen=128) for g in ('leg', 'waist', 'head', 'arm')}
            self.propagation = CachedImuPropagation()
            self.last_emitted_ns = None
            self.counts = Counter(); self.rejects = Counter(); self.state = None
            self.level = None; self.shift = None; self.floor = None; self.cloud = None
            self.static_ready = False; self.mount_changed = False; self.last_pose = None
            self.live_tf_seen = set(); self.last_reason = 'initializing'; self.state_received = None
            self.cloud_times = deque(maxlen=128)
            self.frames = (args.out/'pelvis-frames.jsonl').open('x')
            self.status_log = (args.out/'status.jsonl').open('x')
            best = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
            imu_qos = QoSProfile(depth=400, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
            self.cloud_pub = self.create_publisher(PointCloud2, prefix+'/cloud', best)
            self.imu_pub = self.create_publisher(Imu, prefix+'/imu', 400)
            self.pose_pub = self.create_publisher(Odometry, '/supermimic/pelvis/odometry', 5)
            self.status_pub = self.create_publisher(String, prefix+'/status', 5)
            self.policy_state_pub = self.create_publisher(String, prefix+'/policy_state', best)
            self.subs = [self.create_subscription(Imu, '/aima/hal/sensor/lidar_chest_front/imu', self.on_imu, imu_qos),
                         self.create_subscription(PointCloud2, '/aima/hal/sensor/lidar_chest_front/lidar_pointcloud', self.on_cloud, best),
                         self.create_subscription(Float64MultiArray, prefix+'/filter_state', self.on_state, 20)]
            for group in self.groups:
                self.subs.append(self.create_subscription(JointStateArray, '/aima/hal/joint/'+group+'/state',
                                                          lambda msg, g=group: self.on_joints(msg, g), best))
            tf_qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.subs.append(self.create_subscription(TFMessage, '/tf_static', self.on_tf, tf_qos))
            self.timer = self.create_timer(.02, self.publish_pose)
            self.status_timer = self.create_timer(1., self.publish_status)

        @staticmethod
        def stamp(msg): return msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9

        def on_imu(self, msg):
            self.counts['raw_imu'] += 1
            stamp = self.stamp(msg); w, a = msg.angular_velocity, msg.linear_acceleration
            row = [stamp, w.x, w.y, w.z, a.x, a.y, a.z]
            reason = ('imu frame' if msg.header.frame_id != 'lidar_imu_chest_front' else
                      'imu nonfinite' if not np.isfinite(row).all() else
                      'imu future' if stamp > time.time() else
                      'imu stale at receive' if time.time()-stamp > .1 else
                      'imu nonmonotonic' if self.imu_rows and stamp <= self.imu_rows[-1][0] else None)
            if reason:
                self.rejects[reason] += 1; return
            self.imu_rows.append(row)
            self.imu_stamps_ns.append(msg.header.stamp.sec*10**9+msg.header.stamp.nanosec)
            self.imu_pub.publish(msg); self.counts['adapted_imu'] += 1
            if args.state_clock == 'imu-grid':
                # Emit as soon as actual IMU data bracket the next grid time.
                # The timer remains a retry path, not a second sampling clock.
                self.publish_pose()

        def on_cloud(self, msg):
            before = time.monotonic()
            self.counts['raw_cloud'] += 1
            try:
                if msg.header.frame_id != 'lidar_chest_front' or not 0 <= time.time()-self.stamp(msg) <= .3:
                    raise ValueError('cloud frame or age')
                if not self.static_ready:
                    if len(self.imu_rows)<200: return
                    recent = np.asarray(self.imu_rows)[-200:]
                    if np.linalg.norm(recent[:, 1:4], axis=1).max() > .05 or np.std(recent[:, 4:7], axis=0).max()>.2:
                        raise ValueError('waiting for static IMU initialization')
                    _, joint_speed, _ = self.joints(recent[-1, 0])
                    if np.max(np.abs(joint_speed)) > .05: raise ValueError('waiting for static joints')
                    self.static_ready = True
                converted, _ = convert_cloud(msg, 1e-9)
                # Native host header is retained. The known hardware timing
                # uncertainty is not replaced by a guessed constant offset.
                self.cloud_pub.publish(converted); self.counts['adapted_cloud'] += 1
                if self.shift is None:
                    xyz = np.column_stack([np.ndarray(converted.width, dtype='<f4', buffer=converted.data,
                                             offset=o, strides=(24,))[::12] for o in (0, 4, 8)])
                    self.cloud = xyz
            except ValueError as exc: self.rejects[str(exc)] += 1
            finally: self.cloud_times.append((time.monotonic()-before)*1000)

        def on_joints(self, msg, group):
            self.counts['joints_'+group] += 1
            self.groups[group].append({'received_ns': time.time_ns(),
                                       'header_ns': msg.header.stamp.sec*10**9+msg.header.stamp.nanosec,
                                       'domain': int(msg.state.value),
                                       'joints': [(j.name, j.position, j.velocity, j.error_code) for j in msg.joints]})

        def joints(self, stamp, *, with_rows=False, at_ns=None):
            at_ns = int(stamp*1e9) if at_ns is None else at_ns
            rows = [latest_causal(list(v), time.time_ns(), at_ns=at_ns, max_age_s=.05) for v in self.groups.values()]
            stamps = [r['header_ns'] for r in rows]
            if max(stamps)-min(stamps) > 30_000_000: raise ValueError('joint group skew')
            values = [j for r in rows for j in r['joints']]; lookup = {j[0]: j for j in values}
            if len(values)!=31 or set(lookup)!=set(names) or any(j[3] for j in values): raise ValueError('joint set or error code')
            q = np.asarray([lookup[n][1] for n in names]); v = np.asarray([lookup[n][2] for n in names])
            if not np.isfinite(np.r_[q, v]).all(): raise ValueError('nonfinite joints')
            result = (q, v, [(stamp*1e9-t)*1e-9 for t in stamps])
            return (*result, rows) if with_rows else result

        def on_tf(self, msg):
            for tf in msg.transforms:
                key = tf.header.frame_id+' -> '+tf.child_frame_id
                if key not in transforms: continue
                p, q = tf.transform.translation, tf.transform.rotation
                live_p = np.array([p.x, p.y, p.z]); live_r = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                recorded_p, recorded_r = transform(transforms[key])
                if not np.allclose(live_p, recorded_p, atol=1e-5) or not np.allclose(live_r, recorded_r, atol=1e-5):
                    self.mount_changed = True
                self.live_tf_seen.add(key)

        def on_state(self, msg):
            self.counts['lio_corrections'] += 1
            try:
                state = parse_filter_state(msg.data)
                if state.stamp > time.time() or (self.state and state.stamp <= self.state.stamp):
                    raise ValueError('future or backward LIO state')
                self.state = state; self.state_received = time.monotonic()
                self.counts['lio_native_imu_callbacks'] = int(msg.data[21]); self.counts['lio_native_cloud_callbacks'] = int(msg.data[22])
                if self.level is None:
                    self.level = Rotation.from_matrix(gravity_level_rotation(-state.gravity))
            except ValueError as exc: self.rejects[str(exc)] += 1

        def publish_pose(self):
            try:
                if self.process.poll() is not None: raise RuntimeError('Owned FAST-LIO process exited')
                if self.mount_changed: raise ValueError('Native mount TF changed; restart and review')
                if self.state is None: raise ValueError('waiting for LIO correction')
                now = time.time(); age = now-self.state.stamp
                if not 0 <= age <= .2: raise ValueError('LIO correction stale')
                ns = self.imu_stamps_ns[-1]
                grid = args.state_clock == 'imu-grid'
                support = None
                if grid:
                    ns = ns//20_000_000*20_000_000
                    if self.last_emitted_ns is not None and ns <= self.last_emitted_ns: return
                    raw_ns = list(self.imu_stamps_ns)
                    right = bisect_left(raw_ns, ns)
                    if right == len(raw_ns) or (right == 0 and raw_ns[0] != ns):
                        raise ValueError('IMU does not bracket grid time')
                    left = right if raw_ns[right] == ns else right-1
                    support = [raw_ns[left], raw_ns[right]]
                propagated, gyro = self.propagation.advance(self.state, list(self.imu_rows), now,
                                                           until=ns*1e-9 if grid else None)
                imu_age = now-propagated.stamp
                if not 0 <= imu_age <= .05: raise ValueError('latest IMU stale')
                q, qd, joint_ages, joint_rows = self.joints(propagated.stamp, with_rows=True, at_ns=ns)
                ppi, rpi, chain_omega = mount.mount(q, qd)
                position, rotation = pelvis_from_imu(propagated.position, propagated.rotation.as_matrix(), ppi, rpi)
                position = self.level.apply(position); rotation = self.level.as_matrix()@rotation
                omega = pelvis_angular_velocity(gyro, rpi, chain_omega)
                epsilon = 1e-5
                mount_v = (mount.mount(q+epsilon*qd)[0]-mount.mount(q-epsilon*qd)[0])/(2*epsilon)
                velocity = self.level.apply(propagated.velocity)-rotation@mount_v-np.cross(rotation@omega, rotation@ppi)
                if self.shift is None:
                    if self.cloud is None: raise ValueError('waiting for live floor cloud')
                    world = self.level.apply(self.state.rotation.apply(self.cloud@imu_from_lidar_r.T+imu_from_lidar_p)+self.state.position)
                    floor, quality = fit_level_floor(world, position)
                    self.shift = np.array([-position[0], -position[1], -floor]); self.floor = quality
                    (args.out/'initialization.json').write_text(json.dumps({'session_id': self.session,
                        'shift': self.shift.tolist(), 'level_rotation': self.level.as_matrix().tolist(), 'floor': self.floor,
                        'floor_source': 'live static point-cloud plane; not physical calibration',
                        'hardware_sampling_time_verified': False, 'control_ready': False}, indent=2)+'\n')
                    self.cloud = None
                position += self.shift
                if not np.isfinite(np.r_[position, velocity, omega, rotation.ravel()]).all(): raise ValueError('nonfinite pelvis')
                quat = Rotation.from_matrix(rotation).as_quat()
                out = Odometry()
                out.header.stamp.sec, out.header.stamp.nanosec = divmod(ns, 10**9)
                out.header.frame_id = 'supermimic_world'; out.child_frame_id = 'pelvis'
                out.pose.pose.position.x, out.pose.pose.position.y, out.pose.pose.position.z = map(float, position)
                out.pose.pose.orientation.x, out.pose.pose.orientation.y, out.pose.pose.orientation.z, out.pose.pose.orientation.w = map(float, quat)
                out.twist.twist.linear.x, out.twist.twist.linear.y, out.twist.twist.linear.z = map(float, rotation.T@velocity)
                out.twist.twist.angular.x, out.twist.twist.angular.y, out.twist.twist.angular.z = map(float, omega)
                self.pose_pub.publish(out); self.counts['pelvis_published'] += 1
                self.last_pose = {'published_wall': now, 'state_stamp': propagated.stamp, 'lio_stamp': self.state.stamp,
                                 'imu_age_s': imu_age, 'lio_age_s': age, 'joint_age_s': joint_ages,
                                 'position': position.tolist(), 'quat_xyzw': quat.tolist(),
                                 'angular_velocity_body': omega.tolist(), 'velocity_world': velocity.tolist()}
                packet = dict(self.last_pose, schema=2, session_id=self.session, joint_names=names,
                              quat_wxyz=quat[[3, 0, 1, 2]].tolist(),
                              joint_position=q.tolist(), joint_velocity=qd.tolist(),
                              joint_stamps=[r['header_ns']*1e-9 for r in joint_rows],
                              joint_received_wall=[r['received_ns']*1e-9 for r in joint_rows],
                              joint_domains=[r['domain'] for r in joint_rows],
                              state_stamp_ns=ns, joint_stamps_ns=[r['header_ns'] for r in joint_rows],
                              motor_output_enabled=False, eligible_for_control=False)
                if grid:
                    packet.update(schema=3, state_time_kind='resampled_imu_grid',
                                  imu_support_stamps_ns=support)
                state_msg = String(); state_msg.data = json.dumps(packet, allow_nan=False)
                self.policy_state_pub.publish(state_msg); self.counts['policy_state_published'] += 1
                self.last_emitted_ns = ns
                self.frames.write(json.dumps(self.last_pose, allow_nan=False)+'\n')
                self.last_reason = 'publishing diagnostic pelvis state'
            except ValueError as exc:
                self.last_reason = str(exc); self.rejects[self.last_reason] += 1

        def publish_status(self):
            result = {'session_id': self.session, 'elapsed_s': time.monotonic()-self.started,
                      'running': self.running,
                      'counts': dict(self.counts), 'rejected_ticks': dict(self.rejects), 'status': self.last_reason,
                      'floor': self.floor, 'latest_pose': self.last_pose, 'live_mount_tf_seen': sorted(self.live_tf_seen),
                      'cloud_callback_ms_p50_max': [float(np.median(self.cloud_times)), float(max(self.cloud_times))] if self.cloud_times else None,
                      'header_mode': 'native_host_unchanged', 'hardware_sampling_time_verified': False,
                      'state_clock': args.state_clock,
                      'dynamic_accuracy_validated': False, 'covariance_validated': False,
                      'control_ready': False, 'motor_output_enabled': False}
            msg = String(); msg.data = json.dumps(result, allow_nan=False); self.status_pub.publish(msg)
            self.frames.flush(); self.status_log.write(msg.data+'\n'); self.status_log.flush()
            temporary = args.out/'status-latest.tmp'; temporary.write_text(msg.data+'\n'); temporary.replace(args.out/'status-latest.json')

    rclpy.init(); node = Bridge()
    command = ['nice', '-n', '10', str(args.binary.resolve()), '--ros-args', '--params-file', str(config.resolve()),
               '-r', '__node:=supermimic_fastlio_online', '-r', '__ns:='+prefix,
               '-r', '/Odometry:='+prefix+'/imu_odometry', '-r', '/tf:='+prefix+'/tf',
               '-r', '/path:='+prefix+'/path', '-r', '/cloud_registered:='+prefix+'/registered',
               '-r', '/cloud_registered_body:='+prefix+'/registered_body', '-r', '/Laser_map:='+prefix+'/map']
    (args.out/'command.json').write_text(json.dumps({'argv': command, 'scope': __doc__}, indent=2)+'\n')
    log = (args.out/'fastlio.log').open('x')
    node.process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    (args.out/'fastlio.pid').write_text(str(node.process.pid)+'\n')
    import os
    (args.out/'bridge.pid').write_text(str(os.getpid())+'\n')
    try:
        while rclpy.ok() and (args.seconds == 0 or time.monotonic()-node.started < args.seconds):
            rclpy.spin_once(node, timeout_sec=.02)
    except (KeyboardInterrupt, ExternalShutdownException): pass
    finally:
        node.running = False
        try: node.publish_status()
        except Exception: pass  # ROS may already be shut down; child cleanup still must run.
        node.frames.close(); node.status_log.close()
        node.process.terminate()
        try: node.process.wait(timeout=10)
        except subprocess.TimeoutExpired: node.process.kill(); node.process.wait()
        log.close(); node.destroy_node(); rclpy.try_shutdown()


if __name__ == '__main__': main()
