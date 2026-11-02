"""Record simulator streams in independent ROS subscriber processes.

Each stream has its own executor and JSONL writer. DDS remains best effort, so
header comparison with an independent subscriber is necessary to assess loss.
"""
import argparse
import heapq
import json
import multiprocessing as mp
import os
import signal
import time
from pathlib import Path

GROUPS = ('leg', 'waist', 'head', 'arm')
STREAMS = ('imu', 'chest', 'odom', *GROUPS)


def record_stream(out, stream, stop):
    """Run a single subscription in a process isolated from all other streams."""
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import ExternalShutdownException
    from rclpy.impl.implementation_singleton import rclpy_implementation
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from aimdk_msgs.msg import JointStateArray
    from sensor_msgs.msg import Imu
    from nav_msgs.msg import Odometry

    filenames = {'imu': 'imu.jsonl', 'chest': 'imu-chest.jsonl', 'odom': 'odom.jsonl'}
    filenames.update({g: f'joints-{g}.jsonl' for g in GROUPS})
    topics = {'imu': (Imu, '/aima/hal/imu/torso/state'),
              'chest': (Imu, '/aima/hal/imu/chest/state'),
              'odom': (Odometry, '/aima/hal/odom/state')}
    topics.update({g: (JointStateArray, f'/aima/hal/joint/{g}/state') for g in GROUPS})
    counts = {'received': 0, 'written': 0}
    failure = None
    output = open(os.path.join(out, filenames[stream]), 'x')

    class Recorder(Node):
        def __init__(self):
            super().__init__('supermimic_native_recorder_' + stream)
            qos = QoSProfile(depth=400, reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE)
            kind, topic = topics[stream]
            self.sub = self.create_subscription(kind, topic, self.receive, qos)

        def receive(self, msg):
            recv = time.monotonic(); counts['received'] += 1
            record = {'stamp': [msg.header.stamp.sec, msg.header.stamp.nanosec], 'recv': recv}
            if stream in ('imu', 'chest'):
                q, w, acc = msg.orientation, msg.angular_velocity, msg.linear_acceleration
                record.update(frame_id=msg.header.frame_id, q_xyzw=[q.x, q.y, q.z, q.w],
                              w=[w.x, w.y, w.z], a=[acc.x, acc.y, acc.z])
            elif stream == 'odom':
                pose, tw = msg.pose.pose, msg.twist.twist
                p, q = pose.position, pose.orientation
                record.update(position=[p.x, p.y, p.z], q_xyzw=[q.x, q.y, q.z, q.w],
                              linear=[tw.linear.x, tw.linear.y, tw.linear.z],
                              angular=[tw.angular.x, tw.angular.y, tw.angular.z],
                              frame_id=msg.header.frame_id, child_frame_id=msg.child_frame_id)
            else:
                record.update(group=stream,
                              joints=[[j.name, j.position, j.velocity, j.effort] for j in msg.joints])
            output.write(json.dumps(record, allow_nan=False) + '\n')
            counts['written'] += 1

    node = None
    try:
        rclpy.init(); node = Recorder()
        while not stop.is_set():
            rclpy.spin_once(node, timeout_sec=.05)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except rclpy_implementation.RCLError as exc:
        if rclpy.ok():
            failure = repr(exc)
    except Exception as exc:
        failure = repr(exc)
    finally:
        output.close()
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()
        with open(os.path.join(out, f'writer-summary-{stream}.json'), 'x') as handle:
            json.dump({**counts, 'failure': failure}, handle)
    if failure is not None:
        raise SystemExit(failure)


def finish_worker(worker, timeout):
    """Bound shutdown and terminate only this recorder's own stuck child."""
    worker.join(timeout=max(0., timeout))
    timed_out = worker.is_alive()
    if timed_out:
        worker.terminate()
        worker.join(timeout=2.)
    return not timed_out and worker.exitcode == 0, timed_out


def merge_joints(out):
    """Restore one receive-ordered JSONL without holding the recording in memory."""
    handles = [open(os.path.join(out, f'joints-{group}.jsonl')) for group in GROUPS]
    try:
        with open(os.path.join(out, 'joints.jsonl'), 'x') as target:
            for line in heapq.merge(*handles, key=lambda line: json.loads(line)['recv']):
                target.write(line)
    finally:
        for handle in handles:
            handle.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--seconds', type=float, required=True)
    p.add_argument('--ros-domain', type=int, default=179)
    p.add_argument('--sim-only', action='store_true', required=True)
    a = p.parse_args()
    if not 128 <= a.ros_domain <= 232 or os.environ.get('ROS_LOCALHOST_ONLY') != '1' \
            or os.environ.get('ROS_DOMAIN_ID') != str(a.ros_domain):
        raise RuntimeError('Requires the explicitly selected isolated simulation ROS domain and localhost only')
    if not 1. <= a.seconds <= 600.:
        raise ValueError('Bounded duration required')
    out = os.path.abspath(a.out); os.makedirs(out, exist_ok=False)
    stop = mp.Event()
    workers = {key: mp.Process(target=record_stream, args=(out, key, stop),
                               name='native-recorder-' + key) for key in STREAMS}
    for worker in workers.values():
        worker.start()
    started = time.monotonic(); failure = None
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        while not stop.is_set() and time.monotonic() - started < a.seconds:
            if any(not worker.is_alive() for worker in workers.values()):
                failure = 'subscriber process exited early'
                break
            stop.wait(.1)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        deadline = time.monotonic() + 30.
        finished = {key: finish_worker(workers[key], max(0., deadline-time.monotonic())) for key in STREAMS}
    details = {}
    for key, (ok, timed_out) in finished.items():
        path = Path(out)/f'writer-summary-{key}.json'
        item = json.loads(path.read_text()) if path.exists() else None
        details[key] = {'exit_code': workers[key].exitcode, 'timeout': timed_out, 'counts': item}
    success = (failure is None and all(ok for ok, _ in finished.values()) and
               all(d['counts'] is not None and d['counts']['failure'] is None and
                   d['counts']['received'] == d['counts']['written'] for d in details.values()))
    if success:
        merge_joints(out)
    summary = {'streams': details, 'seconds': float(time.monotonic()-started),
               'failure': failure, 'success': success}
    with open(os.path.join(out, 'summary.json'), 'x') as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary), flush=True)
    if not success:
        raise SystemExit('Recorder subscriber failed; per-stream files retained')


if __name__ == '__main__':
    main()
