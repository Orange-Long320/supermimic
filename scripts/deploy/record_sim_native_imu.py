"""Record simulator streams with raw ROS callbacks and separate writer processes.

Each callback enqueues serialized ROS bytes and receive time. Independent workers
perform deserialization and JSON encoding. Best-effort DDS and bounded queues do
not guarantee delivery; compare header sets with an independent subscriber.
"""
import argparse
import heapq
import json
import multiprocessing as mp
import os
import queue
import time
from pathlib import Path


GROUPS = ('leg', 'waist', 'head', 'arm')
STREAMS = ('imu', 'chest', 'odom', *GROUPS)


def write_records(out, stream, records):
    """Deserialize and write outside the ROS callback process."""
    from rclpy.serialization import deserialize_message
    from aimdk_msgs.msg import JointStateArray
    from sensor_msgs.msg import Imu
    from nav_msgs.msg import Odometry
    names = {'imu': 'imu.jsonl', 'chest': 'imu-chest.jsonl', 'odom': 'odom.jsonl'}
    names.update({g: f'joints-{g}.jsonl' for g in GROUPS})
    msg_type = Imu if stream in ('imu', 'chest') else Odometry if stream == 'odom' else JointStateArray
    count = 0
    with open(os.path.join(out, names[stream]), 'x') as handle:
        while True:
            item = records.get()
            if item is None:
                break
            recv, raw = item
            msg = deserialize_message(raw, msg_type)
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
            handle.write(json.dumps(record, allow_nan=False) + '\n')
            count += 1
    with open(os.path.join(out, f'writer-summary-{stream}.json'), 'x') as handle:
        json.dump({'written': count}, handle)


def finish_writer(writer, records, timeout):
    """Bound shutdown, including failed or blocked children and queue feeder."""
    deadline = time.monotonic() + timeout
    sentinel_sent = False
    while writer.is_alive() and time.monotonic() < deadline:
        try:
            records.put(None, timeout=min(.25, max(.001, deadline-time.monotonic())))
            sentinel_sent = True
            break
        except queue.Full:
            pass
    writer.join(timeout=max(0., deadline-time.monotonic()))
    timed_out = writer.is_alive()
    if timed_out:
        writer.terminate()
        writer.join(timeout=2.)
    ok = sentinel_sent and not timed_out and writer.exitcode == 0
    if not ok:
        records.cancel_join_thread()
    records.close()
    if ok:
        records.join_thread()
    return ok, timed_out


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

    import rclpy
    from rclpy.node import Node
    from rclpy.executors import ExternalShutdownException
    from rclpy.impl.implementation_singleton import rclpy_implementation
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from aimdk_msgs.msg import JointStateArray
    from sensor_msgs.msg import Imu
    from nav_msgs.msg import Odometry

    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=False)
    queues = {key: mp.Queue(maxsize=8192) for key in STREAMS}
    workers = {key: mp.Process(target=write_records, args=(out, key, queues[key]),
                               name='native-json-' + key) for key in STREAMS}
    for worker in workers.values():
        worker.start()
    counts = {key: 0 for key in STREAMS}
    overflow = {key: 0 for key in STREAMS}

    class Recorder(Node):
        def __init__(self):
            super().__init__('supermimic_native_imu_recorder')
            qos = QoSProfile(depth=400, reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE)
            topics = {'imu': (Imu, '/aima/hal/imu/torso/state'),
                      'chest': (Imu, '/aima/hal/imu/chest/state'),
                      'odom': (Odometry, '/aima/hal/odom/state')}
            topics.update({g: (JointStateArray, f'/aima/hal/joint/{g}/state') for g in GROUPS})
            self.subs = [self.create_subscription(kind, topic,
                         lambda raw, key=key: self.receive(key, raw), qos, raw=True)
                         for key, (kind, topic) in topics.items()]

        def receive(self, key, raw):
            counts[key] += 1
            try:
                queues[key].put_nowait((time.monotonic(), raw))
            except queue.Full:
                overflow[key] += 1

    rclpy.init(); node = Recorder(); started = time.monotonic(); failure = None
    try:
        while time.monotonic() - started < a.seconds:
            rclpy.spin_once(node, timeout_sec=.2)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except rclpy_implementation.RCLError as exc:
        if rclpy.ok():
            failure = repr(exc)
    except Exception as exc:
        failure = repr(exc)
    finally:
        node.destroy_node(); rclpy.try_shutdown()
        deadline = time.monotonic() + 30.
        finished = {key: finish_writer(workers[key], queues[key], max(0., deadline-time.monotonic()))
                    for key in STREAMS}
    written = {}
    for key, (ok, _) in finished.items():
        path = Path(out)/f'writer-summary-{key}.json'
        written[key] = json.loads(path.read_text())['written'] if ok and path.exists() else None
    unwritten = {key: (None if written[key] is None else counts[key]-overflow[key]-written[key])
                 for key in STREAMS}
    success = failure is None and all(ok for ok, _ in finished.values()) and not any(overflow.values()) and all(v == 0 for v in unwritten.values())
    if success:
        merge_joints(out)
    summary = {'received_counts': counts, 'queue_overflow': overflow, 'written_counts': written,
               'unwritten_after_enqueue': unwritten, 'writer_timeout': {k: x[1] for k,x in finished.items()},
               'writer_exit_code': {k: workers[k].exitcode for k in STREAMS},
               'seconds': float(time.monotonic()-started), 'failure': failure, 'success': success}
    with open(os.path.join(out, 'summary.json'), 'x') as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary), flush=True)
    if not success:
        raise SystemExit('Recorder lost data or writer failed; per-stream files retained')


if __name__ == '__main__':
    main()
