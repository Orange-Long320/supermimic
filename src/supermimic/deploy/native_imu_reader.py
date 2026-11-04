"""Bounded raw-IMU pipe from the SDK reader; no ROS or motor APIs in this module."""
import os
import select
import struct
import subprocess
from types import SimpleNamespace as NS

FRAME = struct.Struct('<Qqq6d64s31d31d4q4q4iII')
MAGIC = 0x5832524157303032


def decode_frame(data):
    values = FRAME.unpack(data)
    magic, stamp, received, gx, gy, gz, ax, ay, az, frame = values[:10]
    if magic != MAGIC:
        raise ValueError('Raw IMU frame magic/layout mismatch')
    stamps, receives, domains = values[72:76], values[76:80], values[80:84]
    groups, joints = values[84:86]
    if any(groups & (1<<g) and domains[g] != 0 for g in range(4)):
        raise ValueError('Joint state domain is not position')
    snapshot = None
    if groups == 15 and joints == 0x7fffffff:
        if any(not 0 <= stamp-t <= 25_000_000 for t in stamps):
            raise ValueError('Native snapshot has future/stale encoders')
        rows = [dict(stamp_ns=t, recv=r*1e-9, domain=d) for t,r,d in zip(stamps,receives,domains)]
        snapshot = rows, list(values[10:41]), list(values[41:72])
    sec, ns = divmod(stamp, 10**9)
    return NS(header=NS(stamp=NS(sec=sec, nanosec=ns), frame_id=frame.split(b'\0', 1)[0].decode()),
              angular_velocity=NS(x=gx, y=gy, z=gz), linear_acceleration=NS(x=ax, y=ay, z=az),
              transport_received_wall_ns=received, native_joint_snapshot=snapshot)


class NativeImuReader:
    def __init__(self, executable, log_path, joint_names):
        if len(joint_names) != 31 or len(set(joint_names)) != 31 or any(',' in n for n in joint_names):
            raise ValueError("Expected 31 unique profile joints")
        self.log = log_path.open('x')
        self.buffer = bytearray()
        self.frames = 0
        env = dict(os.environ, RCUTILS_LOGGING_USE_STDOUT='0')
        try:
            self.process = subprocess.Popen([str(executable), ','.join(joint_names)], stdout=subprocess.PIPE,
                                            stderr=self.log, env=env, bufsize=0)
        except BaseException:
            self.log.close()
            raise

    def read_batch(self, timeout=.005):
        fd = self.process.stdout.fileno()
        if not select.select([fd], [], [], timeout)[0]:
            return []
        data = os.read(fd, FRAME.size*64)
        if not data:
            raise RuntimeError('Raw IMU reader exited or pipe closed; inspect imu-reader.log')
        self.buffer.extend(data)
        count = len(self.buffer)//FRAME.size
        result = [decode_frame(self.buffer[i*FRAME.size:(i+1)*FRAME.size]) for i in range(count)]
        del self.buffer[:count*FRAME.size]
        self.frames += count
        return result

    def close(self):
        self.process.stdout.close()
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=2.)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2.)
        self.log.close()
        return {'frames': self.frames, 'returncode': self.process.returncode}
