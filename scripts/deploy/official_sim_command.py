"""Simulation-only ROS command publisher for the isolated local official simulator.

Never use with hardware. A missing local target switches this simulator to damping;
this is not a hardware watchdog or a safe physical stopping strategy.
"""
import argparse
import json
import math
import os
import socket
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--sim-only', action='store_true', required=True)
    a = p.parse_args()
    if os.environ.get('ROS_LOCALHOST_ONLY') != '1' or os.environ.get('ROS_DOMAIN_ID') != '179':
        raise RuntimeError('Requires isolated simulation ROS domain 179 and ROS_LOCALHOST_ONLY=1')
    spec = json.loads(a.profile.read_text())
    names = spec['joint_names']
    if spec['robot'] != 'x2_ultra' or len(names) != 31 or len(set(names)) != 31:
        raise ValueError('Expected 31 unique X2 Ultra joints')
    def group(name):
        if name.startswith('head_'): return 'head'
        if name.startswith('waist_'): return 'waist'
        if any(x in name for x in ('hip_', 'knee_', 'ankle_')): return 'leg'
        return 'arm'
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from aimdk_msgs.msg import JointCommandArray, JointCommand

    class SimCommands(Node):
        def __init__(self):
            super().__init__('supermimic_sim_commands')
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(('127.0.0.1', a.port)); self.sock.setblocking(False)
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE)
            self.pubs = {g: self.create_publisher(JointCommandArray, f'/aima/hal/joint/{g}/command', qos)
                               for g in ('leg', 'waist', 'head', 'arm')}
            self.target = None
            self.last = 0.
            self.was_fresh = None
            self.timer = self.create_timer(.002, self.tick)

        def tick(self):
            while True:
                try: raw, _ = self.sock.recvfrom(65535)
                except BlockingIOError: break
                try:
                    packet = json.loads(raw)
                    values = packet['target']
                    if len(values) != len(names) or any(not isinstance(x, (float, int)) or not math.isfinite(x) for x in values):
                        raise ValueError('invalid targets')
                    self.target, self.last = values, time.monotonic()
                except (ValueError, KeyError, TypeError):
                    self.target = None
            if self.target is None:
                return
            fresh = time.monotonic() - self.last < .1
            if fresh != self.was_fresh:
                print(json.dumps({'event': 'target_freshness', 'fresh': fresh,
                                  'monotonic': time.monotonic(),
                                  'target_age_s': time.monotonic() - self.last}), flush=True)
                self.was_fresh = fresh
            for g, publisher in self.pubs.items():
                msg = JointCommandArray()
                for i, name in enumerate(names):
                    if group(name) != g: continue
                    cmd = JointCommand(); cmd.name = name
                    cmd.position = float(self.target[i]) if fresh else 0.
                    cmd.stiffness = float(spec['stiffness'][name]) if fresh else 0.
                    cmd.damping = float(spec['damping'][name]) if fresh else 1.
                    cmd.velocity = cmd.effort = 0.
                    msg.joints.append(cmd)
                publisher.publish(msg)

    rclpy.init(); node = SimCommands()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node(); node.sock.close(); rclpy.try_shutdown()


if __name__ == '__main__': main()
