"""Bounded isolated integration runner; always reap its worker and publisher.

Run inside PC2 lxh. Domain 197 and synthetic topic prefix are mandatory.
No hardware plan or hardware command topic is accepted.
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--prefix', required=True)
    p.add_argument('--hold-active-seconds', type=float, default=60.)
    a = p.parse_args()
    if os.environ.get('ROS_DOMAIN_ID') != '197' or not re.fullmatch(
            '/supermimic/isolated_[a-z0-9_]+', a.prefix):
        raise ValueError('Requires isolated domain and prefix')
    if not .2 <= a.hold_active_seconds <= 180.:
        raise ValueError('Expected bounded hold up to 180 seconds')
    if (a.folder/'fixtures-ready').exists():
        raise ValueError('New test folder required')
    children = []
    logs = []
    try:
        def launch(name, command):
            log = (a.folder/(name+'.log')).open('x')
            logs.append(log)
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            children.append(child)
            return child

        probe = launch('probe', [sys.executable, str(a.folder/'tests/isolated_x2_worker_probe.py'),
            '--folder', str(a.folder), '--release', str(a.release), '--prefix', a.prefix,
            '--hold-active-seconds', str(a.hold_active_seconds)])
        deadline = time.monotonic()+20.
        while not (a.folder/'fixtures-ready').exists():
            if probe.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError('Synthetic fixture setup failed')
            time.sleep(.05)
        for name, script in (('publisher', 'x2_hardware_publisher.py'), ('worker', 'x2_11500_worker.py')):
            launch(name, [sys.executable, str(a.folder/'scripts/deploy'/script),
                '--plan', str(a.folder/'synthetic-plan.json'),
                '--consent', str(a.folder/'synthetic-consent.json'), '--isolated-prefix', a.prefix])
        code = probe.wait(timeout=a.hold_active_seconds+45.)
        if code:
            raise RuntimeError('Isolated integration failed; see probe/worker logs')
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
        for child in reversed(children):
            try:
                child.wait(timeout=5.)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5.)
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
