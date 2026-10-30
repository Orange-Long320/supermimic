#!/usr/bin/env python3
"""Read-only deployment inventory; never sends robot commands or starts ROS nodes."""
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def command(argv):
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=5, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {'error': type(exc).__name__}
    return {'returncode': proc.returncode, 'stdout': proc.stdout.strip(), 'stderr': proc.stderr.strip()}


def main():
    os_release = Path('/etc/os-release')
    report = {'scope': 'read-only inventory; not a deployment or safety validation',
              'machine': platform.machine(), 'platform': platform.platform(),
              'python': sys.version, 'executable': sys.executable,
              'os_release': os_release.read_text() if os_release.exists() else None,
              'ros2_path': shutil.which('ros2'), 'packages': {}}
    for module in ('torch', 'numpy', 'scipy', 'mujoco', 'rclpy', 'aimdk_msgs'):
        # Inspect module metadata without importing ROS or initializing devices.
        from importlib.util import find_spec
        try:
            spec = find_spec(module)
            report['packages'][module] = {'available': spec is not None,
                                          'origin': spec.origin if spec else None}
        except (ImportError, ValueError) as exc:
            report['packages'][module] = {'error': str(exc)}
    report['aimdk_msgs_ros_prefix'] = command(['ros2', 'pkg', 'prefix', 'aimdk_msgs'])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
