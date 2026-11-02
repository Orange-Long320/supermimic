"""Verify a staged 11500 package and inventory ROS publishers; no commands."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.request import urlopen


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.release.resolve()
    if args.out.exists():
        raise ValueError('New report required')
    manifest = json.loads((root/'release-manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or sha(path) != expected:
            raise ValueError('Package integrity failure: '+name)
    expected_policy = '4424f231fc1896da098b2fca366e8522df6791596638b2b9257ba682520bbe4c'
    if sha(root/'policy.pt') != expected_policy:
        raise ValueError('Wrong 11500 policy')
    if sha(root/'dataset/manifest.json') != '40a2a31c1b84160f2ea6aa26548eb966f2e3c9ec23e10fa46ec1e9b2532b5b7d':
        raise ValueError('Wrong selected reference manifest')
    dataset = json.loads((root/'dataset/manifest.json').read_text())
    for clip in dataset['clips']:
        if sha(root/'dataset'/(clip['name']+'.npz')) != clip['sha256']:
            raise ValueError('Reference clip integrity failure')
    import rclpy
    from rclpy.node import Node
    rclpy.init()
    node = Node('supermimic_candidate_inventory', enable_rosout=False, start_parameter_services=False)
    try:
        end = time.monotonic()+3.
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=.05)
        publishers = {f'/aima/hal/joint/{group}/command':
                      node.count_publishers(f'/aima/hal/joint/{group}/command')
                      for group in ('leg', 'waist', 'head', 'arm')}
        try:
            with urlopen('http://10.0.1.40:50080/json/get_all_apps_info', timeout=3) as response:
                mc = json.load(response)['data']['mc']
        except Exception as exc:
            mc = {'query_error': str(exc)}
        result = dict(verified_wall=time.time(), package_integrity_passed=True,
                      policy_sha256=expected_policy, reference_clips_verified=len(dataset['clips']),
                      native_mc=mc, motor_command_publishers=publishers,
                      motor_output_enabled=False, control_transferred=False)
        args.out.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
