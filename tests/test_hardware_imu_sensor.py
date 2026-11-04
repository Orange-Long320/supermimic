"""Transport pairing gates are tested without ROS or robot motor interfaces."""
from collections import deque
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('sensor_bridge', Path(__file__).resolve().parents[1]/'scripts/deploy/publish_pelvis_imu_sensor.py')
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


def test_pairing_uses_causal_encoder_samples_and_rejects_stale_or_wrong_domain():
    def row(t, domain=0):
        return dict(stamp_ns=t, domain=domain, position={'j': 1.}, velocity={'j': 0.})
    history = {g: deque([row(1_000_000_000), row(1_004_000_000)]) for g in bridge.GROUPS}
    rows, q, qd = bridge.causal_joints(history, 1_002_000_000, ['j'])
    assert all(r['stamp_ns'] == 1_000_000_000 for r in rows)
    assert q == [1.] and qd == [0.]
    assert bridge.causal_joints(history, 1_050_000_000, ['j']) is None
    assert bridge.causal_joints(history, 999_000_000, ['j']) is None
    history['leg'][0]['domain'] = 1
    with pytest.raises(ValueError):
        bridge.causal_joints(history, 1_002_000_000, ['j'])


def test_mount_requires_fixed_pelvis_and_known_axes(tmp_path):
    path = tmp_path/'robot.urdf'
    path.write_text('<robot><joint name="imu_in_pelvis_joint" type="fixed"><parent link="pelvis"/><origin xyz=".02 0 .04" rpy="0 0 0"/></joint></robot>')
    assert bridge.pelvis_mount(path) == [.02, 0., .04]
    path.write_text(path.read_text().replace('rpy="0 0 0"', 'rpy="0 0 1"'))
    with pytest.raises(ValueError):
        bridge.pelvis_mount(path)


@pytest.mark.parametrize('fault', [None, 'stale_input', 'timestamp_gap'])
def test_startup_waits_for_post_calibration_encoders_and_keeps_time_guards(tmp_path, monkeypatch, fault):
    """A 20 ms fit must not publish its aged window as the first live frame."""
    import sys
    import json
    import types
    import numpy as np
    import supermimic.deploy.imu_dead_reckoning as core
    import supermimic.deploy.async_sim_diagnostic_log as diagnostic
    from types import SimpleNamespace as NS

    clock = NS(value=100.)
    monkeypatch.setattr(bridge, 'time', NS(time=lambda: clock.value, monotonic=lambda: clock.value))
    names = ['leg_j', 'waist_j', 'head_j', 'arm_j']
    profile = tmp_path/'profile.json'
    profile.write_text(json.dumps({'joint_names': names}))
    urdf = tmp_path/'robot.urdf'
    urdf.write_text('<robot><joint name="imu_in_pelvis_joint" type="fixed"><parent link="pelvis"/><origin xyz="0 0 0"/></joint></robot>')
    out = tmp_path/'run'
    callbacks, packets, updates = {}, [], []
    calibration_end = []
    stream = []
    for i in range(1751):
        wall = 100.+i*.002
        stamp = int(round((wall-.004)*1e9))
        for group, name in zip(bridge.GROUPS, names):
            # Oldest joint is 17 ms old when the IMU callback begins.
            jt = stamp-13_000_000
            stream.append((wall, '/aima/hal/joint/'+group+'/state', NS(
                header=NS(stamp=NS(sec=jt//10**9, nanosec=jt%10**9)),
                state=NS(value=0), joints=[NS(name=name, position=0., velocity=0.)])))
        if fault == 'stale_input' and i == 1700:
            wall += .04
        if fault == 'timestamp_gap' and i == 1700:
            stamp += 40_000_000
        stream.append((wall, '/aima/hal/imu/torso/state', NS(
            header=NS(stamp=NS(sec=stamp//10**9, nanosec=stamp%10**9), frame_id='base_link'),
            angular_velocity=NS(x=0., y=0., z=0.), linear_acceleration=NS(x=0., y=0., z=9.81))))
    iterator = iter(stream)

    class Node:
        def __init__(self, *args, **kwargs):
            pass

        def create_subscription(self, kind, topic, callback, qos):
            callbacks[topic] = callback

        def create_publisher(self, *args):
            return NS(publish=lambda msg: packets.append(json.loads(msg.data)))

        def destroy_node(self):
            pass

    def spin_once(node, timeout_sec):
        try:
            wall, topic, msg = next(iterator)
        except StopIteration:
            raise KeyboardInterrupt()
        clock.value = max(clock.value, wall)
        callbacks[topic](msg)

    class Estimator:
        def __init__(self, *args, **kwargs):
            self.filter = NS(initialize=lambda calibration: None)

        def update(self, stamp, gyro, acc):
            updates.append(stamp)

        def state(self, joints):
            return np.array([0., 0., .8]), np.array([1., 0., 0., 0.]), np.zeros(3), np.array([0., 0., -1.])

    def fit(samples, **kwargs):
        # The measured sensors are fresh on entry, but no longer form a fresh
        # packet after the CPU work. Keep the actual packet validator in place.
        calibration_end.append(int(round((99.996+samples[-1]['time_s'])*1e9)))
        clock.value += .020
        return {'gyro_bias': [0., 0., 0.], 'quat_wxyz': [1., 0., 0., 0.]}

    class Log:
        def __init__(self, out):
            pass

        def emit(self, *args):
            pass

        def close(self):
            return {'success': True}

    modules = {
        'rclpy': dict(init=lambda: None, ok=lambda: True, spin_once=spin_once, try_shutdown=lambda: None),
        'rclpy.node': dict(Node=Node),
        'rclpy.qos': dict(QoSProfile=lambda **kw: kw, ReliabilityPolicy=NS(BEST_EFFORT=1), DurabilityPolicy=NS(VOLATILE=1)),
        'sensor_msgs.msg': dict(Imu=object), 'aimdk_msgs.msg': dict(JointStateArray=object),
        'std_msgs.msg': dict(String=lambda **kw: NS(**kw)),
    }
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(core, 'EncoderImuState', Estimator)
    monkeypatch.setattr(core, 'fit_static_initialization', fit)
    monkeypatch.setattr(diagnostic, 'AsyncDiagnosticLog', Log)
    monkeypatch.setattr(sys, 'argv', ['bridge', '--profile', str(profile), '--model', 'fake-model',
                                    '--urdf', str(urdf), '--out', str(out), '--assume-initial-rest'])
    if fault:
        with pytest.raises(ValueError, match='IMU'):
            bridge.main()
    else:
        bridge.main()
    assert packets
    assert packets[0]['state_stamp_ns'] > calibration_end[0]
    assert min(packets[0]['joint_stamps_ns']) > calibration_end[0]
    assert abs(updates[0]-3.) < .003  # Integration still bootstraps at calibration end.
    assert all(b > a for a, b in zip(updates, updates[1:]))
    summary = json.loads((out/'summary.json').read_text())
    assert summary['startup_waiting_fresh'] > 0
    assert bool(summary['fault']) == bool(fault)
