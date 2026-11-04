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


@pytest.mark.parametrize('fault', [None, 'future_input', 'imu_stopped', 'buffered_110ms', 'timestamp_gap', 'joint_delayed_once', 'joint_delayed_forever', 'transport_delay', 'buffered_delay', 'processing_delay'])
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
    for i in range(1801):
        wall = 100.+i*.002
        stamp = int(round((wall-(.020 if fault == 'transport_delay' else .004))*1e9))
        for group, name in zip(bridge.GROUPS, names):
            # Oldest joint is 17 ms old when the IMU callback begins.
            jt = stamp-(0 if fault == 'transport_delay' else 13_000_000)
            delayed = fault in ('joint_delayed_once', 'joint_delayed_forever') and i >= 1700 and (fault == 'joint_delayed_forever' or i < 1708)
            if delayed and group == 'leg':
                continue
            stream.append((wall, '/aima/hal/joint/'+group+'/state', NS(
                header=NS(stamp=NS(sec=jt//10**9, nanosec=jt%10**9)),
                state=NS(value=0), joints=[NS(name=name, position=0., velocity=0.)])))
        if fault == 'imu_stopped' and i >= 1700:
            continue
        if fault == 'future_input' and i == 1700:
            stamp += 200_000_000
        if fault == 'timestamp_gap' and i == 1700:
            stamp += 40_000_000
        stream.append((wall, '/aima/hal/imu/torso/state', NS(
            header=NS(stamp=NS(sec=stamp//10**9, nanosec=stamp%10**9), frame_id='base_link'),
            angular_velocity=NS(x=0., y=0., z=0.), linear_acceleration=NS(x=0., y=0., z=9.81))))
    pending = deque(stream)

    class Handle:
        def __init__(self, depth):
            self.messages = deque(maxlen=depth)
            self.pointer = id(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def take_message(self, *args):
            return (self.messages.popleft(), None) if self.messages else None

    class Node:
        def __init__(self, name, **kwargs):
            self.context = NS(handle=None)

        def create_subscription(self, kind, topic, callback, qos):
            handle = Handle(qos['depth'])
            callbacks[topic] = handle.messages.append
            return NS(handle=handle, msg_type=kind, raw=False)

        def create_publisher(self, *args):
            return NS(publish=lambda msg: packets.append(json.loads(msg.data)))

        def destroy_node(self):
            pass

    delayed_once = [False]

    def deliver_event():
        if not pending:
            raise KeyboardInterrupt()
        wall, topic, msg = pending.popleft()
        clock.value = max(clock.value, wall)
        callbacks[topic](msg)

    class Estimator:
        def __init__(self, *args, **kwargs):
            self.filter = NS(initialize=lambda calibration: None)

        def update(self, stamp, gyro, acc):
            updates.append(stamp)
            if fault == 'processing_delay' and stamp >= 3.4 and not delayed_once[0]:
                clock.value += .060
                delayed_once[0] = True

        def state(self, joints):
            return np.array([0., 0., .8]), np.array([1., 0., 0., 0.]), np.zeros(3), np.array([0., 0., -1.])

    def fit(samples, **kwargs):
        # The measured sensors are fresh on entry, but no longer form a fresh
        # packet after the CPU work. Keep the actual packet validator in place.
        calibration_end.append(int(round(((99.980 if fault == 'transport_delay' else 99.996)+samples[-1]['time_s'])*1e9)))
        clock.value += 0. if fault == 'transport_delay' else .020
        return {'gyro_bias': [0., 0., 0.], 'quat_wxyz': [1., 0., 0., 0.]}

    class Log:
        def __init__(self, out):
            pass

        def emit(self, *args):
            pass

        def close(self):
            return {'success': True}

    class WaitSet:
        def __init__(self, *args):
            pass

        def clear_entities(self):
            pass

        def add_subscription(self, handle):
            self.handle = handle

        def wait(self, timeout):
            if fault in ('buffered_delay', 'buffered_110ms') and clock.value >= 103.4 and not delayed_once[0]:
                clock.value += .110 if fault == 'buffered_110ms' else .060
                delayed_once[0] = True
            end = clock.value+timeout*1e-9
            while not self.handle.messages:
                if pending and pending[0][0] > end:
                    clock.value = end
                    return
                deliver_event()
            while pending and pending[0][0] <= clock.value:
                deliver_event()

        def get_ready_entities(self, kind):
            return [self.handle.pointer] if self.handle.messages else []

    modules = {
        'rclpy.impl': {},
        'rclpy.impl.implementation_singleton': dict(rclpy_implementation=NS(WaitSet=WaitSet)),
        'rclpy': dict(init=lambda: None, ok=lambda: True, try_shutdown=lambda: None),
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
    fails = fault in ('future_input', 'timestamp_gap', 'imu_stopped')
    if fails:
        with pytest.raises(ValueError, match='IMU|output timeout'):
            bridge.main()
    else:
        bridge.main()
    assert packets
    assert packets[0]['state_stamp_ns'] > calibration_end[0]
    assert min(packets[0]['joint_stamps_ns']) > calibration_end[0]
    assert abs(updates[0]-3.) < .003  # Integration still bootstraps at calibration end.
    assert all(b > a for a, b in zip(updates, updates[1:]))
    summary = json.loads((out/'summary.json').read_text())
    assert bool(summary['fault']) == fails
    assert summary['joints_received'] == summary['joints']
    assert summary['encoder_polls'] > 0
    if not fails:
        assert len(updates) == 301  # Every post-calibration IMU is still integrated.
    if fault == 'joint_delayed_forever':
        assert summary['max_output_silence_s'] > .1
        assert packets[-1]['published_wall'] < 103.45
    if fault in ('buffered_delay', 'buffered_110ms', 'processing_delay'):
        assert summary['max_imu_batch'] > 20
        assert summary['buffered_imu_integrated'] > 0


@pytest.mark.parametrize('failure', ['dead_worker', 'stale_heartbeat', 'interrupt'])
def test_first_command_probe_exits_when_children_or_state_fail(tmp_path, monkeypatch, capsys, failure):
    import sys
    import time
    import json
    import types
    from types import SimpleNamespace as NS
    runner = Path(__file__).resolve().parents[1]/'scripts/deploy/run_x2_keyboard_sensor_20261104.sh'
    code = runner.read_text().split("<<'WAITPY'\n", 1)[1].split('\nWAITPY', 1)[0]
    # Inject process/heartbeat reads; execute the runner's actual probe body.
    code = code.replace('from pathlib import Path', '')
    ready = tmp_path/'state'/'ready.json'
    ready.parent.mkdir()
    ready.write_text(json.dumps({'wall': time.time()-(1 if failure == 'stale_heartbeat' else 0)}))
    spins = []

    def path(*parts):
        if str(parts[0]) == '/proc':
            state = 'Z' if failure == 'dead_worker' and parts[1] == '3' else 'S'
            return NS(read_text=lambda: '3 (python) '+state+' 1 2 3')
        return Path(*parts)

    def spin(*args, **kwargs):
        spins.append(True)
        raise KeyboardInterrupt()

    class Node:
        def __init__(self, *args, **kwargs):
            pass

        def create_subscription(self, *args):
            pass

        def destroy_node(self):
            pass

    for name, attrs in {
        'rclpy': dict(init=lambda: None, spin_once=spin, try_shutdown=lambda: None),
        'rclpy.node': dict(Node=Node), 'aimdk_msgs.msg': dict(JointCommandArray=object),
    }.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, 'argv', ['wait', '1', '2', '3', '4', str(ready)])
    with pytest.raises(SystemExit) as result:
        exec(compile(code, '<first-command-probe>', 'exec'), {'Path': path})
    assert result.value.code == 1
    text = capsys.readouterr().out
    assert {'dead_worker': 'policy worker exited', 'stale_heartbeat': 'heartbeat expired',
            'interrupt': 'Operator interrupted startup'}[failure] in text
    assert bool(spins) == (failure == 'interrupt')


def test_saved_calibration_cannot_be_selected_for_live_state_publication(monkeypatch):
    import sys
    monkeypatch.setattr(sys, 'argv', ['bridge', '--profile', 'profile', '--model', 'model',
                                    '--urdf', 'urdf', '--out', 'out', '--assume-initial-rest',
                                    '--diagnostic-calibration', 'old-calibration.json'])
    with pytest.raises(SystemExit) as exc:
        bridge.main()
    assert exc.value.code == 2
