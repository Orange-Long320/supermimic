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
