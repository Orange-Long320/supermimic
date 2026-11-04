"""Warmup must finish before control peers are awaited or targets are published."""
import importlib.util
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace as NS
import numpy as np
import pytest


@pytest.mark.parametrize('zero_horizontal', [False, True])
@pytest.mark.parametrize('peers_arrive', [True, False])
def test_worker_announces_warm_models_before_it_can_create_target_publisher(tmp_path, monkeypatch, peers_arrive, zero_horizontal):
    release = tmp_path/'release'
    (release/'fk-only').mkdir(parents=True)
    (release/'dataset').mkdir()
    startup = {'root_position': [0, 0, .8], 'root_quat_wxyz': [1, 0, 0, 0], 'position': [0.]}
    (release/'startup-candidate.json').write_text(json.dumps({'joint_names': ['j'], 'startup': startup}))
    fixture = tmp_path/'fixture.json'
    fixture.write_text(json.dumps({'obs': [0.], 'goal': [0.], 'expected': [0.]}))
    ready = tmp_path/'worker-warm.json'
    plan = dict(release_path=str(release), joint_names=['j'], candidate_sha256='sha',
                profile_sha256='sha', dataset_manifest_sha256='sha', policy_backend='numpy',
                numpy_policy_path='test.npz', numpy_policy_sha256='sha',
                policy_fixture_path=str(fixture), policy_fixture_sha256='sha')
    clock = NS(now=0.)
    calls = {'infer': 0, 'goal': 0, 'pub': 0, 'spins': 0}

    class Policy:
        def __init__(self, *args, **kwargs):
            pass

        def infer(self, *args):
            calls['infer'] += 1
            return np.array([0.])

    class Goal:
        def __init__(self, *args, **kwargs):
            calls['goal_options'] = kwargs

        def goal(self, *args):
            calls['goal'] += 1
            return np.array([0.])

    class Node:
        def __init__(self, *args, **kwargs):
            pass

        def count_publishers(self, topic):
            if topic.endswith('control_target'):
                return 0
            return int(peers_arrive and calls['spins'] > 0)

        def count_subscribers(self, topic):
            return int(peers_arrive and calls['spins'] > 0)

        def create_publisher(self, *args):
            assert json.loads(ready.read_text())['model_warm']
            assert calls['infer'] == 30 and calls['goal'] == 10
            assert peers_arrive and calls['spins'] > 0
            calls['pub'] += 1
            return NS()

        def create_subscription(self, *args):
            pass

        def destroy_node(self):
            pass

    def spin_once(*args, **kwargs):
        assert calls['pub'] == 0
        assert json.loads(ready.read_text())['target_publisher_created'] is False
        clock.now += 1.
        calls['spins'] += 1

    modules = {
        'x2_hardware_publisher': dict(digest=lambda p: 'policy' if p.name == 'policy.pt' else 'sha',
                                    validated_contract=lambda *a, **kw: plan, POLICY_SHA='policy'),
        'supermimic.deploy.early_handover': dict(EarlyHandover=object, Phase=NS(FAULT='fault')),
        'supermimic.deploy.skill_goal': dict(SkillGoal=Goal),
        'supermimic.deploy.numpy_policy': dict(NumpyPolicy=Policy),
        'supermimic.deploy.torch_policy': dict(BufferedTorchPolicy=object),
        'torch': dict(set_num_threads=lambda n: None),
        'rclpy': dict(init=lambda: None, spin_once=spin_once, spin=lambda node: None,
                     try_shutdown=lambda: None),
        'rclpy.node': dict(Node=Node),
        'rclpy.qos': dict(QoSProfile=lambda **kw: kw, ReliabilityPolicy=NS(RELIABLE=1, BEST_EFFORT=2),
                          DurabilityPolicy=NS(VOLATILE=1)),
        'std_msgs.msg': dict(String=object),
    }
    for name, attrs in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[1]/'scripts/deploy/x2_11500_worker.py'
    spec = importlib.util.spec_from_file_location('warmup_worker', path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    monkeypatch.setattr(worker, 'time', NS(time=lambda: 100., monotonic=lambda: clock.now))
    monkeypatch.delenv('ROS_DOMAIN_ID', raising=False)
    monkeypatch.setattr(sys, 'argv', ['worker', '--plan', str(tmp_path/'plan.json'),
                                    '--consent', str(tmp_path/'consent.json'),
                                    '--startup-ready-file', str(ready)]+(['--zero-horizontal-anchor-feedback'] if zero_horizontal else []))
    if peers_arrive:
        worker.main()
        assert calls['pub'] == 1
    else:
        with pytest.raises(ValueError, match='Expected one state producer'):
            worker.main()
        assert calls['pub'] == 0
    assert json.loads(ready.read_text())['model_warm'] is True
    modes=('stand','turn_left_inplace','turn_right_inplace') if zero_horizontal else ()
    assert calls['goal_options'].get('zero_horizontal_anchor_modes',())==modes
    assert json.loads((tmp_path/'anchor-feedback.json').read_text())['zero_world_horizontal_modes']==list(modes)
