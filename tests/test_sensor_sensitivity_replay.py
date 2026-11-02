import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('sensitivity_replay',
    Path(__file__).resolve().parents[1]/'scripts/experiments/replay_sim_sensor_sensitivity.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class NeverSupported:
    def __init__(self, _):
        pass
    def observe(self, *_):
        return None, 'hold'


def test_no_output_and_timestamp_gap_are_both_reported():
    t=np.array([0.,.04]); rot=np.repeat(np.eye(3)[None,:,:],2,axis=0)
    zeros=np.zeros((2,3)); acc=np.tile([0.,0.,9.81],(2,1))
    result,rows=replay.replay({},t,rot,zeros,acc,np.array([0.]),np.zeros((1,1)),
        np.zeros((1,1)),['joint'],np.array([-.1,.2]),np.zeros((2,3)),
        'unused',NeverSupported,.2,1)
    assert rows == []
    assert result['failure'] == 'no usable output'
    assert result['runtime_failure'] == 'imu_timestamp_gap'
    assert result['early_failure']
    assert abs(result['failure_detail']['reported_interval_ms']-40.) < 1e-9
