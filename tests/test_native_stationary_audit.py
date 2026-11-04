import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('stationary_audit',Path(__file__).resolve().parents[1]/'scripts/evaluation/audit_native_stationary_imu.py')
audit_module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


@pytest.mark.parametrize('wire_size',[720,832])
def test_stationary_audit_calibrates_initial_window_and_evaluates_later_data(tmp_path,wire_size):
    m=audit_module
    dtype=m.DTYPE if wire_size==832 else np.dtype([(n,m.DTYPE.fields[n][0]) for n in m.DTYPE.names[:12]])
    data=np.zeros(4001,dtype=dtype)
    data['magic']=m.MAGIC if wire_size==832 else m.MAGIC-1
    data['stamp']=1793770000000000000+np.arange(len(data),dtype=np.int64)*2000000
    data['received']=data['stamp']+1000000
    data['joint_stamps']=data['stamp'][:,None]
    data['joint_received']=data['received'][:,None]
    data['groups']=15;data['joints']=0x7fffffff
    bias=[.003,-.004,.002]
    data['imu'][:,:3]=bias;data['imu'][:,5]=9.81
    data.tofile(tmp_path/'raw.bin')
    (tmp_path/'capture.json').write_text(json.dumps(dict(operator_declared_stationary=True,wire_size=wire_size)))
    result=m.audit(tmp_path)
    assert result['samples']==4001 and result['gaps_above_3ms']==0
    assert result['evaluation_seconds']==4.
    np.testing.assert_allclose(result['calibration']['gyro_bias'],bias,atol=1e-14)
    assert result['variants']['calibrated']['max_xy_displacement_m']<1e-10
    assert result['variants']['without_gyro_bias_subtraction']['max_xy_displacement_m']>.1
