"""Compile only the numerical header: no ROS, hardware or physics simulation."""
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from supermimic.deploy.imu_dead_reckoning import ImuDeadReckoning


def test_native_point_propagation_matches_python_and_rejects_bad_time(tmp_path):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler unavailable')
    header = Path(__file__).resolve().parents[1]/'scripts/deploy/imu_raw_reader'
    source = tmp_path/'check.cpp'
    source.write_text('''#include "point_integrator.hpp"
#include <iostream>
#include <iomanip>
int main(){
 PointIntegrator f; PointIntegrator::V b; std::array<double,4> q;
 for(auto& v:b)std::cin>>v;for(auto& v:q)std::cin>>v;f.initialize(b,q);
 int64_t t;PointIntegrator::V g,a;
 std::cout<<std::setprecision(17);
 while(std::cin>>t>>g[0]>>g[1]>>g[2]>>a[0]>>a[1]>>a[2]){
  try {f.update(t,g,a);}catch(const std::exception& e){std::cerr<<e.what();return 2;}
  for(double v:f.q)std::cout<<v<<' ';for(double v:f.x)std::cout<<v<<' ';
  for(double v:f.gyro)std::cout<<v<<' ';std::cout<<'\\n';
 }
}
''')
    exe=tmp_path/'check'
    subprocess.run([compiler,'-std=c++17','-O2','-I',str(header),str(source),'-o',str(exe)],check=True)
    rng=np.random.default_rng(431)
    bias=np.array([.002,-.004,.001])
    quat=Rotation.from_euler('xyz',[.23,-.18,.4]).as_quat()[[3,0,1,2]]
    cal=dict(gyro_bias=bias.tolist(),quat_wxyz=quat.tolist())
    f=ImuDeadReckoning(interval_average=False,point_trapezoid=True,max_gap_seconds=.04)
    f.initialize(cal)
    stamps=10**9+np.cumsum(rng.integers(1800000,2200000,size=5000))
    gyro=rng.normal(0,.2,(5000,3))+bias
    acc=rng.normal(0,.3,(5000,3))+[1.,-.4,9.6]
    lines=[' '.join(map(str,[*bias,*quat]))]
    expected=[]
    for t,g,a in zip(stamps,gyro,acc):
        lines.append(' '.join(map(str,[t,*g,*a])))
        f.update((int(t)-int(stamps[0]))*1e-9,g,a)
        expected.append([*f.rotation.as_quat()[[3,0,1,2]],*f.x,*f.gyro])
    result=subprocess.run([str(exe)],input='\n'.join(lines),text=True,capture_output=True,check=True)
    actual=np.array([list(map(float,l.split())) for l in result.stdout.splitlines()])
    np.testing.assert_allclose(actual,expected,atol=2e-10,rtol=2e-10)
    for delta in [0,-1,40000001]:
        bad=lines[:2]+[' '.join(map(str,[int(stamps[0])+delta,*gyro[0],*acc[0]]))]
        result=subprocess.run([str(exe)],input='\n'.join(bad),text=True,capture_output=True)
        assert result.returncode==2 and 'gap' in result.stderr
