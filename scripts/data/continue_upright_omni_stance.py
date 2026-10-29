"""Server contact-preserving continuation from a known balanced crouch.

Every stage remains a pose experiment; complete-motion approval is separate.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['model','start','target','calibrator','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--support-height-m',type=float,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('New output required')
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation
    m=mujoco.MjModel.from_binary_path(str(a.model));d=mujoco.MjData(m)
    source=np.load(a.start)['qpos'].copy(); target=np.load(a.target)['qpos'].copy(); previous=source.copy()
    foot=m.body('robot/left_ankle_roll_link').id
    a.out.mkdir(parents=True);(a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    report={'scope':__doc__,'stages':[],'approved_for_training':False}
    for stage,fraction in enumerate(np.linspace(.1,1.,10),1):
        folder=a.out/f'{stage:02d}';folder.mkdir()
        d.qpos[:]=previous;mujoco.mj_forward(m,d)
        position=d.xpos[foot].copy();rotation=Rotation.from_matrix(d.xmat[foot].reshape(3,3))
        q=source.copy();q[7:]=(1-fraction)*source[7:]+fraction*target[7:]
        d.qpos[:]=q;d.qpos[:3]=0;d.qpos[3:7]=[1,0,0,0];mujoco.mj_forward(m,d)
        rootrot=rotation*Rotation.from_matrix(d.xmat[foot].reshape(3,3)).inv()
        q[3:7]=rootrot.as_quat()[[3,0,1,2]];q[:3]=position-rootrot.apply(d.xpos[foot])
        np.savez_compressed(folder/'initial.npz',state=np.r_[q,np.zeros(m.nv)][None,:])
        knees=[float(np.degrees(q[m.jnt_qposadr[m.joint('robot/'+s+'_knee_joint').id]])) for s in ['left','right']]
        argv=[sys.executable,str(a.calibrator),'--model',str(a.model),'--initial',str(folder/'initial.npz'),
              '--out',str(folder/'calibration'),'--support-height-m',str(a.support_height_m),'--anchor-left-foot',
              '--polish-fixed-pressure','--joint-limit-factor','0.9','--minimum-root-up-z','0.99',
              '--maximum-foot-penetration-m','0.001','--max-trials','1','--max-nfev','100','--leg-adjustments',
              '--maximum-knee-deg',str(max(knees)+1),'--minimum-knee-deg',str(min(knees)-1)]
        (folder/'command.json').write_text(json.dumps(argv,indent=2))
        with (folder/'solver.log').open('w') as log:
            proc=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=60)
        if proc.returncode:raise RuntimeError(f'Stage {stage} failed; inspect solver.log')
        result=json.loads((folder/'calibration/result.json').read_text())['best']
        previous=np.load(folder/'calibration/equilibrium.npz')['qpos'].copy()
        item={'stage':stage,'fraction':float(fraction),'root_residual':result['root_force_residual_normalized'],
              'static_force_feasible':result['static_force_feasible'],'maximum_motor_ratio':result['maximum_motor_ratio'],
              'path':str(folder/'calibration/equilibrium.npz')}
        report['stages'].append(item);(a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(item),flush=True)
    np.savez_compressed(a.out/'equilibrium.npz',qpos=previous)


if __name__=='__main__':main()
