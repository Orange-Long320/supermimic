"""Unreset native stance diagnostic with inverse-dynamics nominal motor targets."""
import argparse,json
from pathlib import Path
import numpy as np,mujoco


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--pose',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--seconds',type=float,default=5.);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    m=mujoco.MjModel.from_binary_path(str(a.model));d=mujoco.MjData(m);q=np.load(a.pose)['qpos'];d.qpos[:]=q;d.qacc[:]=0;d.qvel[:]=0;mujoco.mj_inverse(m,d)
    joint=m.actuator_trnid[:,0].astype(int);qc=m.jnt_qposadr[joint];vc=m.jnt_dofadr[joint];target=q[qc]+d.qfrc_inverse[vc]/m.actuator_gainprm[:,0]
    d.ctrl[:]=target;mujoco.mj_forward(m,d);states=[];status='completed';root=int(m.jnt_bodyid[0]);hardmax=0.;upmin=1.;driftmax=0.
    for step in range(round(a.seconds/m.opt.timestep)):
        mujoco.mj_step(m,d);mujoco.mj_forward(m,d);states.append(np.r_[d.qpos.copy(),d.qvel.copy()]);qj=d.qpos[qc];lim=m.jnt_range[joint];hard=float(np.maximum(np.maximum(lim[:,0]-qj,qj-lim[:,1]),0).max());up=float(d.xmat[root].reshape(3,3)[2,2]);drift=float(np.linalg.norm(d.qpos[:3]-q[:3]));hardmax=max(hardmax,hard);upmin=min(upmin,up);driftmax=max(driftmax,drift)
        if not np.isfinite(states[-1]).all() or hard>.001 or up<.9 or drift>.15:status='safety_stop';break
    a.out.mkdir(parents=True);np.savez(a.out/'trace.npz',states=np.array(states),dt=m.opt.timestep,target=target)
    report={'status':status,'seconds':len(states)*m.opt.timestep,'max_joint_hard_violation_rad':hardmax,'minimum_root_up':upmin,'max_root_displacement_m':driftmax,'state_resets_after_initialization':0,'controller':'fixed joint servo targets plus inverse-dynamics nominal torques','policy_evaluation':False,'approved_for_training':False}
    (a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
