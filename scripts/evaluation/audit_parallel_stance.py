"""Inspect saved native stance holds for parallel calves and physical limits."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['attribution','repair','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('New output required')
    import mujoco
    import numpy as np
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra',check_assets=True)
    from supermimic.robots.dimensions import PROFILE
    report={'scope':__doc__,'full_transition_verified':False,'approved_for_training':False,'tasks':{}}
    modes=json.loads((a.repair/'report.json').read_text())['tasks']
    for name,row in json.loads(a.attribution.read_text()).items():
        folder=a.repair/name;m=mujoco.MjModel.from_binary_path(row['model']);d=mujoco.MjData(m)
        q=np.load(folder/'terminal-candidates.npz')['candidate'];z=np.load(folder/'candidate-hold.npz');states=z['state'];dt=float(z['dt'])
        root=int(m.jnt_bodyid[0]);d.qpos[:]=q;mujoco.mj_forward(m,d)
        axis=d.xmat[root].reshape(3,3)[:,0];yaw=np.arctan2(axis[1],axis[0]);lateral=np.array([-np.sin(yaw),np.cos(yaw),0.])
        initial=mujoco.MjData(m);initial.qpos[:]=q;mujoco.mj_inverse(m,initial)
        joint=m.actuator_trnid[:,0].astype(int);qcols=m.jnt_qposadr[joint];vcols=m.jnt_dofadr[joint]
        ctrl=q[qcols]+initial.qfrc_inverse[vcols]/m.actuator_gainprm[:,0]
        if modes[name].get('feedforward_mode','inverse')=='none':ctrl=q[qcols].copy()
        speed=np.array([PROFILE['velocity_limit'][m.joint(j).name.removeprefix('robot/')] for j in joint])
        cap=np.maximum(abs(m.actuator_forcerange).max(1),1e-10)
        values=[];maxhard=0.;maxpen=0.;motor=0.;speedratio=0.;minup=1.;maxdrift=0.;terminal_com=[];jac=np.zeros((3,m.nv))
        for i,state in enumerate(np.vstack([np.r_[q,np.zeros(m.nv)],states])):
            d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:];d.ctrl[:]=ctrl;mujoco.mj_forward(m,d)
            vectors=[];front=[]
            for side in ['left','right']:
                vector=d.xpos[m.body('robot/'+side+'_knee_link').id]-d.xpos[m.body('robot/'+side+'_ankle_roll_link').id]
                vectors.append(vector/np.linalg.norm(vector));front.append(float(np.degrees(np.arctan2(vector@lateral,vector[2]))))
            angle=float(np.degrees(np.arccos(np.clip(vectors[0]@vectors[1],-1.,1.))))
            values.append([angle,*front])
            maxhard=max(maxhard,float(np.maximum(np.maximum(m.jnt_range[1:,0]-d.qpos[7:],d.qpos[7:]-m.jnt_range[1:,1]),0).max()))
            maxpen=max([maxpen]+[-float(c.dist) for c in d.contact[:d.ncon]])
            motor=max(motor,float((abs(d.actuator_force)/cap).max()));speedratio=max(speedratio,float((abs(d.qvel[vcols])/speed).max()))
            minup=min(minup,float(d.xmat[root].reshape(3,3)[2,2]));maxdrift=max(maxdrift,float(np.linalg.norm(d.qpos[:3]-q[:3])))
            if i*dt>=1.5:
                mujoco.mj_jacSubtreeCom(m,d,jac,root);terminal_com.append(float(np.linalg.norm(jac@d.qvel)))
        values=np.asarray(values);pair=float(values[:,0].max());frontal=float(abs(values[:,1:]).max())
        physics=bool(len(states)*dt>=2 and maxhard<1e-6 and maxpen<.001 and motor<=1+1e-8 and speedratio<=1 and minup>.99 and maxdrift<.01 and max(terminal_com,default=1)<.01)
        posture=bool(pair<2 and frontal<2)
        report['tasks'][name]={'seconds':len(states)*dt,'initial_calf_pair_angle_deg':values[0,0],
            'initial_frontal_angles_deg':values[0,1:].tolist(),'maximum_calf_pair_angle_deg':pair,
            'maximum_absolute_frontal_angle_deg':frontal,'maximum_hard_violation_rad':maxhard,
            'maximum_contact_penetration_m':maxpen,'maximum_motor_ratio':motor,'maximum_joint_speed_ratio':speedratio,
            'minimum_root_up':minup,'maximum_root_drift_m':maxdrift,'terminal_com_speed_m_s':max(terminal_com,default=None),
            'physical_hold_passed':physics,'parallel_calves_passed':posture,'combined_passed':physics and posture}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
    a.out.with_suffix('.as-run.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(report,indent=2))


if __name__=='__main__':main()
