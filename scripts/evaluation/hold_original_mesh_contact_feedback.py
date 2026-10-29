"""Original X2 meshes only: feedback linearized along planted-foot motions.

The derivative queries preserve foot poses to avoid differentiating a switch
of collision features. Actual execution is unconstrained native simulation:
only original joint motors act, with no state resets or external forces.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['attribution','targets','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--task')
    p.add_argument('--seconds',type=float,default=5.)
    p.add_argument('--foot-attitude-feedback',action='store_true',help='Close the compliant foot rotation modes using the original ankle gains')
    a=p.parse_args()
    if a.out.exists() or not 2<=a.seconds<=10:p.error('New output and 2–10 seconds required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Explicit CPU-only server execution required')
    import mujoco
    import numpy as np
    from scipy.linalg import null_space,block_diag
    from scipy.spatial.transform import Rotation
    a.out.mkdir(parents=True);(a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    report={'scope':__doc__,'physical_geometry_changed':False,'tasks':{}}
    for name,row in json.loads(a.attribution.read_text()).items():
        if a.task and name!=a.task:continue
        folder=a.out/name;folder.mkdir();model_path=Path(row['model']);m=mujoco.MjModel.from_binary_path(str(model_path))
        feet=[m.body('robot/'+s+'_ankle_roll_link').id for s in ['left','right']]
        for b in feet:
            geoms=[g for g in range(m.ngeom) if m.geom_bodyid[g]==b and (m.geom_contype[g] or m.geom_conaffinity[g])]
            if len(geoms)!=1 or m.geom_type[geoms[0]]!=mujoco.mjtGeom.mjGEOM_MESH:raise ValueError('Original foot mesh required; diagnostic boxes rejected')
        target=a.targets/name/'terminal-candidates.npz';(folder/'terminal-candidates.npz').write_bytes(target.read_bytes());q=np.load(target)['candidate']
        d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d);anchors=[(d.xpos[b].copy(),Rotation.from_matrix(d.xmat[b].reshape(3,3))) for b in feet]
        def foot_jac(data):
            result=[]
            for b in feet:
                jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacBody(m,data,jp,jr,b);result.extend([jp,jr])
            return np.vstack(result)
        J=foot_jac(d);N=null_space(J,rcond=1e-10);nr=N.shape[1];T=block_diag(N.T,N.T)
        joint=m.actuator_trnid[:,0].astype(int);qc=m.jnt_qposadr[joint];vc=m.jnt_dofadr[joint]
        ankle_gains=[]
        for side in ['left','right']:
            indices=[int(np.where(joint==m.joint('robot/'+side+'_ankle_'+axis+'_joint').id)[0][0]) for axis in ['roll','pitch']]
            ankle_gains.append((np.r_[m.actuator_gainprm[indices,0],0.],np.r_[-m.actuator_biasprm[indices,2],0.]))
        d.qacc[:]=0;mujoco.mj_inverse(m,d);nominal=q[qc]+d.qfrc_inverse[vc]/m.actuator_gainprm[:,0]
        maximum_anchor_error=0.
        def plant(z=None,du=None):
            nonlocal maximum_anchor_error
            data=mujoco.MjData(m);data.qpos[:]=q;data.ctrl[:]=nominal
            if z is not None:
                mujoco.mj_integratePos(m,data.qpos,N@z[:nr],1.)
                # This projection is for derivative queries only, never for
                # the actual execution state below.
                for _ in range(4):
                    mujoco.mj_kinematics(m,data);mujoco.mj_comPos(m,data)
                    error=np.concatenate([np.r_[position-data.xpos[b],(rotation*Rotation.from_matrix(data.xmat[b].reshape(3,3)).inv()).as_rotvec()] for b,(position,rotation) in zip(feet,anchors)])
                    if abs(error).max()<1e-13:break
                    velocity=np.linalg.lstsq(foot_jac(data),error,rcond=1e-10)[0];mujoco.mj_integratePos(m,data.qpos,velocity,1.)
                maximum_anchor_error=max(maximum_anchor_error,float(abs(error).max()))
                data.qvel[:]=N@z[nr:]
            if du is not None:data.ctrl[:]+=du
            mujoco.mj_forward(m,data);mujoco.mj_step(m,data);return data
        A=np.zeros((2*nr,2*nr));eps=1e-4;tangent=np.zeros(m.nv)
        for j in range(2*nr):
            z=np.zeros(2*nr);z[j]=eps;plus=plant(z);minus=plant(-z)
            mujoco.mj_differentiatePos(m,tangent,2*eps,minus.qpos,plus.qpos)
            A[:,j]=np.r_[N.T@tangent,N.T@((plus.qvel-minus.qvel)/(2*eps))]
        rawB=np.zeros((2*m.nv,m.nu));linear=mujoco.MjData(m);linear.qpos[:]=q;linear.ctrl[:]=nominal;mujoco.mj_forward(m,linear)
        mujoco.mjd_transitionFD(m,linear,1e-4,True,None,rawB,None,None);B=T@rawB
        Q=block_diag(N.T@np.diag(np.r_[np.full(3,2000.),np.full(3,1000.),np.full(m.nv-6,100.)])@N,
                     N.T@np.diag(np.r_[np.full(6,20.),np.full(m.nv-6,.2)])@N);R=np.eye(m.nu);P=Q.copy()
        for _ in range(1000):
            K=np.linalg.solve(R+B.T@P@B,B.T@P@A);P=Q+A.T@P@(A-B@K);P=(P+P.T)/2
        K=np.linalg.solve(R+B.T@P@B,B.T@P@A);radius=float(abs(np.linalg.eigvals(A-B@K)).max())
        if not np.isfinite(K).all():raise ValueError('Nonfinite feedback')
        np.savez_compressed(folder/'feedback.npz',N=N,A=A,B=B,K=K,nominal_control=nominal)
        actual=mujoco.MjData(m);actual.qpos[:]=q;actual.ctrl[:]=nominal;mujoco.mj_forward(m,actual)
        states=[];controls=[];rows=[];root=int(m.jnt_bodyid[0]);status='completed'
        for i in range(round(a.seconds/m.opt.timestep)):
            mujoco.mj_differentiatePos(m,tangent,1.,q,actual.qpos)
            actual.ctrl[:]=nominal-K@(T@np.r_[tangent,actual.qvel]);controls.append(actual.ctrl.copy())
            if a.foot_attitude_feedback:
                torque=np.zeros(m.nu)
                for b,(_,wanted),(stiffness,damping) in zip(feet,anchors,ankle_gains):
                    jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacBody(m,actual,jp,jr,b)
                    current=Rotation.from_matrix(actual.xmat[b].reshape(3,3));basis=wanted.as_matrix()
                    error=basis.T@(wanted*current.inv()).as_rotvec();velocity=basis.T@(jr@actual.qvel)
                    wrench=basis@(stiffness*error-damping*velocity)
                    torque+=jr[:,vc].T@wrench
                actual.ctrl[:]+=torque/m.actuator_gainprm[:,0]
                controls[-1]=actual.ctrl.copy()
            mujoco.mj_step(m,actual);mujoco.mj_forward(m,actual);states.append(np.r_[actual.qpos.copy(),actual.qvel.copy()])
            hard=float(np.maximum(np.maximum(m.jnt_range[1:,0]-actual.qpos[7:],actual.qpos[7:]-m.jnt_range[1:,1]),0).max());drift=float(np.linalg.norm(actual.qpos[:3]-q[:3]));up=float(actual.xmat[root].reshape(3,3)[2,2])
            rows.append(dict(time_s=(i+1)*m.opt.timestep,hard=hard,drift=drift,up=up))
            if hard>.001 or drift>.15 or up<.9 or not np.isfinite(states[-1]).all():status='safety_stop';break
        np.savez_compressed(folder/'candidate-hold.npz',state=np.asarray(states),controls=np.asarray(controls),dt=m.opt.timestep)
        item={'status':status,'seconds':len(states)*m.opt.timestep,'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
              'physical_geometry_changed':False,'query_foot_anchor_error':maximum_anchor_error,'reduced_state_dimension':2*nr,
              'foot_attitude_feedback':a.foot_attitude_feedback,
              'linear_radius':radius,'A_max':float(abs(A).max()),'K_max':float(abs(K).max()),'max_hard_violation_rad':max(r['hard'] for r in rows),
              'max_root_drift_m':max(r['drift'] for r in rows),'min_root_up':min(r['up'] for r in rows),'feedforward_mode':'inverse','approved_for_training':False}
        report['tasks'][name]=item;(a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({name:item}),flush=True)


if __name__=='__main__':main()
