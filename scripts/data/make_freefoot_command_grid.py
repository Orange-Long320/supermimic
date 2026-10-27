"""Create command-only warm starts that lift the free foot before backswing."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['model','initial-physics','preparation','configuration','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--lift',type=float,default=.05)
    p.add_argument('--scales',type=float,nargs='+',default=[0,.5,1,1.5,2,3,4,6])
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    from supermimic.robots.dimensions import PROFILE
    m=mujoco.MjModel.from_binary_path(str(args.model));d=mujoco.MjData(m)
    with np.load(args.initial_physics) as z:q0=z['initial_qpos'].copy()
    with np.load(args.preparation) as z:parameters=z['parameters'].copy()
    groups=json.loads(args.configuration.read_text())['groups'];count=len(groups);base=1+3*count
    if parameters.shape==(base,):parameters=np.r_[parameters,np.zeros(count)]
    if parameters.shape!=(base+count,):raise ValueError('Expected preparation with optional initial residual knot')
    joint_names=['right_hip_pitch_joint','right_knee_joint','right_ankle_pitch_joint','right_ankle_roll_joint']
    joints=[m.joint('robot/'+n).id for n in joint_names];columns=m.jnt_qposadr[joints]
    foot=m.body('robot/right_ankle_roll_link').id;d.qpos[:]=q0;mujoco.mj_kinematics(m,d)
    rotation=Rotation.from_matrix(d.xmat[foot].reshape(3,3));meshes=[]
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g])!=foot or not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
        k=m.geom_dataid[g];a,n=m.mesh_vertadr[k],m.mesh_vertnum[k];meshes.append((g,m.mesh_vert[a:a+n].copy()))
    def sole_height(x):
        d.qpos[:]=q0;d.qpos[columns]=x;mujoco.mj_kinematics(m,d)
        return min(float((v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])[:,2].min()) for g,v in meshes)
    height=sole_height(q0[columns])+args.lift
    lower=m.jnt_range[joints,0]+.025;upper=m.jnt_range[joints,1]-.025
    weights=np.array([1.,.5,2.,2.])
    fit=minimize(lambda x:float(np.dot(weights,(x-q0[columns])**2)),np.clip(q0[columns],lower,upper),
                 bounds=list(zip(lower,upper)),constraints=[{'type':'eq','fun':lambda x:sole_height(x)-height}],
                 method='SLSQP',options={'maxiter':200,'ftol':1e-12})
    error=float(sole_height(fit.x)-height)
    angular=(rotation.inv()*Rotation.from_matrix(d.xmat[foot].reshape(3,3))).as_rotvec()
    if not fit.success or abs(error)>1e-5:raise ValueError('Free-leg sole-height lift did not fit: '+str(error))
    deltas=fit.x-q0[columns];group_delta=np.zeros(count)
    for name,delta in zip(joint_names,deltas):
        group=next(i for i,g in enumerate(groups) if len(g)==1 and g[0][0]==name and g[0][1]==1)
        group_delta[group]=delta*PROFILE['stiffness'][name]/PROFILE['effort_limit'][name]
    candidates=[]
    for scale in args.scales:
        candidate=parameters.copy();candidate[base:]+=scale*group_delta;candidate[1:1+count]+=scale*.75*group_delta
        candidate[1:]=np.clip(candidate[1:],-1.5,1.5);candidates.append(candidate)
    np.savez(args.out/'parameter-grid.npz',parameters=np.asarray(candidates))
    report={'purpose':__doc__,'lift_m':args.lift,'joint_names':joint_names,'kinematic_joint_delta_rad':deltas.tolist(),
            'kinematic_height_error_m':error,'freefoot_orientation_change_rad':angular.tolist(),
            'scales':args.scales,'state_edits_after_start':False,
            'actual_initial_state_changed':False,'approved_for_training':False,
            'note':'IK is only a direction for motor commands. Actual motion, contact and joint limits require physical evaluation.',
            'source_sha256':{k:hashlib.sha256(getattr(args,k).read_bytes()).hexdigest() for k in ['model','initial_physics','preparation','configuration']}}
    (args.out/'configuration.json').write_text(json.dumps(report,indent=2));(args.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
