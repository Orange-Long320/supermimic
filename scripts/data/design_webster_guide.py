"""Create an interior, feet-flat preparation guide for CPU motion search.

This is a kinematic guide, not an executable reference and not training data.
Only a separately simulated, independently validated trajectory may be trained.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--ready-config",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--margin",type=float,default=.15)
    args=p.parse_args()
    from supermimic.data.motion_convert import load_g1_model,convert_csv_to_npz
    from supermimic.robots.dimensions import PROFILE
    m=load_g1_model();d=mujoco.MjData(m)
    z=np.load(args.source);root=int(m.jnt_bodyid[0])
    r=z["body_names"].tolist().index(m.body(root).name)
    old=np.c_[z["body_pos_w"][:,r],z["body_quat_w"][:,r],z["joint_pos"]].astype(float)
    ready=np.array(json.loads(args.ready_config.read_text())["initial_qpos"])
    assert ready.shape==(m.nq,) and np.isfinite(ready).all()
    q=old.copy();takeoff,landing=30,55
    c=m.jnt_range[1:].mean(axis=1);h=.45*np.diff(m.jnt_range[1:],axis=1).ravel()
    q[:,7:]=np.clip(q[:,7:],c-h+args.margin,c+h-args.margin)
    for j,name in enumerate(PROFILE["joint_names"]):
        if name.startswith("head_") or "wrist_" in name or name.startswith("waist_"):
            q[:,j+7]=ready[j+7]
    r0=Rotation.from_quat(ready[[4,5,6,3]])
    for i in range(takeoff+1):
        s=i/takeoff;w=s*s*(3-2*s)
        q[i,:3]=ready[:3]*(1-w)+old[i,:3]*w
        q[i,7:]=ready[7:]*(1-w)+q[i,7:]*w
        ri=Rotation.from_quat(old[i,[4,5,6,3]])
        q[i,3:7]=(r0*Rotation.from_rotvec((r0.inv()*ri).as_rotvec()*w)).as_quat()[[3,0,1,2]]
    feet=[]
    for g in range(m.ngeom):
        if m.geom(g).name.startswith("foot_"):
            k=m.geom_dataid[g];a,n=m.mesh_vertadr[k],m.mesh_vertnum[k]
            feet.append((g,m.mesh_vert[a:a+n].copy()))
    def fk(row):
        d.qpos[:]=row;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        sole=min((v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])[:,2].min() for g,v in feet)
        return d.subtree_com[root].copy(),float(sole)
    # Geometry is only a guide here: it is never used as a physical certificate.
    for i in list(range(takeoff+1))+list(range(landing,len(q))):
        q[i,2]+=.005-fk(q[i])[1]
    a,b=fk(q[takeoff])[0],fk(q[landing])[0]
    duration=(landing-takeoff)/50
    velocity=(b-a)/duration+np.array([0,0,.5*9.81*duration])
    for i in range(takeoff+1,landing):
        t=(i-takeoff)/50
        desired=a+t*velocity-np.array([0,0,.5*9.81*t*t])
        q[i,:3]+=desired-fk(q[i])[0]
    args.out.mkdir(parents=True,exist_ok=False)
    np.savetxt(args.out/"webster.csv",np.c_[q[:,:3],q[:,[4,5,6,3]],q[:,7:]],delimiter=",",fmt="%.10f")
    convert_csv_to_npz(args.out/"webster.csv",args.out/"webster.npz",50,50)
    report={"purpose":"KINEMATIC GUIDE ONLY for exact CPU search; never automatically train this guide",
        "approved_for_training":False,"source":str(args.source.resolve()),
        "source_sha256":hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "ready_config":str(args.ready_config.resolve()),"ready_sha256":hashlib.sha256(args.ready_config.read_bytes()).hexdigest(),
        "initial_pose":"project neutral crouch with collision-sole normals flattened within soft limits",
        "initial_velocity":"smooth preparation blend, not a pose copied from a moving video",
        "joint_margin_rad":args.margin,"wrists_head_waist":"neutral guide", "takeoff_frame":takeoff,"landing_frame":landing,
        "remaining":"Support, motor effort, angular momentum and full flip/landing must be generated and checked in actual project physics."}
    (args.out/"GUIDE_ONLY.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=="__main__":main()
