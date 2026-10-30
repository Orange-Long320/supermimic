"""Check a reference pose in the unchanged native X2 floor-contact model."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial import ConvexHull,QhullError


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--frame',type=int,default=-1);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    m=mujoco.MjModel.from_binary_path(str(a.model));d=mujoco.MjData(m)
    with np.load(a.reference,allow_pickle=False) as z:
        root=list(z['body_names']).index('pelvis');n=len(z['joint_pos']);i=a.frame%n
        q=np.r_[z['body_pos_w'][i,root],z['body_quat_w'][i,root],z['joint_pos'][i]].astype(float);reference_speed=float(np.linalg.norm(z['joint_vel'][i]));
    d.qpos[:]=q;d.qvel[:]=0;d.qacc[:]=0;mujoco.mj_inverse(m,d)
    force=d.qfrc_inverse.copy();rootbody=int(m.jnt_bodyid[0]);feet={s:m.body('robot/'+s+'_ankle_roll_link').id for s in ('left','right')};normal={s:0. for s in feet};points=[];counts={s:0 for s in feet}
    for j,c in enumerate(d.contact[:d.ncon]):
        bodies=[int(m.geom_bodyid[int(g)]) for g in c.geom];w=np.zeros(6);mujoco.mj_contactForce(m,d,j,w)
        for side,body in feet.items():
            if body in bodies and any(m.body_rootid[b]!=rootbody for b in bodies) and w[0]>1e-5:
                normal[side]+=float(w[0]);counts[side]+=1;points.append(c.pos[:2].copy())
    margin=None
    try:
        hull=ConvexHull(np.asarray(points));com=d.subtree_com[rootbody,:2];margin=float(np.min(-(hull.equations[:,:2]@com+hull.equations[:,2])/np.linalg.norm(hull.equations[:,:2],axis=1)))
    except (QhullError,ValueError):pass
    joints=m.actuator_trnid[:,0].astype(int);limits=m.jnt_range[joints];v=q[m.jnt_qposadr[joints]]
    result={'reference':str(a.reference),'reference_sha256':hashlib.sha256(a.reference.read_bytes()).hexdigest(),'frame':i,'frames':n,'original_joint_speed_norm_rad_s':reference_speed,'qpos':q.tolist(),'root_force_residual':force[:6].tolist(),'left_right_normal_force_n':normal,'loaded_contact_points':counts,'com_loaded_support_margin_m':margin,'root_up':float(d.xmat[rootbody].reshape(3,3)[2,2]),'max_hard_joint_violation_rad':float(np.maximum(np.maximum(limits[:,0]-v,v-limits[:,1]),0).max()),'scope':'Static inverse-dynamics snapshot only, not proof of closed-loop or physical stability'}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='qpos'},indent=2))
if __name__=='__main__':main()
