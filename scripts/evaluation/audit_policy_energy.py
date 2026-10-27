"""Measure work and energy from recorded native physical rollouts."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--physics',type=Path,nargs='+',required=True)
    p.add_argument('--until',type=float,default=.8)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    m=mujoco.MjModel.from_binary_path(str(args.model));d=mujoco.MjData(m);root=int(m.jnt_bodyid[0])
    assert m.nu==m.nv-6 and np.allclose(m.actuator_gear[:,0],1)
    assert np.array_equal(m.jnt_dofadr[m.actuator_trnid[:,0]],np.arange(6,m.nv))
    mass=float(m.body_mass.sum());groups={k:[] for k in ['left_leg','right_leg','arms','waist']}
    for i in range(m.nu):
        name=m.joint(int(m.actuator_trnid[i,0])).name.rsplit('/',1)[-1]
        group=('left_leg' if name.startswith('left_') else 'right_leg') if any(x in name for x in ['hip_','knee_','ankle_']) else 'waist' if name.startswith('waist_') else 'arms'
        groups[group].append(i)
    def state_energy(q,v):
        d.qpos[:]=q;d.qvel[:]=v;mujoco.mj_forward(m,d);mujoco.mj_energyPos(m,d);mujoco.mj_energyVel(m,d)
        jac=np.zeros((3,m.nv));angular=np.zeros_like(jac)
        mujoco.mj_jacSubtreeCom(m,d,jac,root);mujoco.mj_angmomMat(m,d,angular,root)
        com_velocity=jac@v
        return {'potential_j':float(d.energy[0]),'kinetic_j':float(d.energy[1]),
                'vertical_kinetic_j':float(.5*mass*com_velocity[2]**2),
                'horizontal_kinetic_j':float(.5*mass*np.dot(com_velocity[:2],com_velocity[:2])),
                'com_velocity':com_velocity.tolist(),'angular_momentum':(angular@v).tolist(),
                'com_height_m':float(d.subtree_com[root,2])}
    reports=[]
    for path in args.physics:
        with np.load(path) as z:
            q=z['qpos'].copy();v=z['qvel'].copy();force=z['actuator_force'].copy()
            initial=z['initial_qpos'].copy();initial_v=z['initial_qvel'].copy();fps=float(z['fps'].reshape(-1)[0])
        if q.shape!=(len(q),m.nq) or v.shape!=(len(q),m.nv) or force.shape!=(len(q),m.nu):raise ValueError('Model/trace dimensions differ')
        count=min(len(q),round(args.until*fps));start=state_energy(initial,initial_v);e0=start['potential_j']+start['kinetic_j']
        previous=initial;positive=np.zeros(m.nu);negative=np.zeros(m.nu);rows=[]
        for i in range(count):
            work=force[i]*(q[i,7:]-previous[7:]);positive+=np.maximum(work,0);negative+=np.minimum(work,0)
            energy=state_energy(q[i],v[i]);net=float((positive+negative).sum())
            energy.update(time_s=(i+1)/fps,positive_motor_work_j=float(positive.sum()),negative_motor_work_j=float(negative.sum()),
                          net_motor_work_j=net,unattributed_energy_change_j=energy['potential_j']+energy['kinetic_j']-e0-net)
            rows.append(energy);previous=q[i]
        reports.append({'physics':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'initial':start,'rows':rows,
                        'group_work_j':{k:{'positive':float(positive[idx].sum()),'negative':float(negative[idx].sum())} for k,idx in groups.items()}})
    result={'model':str(args.model),'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),'mass_kg':mass,
            'runs':reports,'approved_for_training':False,
            'limitations':'Motor work is discrete applied torque times joint displacement. Remaining energy change combines passive damping, contact and limit work, and integration error; it is not assigned solely to impacts.'}
    (args.out/'result.json').write_text(json.dumps(result,indent=2));(args.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps([{k:v for k,v in r.items() if k not in ['rows','initial']}|{'final':r['rows'][-1]} for r in reports]),flush=True)


if __name__=='__main__':main()
