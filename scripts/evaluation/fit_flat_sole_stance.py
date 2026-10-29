"""Server stance calibration against actual maximal sole-hull facets."""
import argparse
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['model','initial','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--nested-pressure',action='store_true',help='Solve total vertical load first, then balance root moments')
    p.add_argument('--bracket-moments',action='store_true',help='Use bracketed moment roots instead of local contact Jacobians')
    p.add_argument('--hip-pitch-domain-deg',type=float,default=5.7295779513,help='Symmetric hip-pitch search radius; candidate joint limits are checked separately')
    p.add_argument('--global-moments',action='store_true',help='Bounded derivative-free search when contact branches break local moment roots')
    a=p.parse_args()
    if a.out.exists():p.error('New output required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Explicit CPU execution required')
    import mujoco
    import numpy as np
    from scipy.optimize import least_squares,brentq,differential_evolution,minimize
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation
    m=mujoco.MjModel.from_binary_path(str(a.model));d=mujoco.MjData(m);q0=np.load(a.initial)['qpos'].copy();normals={};meshes={};sides=['left','right']
    def column(side,name):return int(m.jnt_qposadr[m.joint('robot/'+side+'_'+name+'_joint').id])
    for side in sides:
        b=m.body('robot/'+side+'_ankle_roll_link').id
        g=next(g for g in range(m.ngeom) if m.geom_bodyid[g]==b and (m.geom_contype[g] or m.geom_conaffinity[g]))
        mesh=m.geom_dataid[g];start,n=m.mesh_vertadr[mesh],m.mesh_vertnum[mesh]
        vertices=m.mesh_vert[start:start+n]@Rotation.from_quat(m.geom_quat[g][[1,2,3,0]]).as_matrix().T+m.geom_pos[g]
        meshes[b]=vertices
        hull=ConvexHull(vertices);tri=vertices[hull.simplices];areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
        indices=np.where(hull.equations[:,2]<-.95)[0];chosen=indices[np.argmax(areas[indices])];normals[side]=hull.equations[chosen,:3].copy()
        np.testing.assert_allclose(m.body_quat[b],[1,0,0,0],atol=1e-12)
        np.testing.assert_allclose(m.jnt_axis[m.joint('robot/'+side+'_ankle_pitch_joint').id],[0,1,0],atol=1e-12)
        np.testing.assert_allclose(m.jnt_axis[m.joint('robot/'+side+'_ankle_roll_joint').id],[1,0,0],atol=1e-12)
    hip=float(np.mean([q0[column(s,'hip_pitch')] for s in sides]));weight=float(m.body_mass.sum())*9.81
    pitch_domain=np.radians(a.hip_pitch_domain_deg)/.1
    def bottom(q):
        d.qpos[:]=q;mujoco.mj_kinematics(m,d)
        return min(float((v@d.xmat[b].reshape(3,3).T+d.xpos[b])[:,2].min()) for b,v in meshes.items())
    reference_bottom=bottom(q0)
    def pose(x):
        q=q0.copy()
        for side in sides:
            q[column(side,'hip_pitch')]=hip+x[1]*.1;q[column(side,'hip_roll')]=x[2]*.02
            q[column(side,'ankle_pitch')]=q[column(side,'ankle_roll')]=0.
        d.qpos[:]=q;mujoco.mj_kinematics(m,d)
        for side in sides:
            b=m.body('robot/'+side+'_ankle_roll_link').id;target=d.xmat[b].reshape(3,3).T@np.array([0.,0.,-1.]);n=normals[side]
            roll=np.arcsin(np.clip(target[1]/np.hypot(n[1],n[2]),-1,1))-np.arctan2(n[1],-n[2])
            v=Rotation.from_rotvec([roll,0,0]).apply(n);pitch=np.arctan2(v[0],-v[2])-np.arctan2(target[0],-target[2])
            q[column(side,'ankle_roll')]=roll;q[column(side,'ankle_pitch')]=pitch
        # Hip changes alter leg height. Solve compression relative to the
        # actual sole, rather than restricting root height to a fixed millimetre.
        q[2]+=reference_bottom-bottom(q)+x[0]*.001
        return q
    def force(x):
        q=pose(x);d.qpos[:]=q;d.qvel[:]=0;d.qacc[:]=0;mujoco.mj_inverse(m,d)
        return d.qfrc_inverse[:6].copy()
    def residual(x):
        f=force(x);return f[[2,3,4]]/np.array([weight,weight*.7,weight*.7])
    def jac(x):
        result=[]
        for i in range(3):
            dx=np.zeros(3);dx[i]=1e-3;result.append((residual(x+dx)-residual(x-dx))/.002)
        return np.stack(result,axis=1)
    if a.nested_pressure:
        def pressure(y):
            z=brentq(lambda z:force(np.r_[z,y])[2],-3.,1.,xtol=1e-10,maxiter=100)
            return np.r_[z,y]
        def moments(y):return force(pressure(y))[[3,4]]/(weight*.7)
        def moment_jac(y):
            result=[]
            for i in range(2):
                dy=np.zeros(2);dy[i]=1e-3;result.append((moments(y+dy)-moments(y-dy))/.002)
            return np.stack(result,axis=1)
        if a.global_moments:
            def cost(y):return float(np.square(moments(y)).sum())
            global_fit=differential_evolution(cost,[(-pitch_domain,pitch_domain),(-1.,1.)],maxiter=50,popsize=8,seed=0,polish=False,tol=1e-10)
            fit=minimize(cost,global_fit.x,method='Nelder-Mead',bounds=[(-pitch_domain,pitch_domain),(-1.,1.)],options={'maxiter':200,'xatol':1e-10,'fatol':1e-16})
            x=pressure(fit.x)
        elif a.bracket_moments:
            def lateral(pitch):return brentq(lambda roll:moments(np.array([pitch,roll]))[0],-1.,1.,xtol=1e-9,maxiter=100)
            brackets=[]
            for trial in np.linspace(-pitch_domain,pitch_domain,13):
                try:
                    roll=lateral(trial);value=moments(np.array([trial,roll]))[1]
                    brackets.append((float(trial),float(value)))
                except ValueError:continue
            pair=next(((u[0],v[0]) for u,v in zip(brackets,brackets[1:]) if u[1]*v[1]<=0),None)
            if pair is None:raise ValueError(f'No coupled moment bracket within posture bounds: {brackets}')
            pitch=brentq(lambda pitch:moments(np.array([pitch,lateral(pitch)]))[1],*pair,xtol=1e-9,maxiter=100)
            x=pressure(np.array([pitch,lateral(pitch)]));fit=None
        else:
            fit=least_squares(moments,np.zeros(2),jac=moment_jac,bounds=([-1,-1],[1,1]),max_nfev=60,ftol=1e-12,xtol=1e-12,gtol=1e-12)
            x=pressure(fit.x)
    else:
        fit=least_squares(residual,np.zeros(3),jac=jac,bounds=([-2,-1,-1],[2,1,1]),max_nfev=150,ftol=1e-13,xtol=1e-13,gtol=1e-13)
        x=fit.x
    q=pose(x);f=force(x);a.out.mkdir(parents=True);np.savez_compressed(a.out/'equilibrium.npz',qpos=q)
    report={'configuration':{k:str(v) for k,v in vars(a).items()},'facet_normals':{s:n.tolist() for s,n in normals.items()},
            'x':x.tolist(),'solver_success':bool(fit.success) if fit is not None else None,'nfev':fit.nfev if fit is not None else None,'root_wrench':f.tolist(),'approved_for_training':False}
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');(a.out/'as-run.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(report),flush=True)


if __name__=='__main__':main()
