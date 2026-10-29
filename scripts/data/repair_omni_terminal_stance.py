"""Bounded isolated terminal-stance experiment. Never replaces training data.

Fits sole headings/support width/COM, then native contact equilibrium, and
compares two-second motor-only holds. No claim about a complete transition.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attribution', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--template', type=Path, help='Previously verified stance; rigidly place at terminal support, no joint fitting')
    p.add_argument('--template-support-height-m', type=float)
    p.add_argument('--feedforward-mode',choices=['inverse','none'],default='inverse',help='Motor-only hold diagnostic: inverse feedforward or original PD alone')
    p.add_argument('--hold-seconds',type=float,default=2.)
    a = p.parse_args()
    if a.out.exists():
        p.error('New output required')
    if a.template and a.template_support_height_m is None:
        p.error('Template support height required')
    if not 2<=a.hold_seconds<=10:p.error('Hold duration must be 2–10 seconds')
    import mujoco
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation
    a.out.mkdir(parents=True)
    (a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    rows = json.loads(a.attribution.read_text())
    report = {'scope': __doc__, 'approved_for_training': False, 'tasks': {}}
    for name, row in rows.items():
        m = mujoco.MjModel.from_binary_path(row['model']); d = mujoco.MjData(m)
        base = np.load(row['reference'])['qpos'][-1].copy()
        root = int(m.jnt_bodyid[0]); feet = [m.body('robot/'+s+'_ankle_roll_link').id for s in ['left','right']]
        ids = [j for j in range(1,m.njnt) if any(k in m.joint(j).name for k in ['hip_', 'knee_', 'ankle_', 'waist_'])]
        cols = np.r_[np.arange(6),m.jnt_dofadr[ids]].astype(int)
        qcols = m.jnt_qposadr[ids]
        mid = m.jnt_range[ids].mean(1); half = np.diff(m.jnt_range[ids],axis=1).ravel()*.45
        lower = np.r_[[-.15]*3,[-.35]*3,np.maximum(mid-half-base[qcols],-.6)]
        upper = np.r_[[.15]*3,[.35]*3,np.minimum(mid+half-base[qcols],.6)]
        weight = float(m.body_mass.sum())*9.81
        geoms = [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]]==root and (m.geom_contype[g] or m.geom_conaffinity[g])]
        terrain = [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]]!=root and (m.geom_contype[g] or m.geom_conaffinity[g])]
        footgeom = [next(g for g in geoms if m.geom_bodyid[g]==b) for b in feet]
        meshes = {}
        for g in geoms:
            k = m.geom_dataid[g]
            if k>=0:
                start,n = m.mesh_vertadr[k],m.mesh_vertnum[k]
                meshes[g] = m.mesh_vert[start:start+n].copy()
        def place(x):
            q = base.copy(); v = np.zeros(m.nv); v[cols]=x
            mujoco.mj_integratePos(m,q,v,1.); d.qpos[:]=q; d.qvel[:]=0; d.qacc[:]=0
            mujoco.mj_forward(m,d)
            return q
        place(np.zeros(len(cols)))
        heading = float(np.arctan2(d.xmat[root].reshape(3,3)[1,0], d.xmat[root].reshape(3,3)[0,0]))
        forward=np.array([np.cos(heading),np.sin(heading),0.]); lateral=np.cross([0,0,1],forward)
        targets=d.xpos[feet].copy(); width=float((targets[0]-targets[1])@lateral)
        widening=max(0.,.20-width)
        targets[0]+=lateral*widening/2; targets[1]-=lateral*widening/2
        targetrot=[Rotation.from_euler('z',heading+sign*np.radians(5)) for sign in [1,-1]]
        floors=[]
        for g in footgeom:
            w=np.zeros(6); o=min(terrain,key=lambda o:mujoco.mj_geomDistance(m,d,g,o,10.,w))
            floors.append(float(m.geom_pos[o,2]+m.geom_size[o,2]) if m.geom_type[o]==mujoco.mjtGeom.mjGEOM_BOX else 0.)
        caps=np.maximum(abs(m.actuator_forcerange).max(1),1e-8)
        actj=m.actuator_trnid[:,0].astype(int); ac=m.jnt_qposadr[actj]; av=m.jnt_dofadr[actj]
        def shape(x, physical=False):
            q=place(x)
            footr=[]; corners=[]; height=[]
            for i,(b,g) in enumerate(zip(feet,footgeom)):
                vertices=meshes[g]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
                bottom=vertices[:,2].min(); height.append(bottom-floors[i])
                corners.extend(vertices[vertices[:,2]<bottom+.003,:2])
                footr.extend((d.xpos[b,:2]-targets[i,:2])* (20 if physical else 100))
                footr.extend((targetrot[i].inv()*Rotation.from_matrix(d.xmat[b].reshape(3,3))).as_rotvec()*(1 if physical else 5))
            poly=ConvexHull(np.array(corners)).equations
            com=d.subtree_com[root].copy()
            balance=np.maximum(poly[:,:2]@com[:2]+poly[:,2]+.02,0)*40
            # Keep residual size fixed: use COM-to-centre instead of hull faces.
            balance=(com[:2]-targets[:,:2].mean(0))* (5 if physical else 30)
            clearance=[]
            for g in geoms:
                for o in terrain:
                    dist=mujoco.mj_geomDistance(m,d,g,o,.02,np.zeros(6))
                    clearance.append(max(-.0009-dist,0)*200)
            up=Rotation.from_matrix(d.xmat[root].reshape(3,3)).as_euler('xyz')[:2]
            r=np.r_[footr,balance,(np.array(height)+.0005)*200,clearance,up*2,x*.03]
            if physical:
                d.qacc[:]=0; mujoco.mj_inverse(m,d)
                r=np.r_[r,d.qfrc_inverse[:3]/weight*40,d.qfrc_inverse[3:6]/(weight*.7)*40,
                        np.maximum(abs(d.qfrc_inverse[av])/caps-.9,0)*10]
            return r
        x=np.clip(np.zeros(len(cols)),lower+1e-9,upper-1e-9)
        if a.template:
            if abs(floors[0]-floors[1])>1e-8:raise ValueError('Template needs one horizontal support plane')
            candidate=np.load(a.template)['qpos'].copy()
            if candidate.shape!=base.shape:raise ValueError('Template dimensions differ')
            d.qpos[:]=candidate; mujoco.mj_forward(m,d)
            oldcentre=d.xpos[feet].mean(0).copy()
            oldyaw=float(np.arctan2(d.xmat[root].reshape(3,3)[1,0],d.xmat[root].reshape(3,3)[0,0]))
            rot=Rotation.from_euler('z',heading-oldyaw)
            candidate[:3]=rot.apply(candidate[:3]-oldcentre)+oldcentre
            candidate[3:7]=(rot*Rotation.from_quat(candidate[[4,5,6,3]])).as_quat()[[3,0,1,2]]
            candidate[:2]+=targets[:,:2].mean(0)-oldcentre[:2]
            candidate[2]+=floors[0]-a.template_support_height_m
            geometric=candidate.copy(); fit=physical=None
        else:
            fit=least_squares(shape,x,bounds=(lower,upper),max_nfev=160,ftol=1e-9,xtol=1e-9,gtol=1e-9)
            geometric=place(fit.x).copy()
            physical=least_squares(lambda x:shape(x,True),fit.x,bounds=(lower,upper),max_nfev=200,
                                   diff_step=1e-5,ftol=1e-9,xtol=1e-9,gtol=1e-9)
            candidate=place(physical.x).copy()
        output=a.out/name; output.mkdir()
        np.savez_compressed(output/'terminal-candidates.npz',original=base,geometric=geometric,candidate=candidate)

        def hold(q, label):
            actual=mujoco.MjData(m); desired=mujoco.MjData(m)
            actual.qpos[:]=q; desired.qpos[:]=q; mujoco.mj_inverse(m,desired)
            kp=m.actuator_gainprm[:,0]; feed=desired.qfrc_inverse[av]
            if a.feedforward_mode=='none':feed=np.zeros_like(feed)
            states=[]; maxhard=0.; maxpenetration=0.; minup=1.; drift=0.
            for step in range(round(a.hold_seconds/m.opt.timestep)):
                actual.ctrl[:]=q[ac]+feed/kp
                mujoco.mj_step(m,actual); mujoco.mj_forward(m,actual)
                states.append(np.r_[actual.qpos.copy(),actual.qvel.copy()])
                maxhard=max(maxhard,float(np.maximum(np.maximum(m.jnt_range[1:,0]-actual.qpos[7:],actual.qpos[7:]-m.jnt_range[1:,1]),0).max()))
                minup=min(minup,float(actual.xmat[root].reshape(3,3)[2,2]))
                drift=max(drift,float(np.linalg.norm(actual.qpos[:3]-q[:3])))
                maxpenetration=max([maxpenetration]+[-float(c.dist) for c in actual.contact[:actual.ncon]])
                if minup<.3 or not np.isfinite(actual.qpos).all():break
            np.savez_compressed(output/f'{label}-hold.npz',state=np.asarray(states),dt=m.opt.timestep)
            return dict(seconds=len(states)*m.opt.timestep,min_root_up=minup,max_root_drift_m=drift,
                        maximum_hard_violation_rad=maxhard,maximum_contact_penetration_m=maxpenetration,
                        narrow_hold_gate_passed=bool(len(states)*m.opt.timestep>=a.hold_seconds-1e-9 and minup>.9 and drift<.03 and maxhard<1e-6 and maxpenetration<.005),
                        scope=f'Motor PD, feedforward={a.feedforward_mode}, terminal initialized; not learned policy, full task, or transition acceptance')
        task={'model_sha256':hashlib.sha256(Path(row['model']).read_bytes()).hexdigest(),
              'reference_sha256':hashlib.sha256(Path(row['reference']).read_bytes()).hexdigest(),
              'template':str(a.template) if a.template else None,
              'feedforward_mode':a.feedforward_mode,
              'geometric_solver_success':bool(fit.success) if fit else None,'physical_solver_success':bool(physical.success) if physical else None,
              'geometric_nfev':fit.nfev if fit else 0,'physical_nfev':physical.nfev if physical else 0,
              'original_hold':hold(base,'original'),'candidate_hold':hold(candidate,'candidate'),
              'max_joint_change_rad':float(abs(candidate[7:]-base[7:]).max()),
              'transition_verified':False,'approved_for_training':False}
        report['tasks'][name]=task
        (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({name:task}),flush=True)


if __name__=='__main__':
    main()
