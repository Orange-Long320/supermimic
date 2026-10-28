"""Small-step contact-dynamics QPs with native hard-geometry acceptance.

Dynamics residual variables are offline optimization errors, never applied root
forces. A candidate is not approved without dense independent physical checks
and uninterrupted actual controller execution. Time and original caps stay fixed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['reference','model','scene','out','solver-python','solver-script']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--start-frame',type=int,required=True);p.add_argument('--end-frame',type=int,required=True)
    p.add_argument('--iterations',type=int,default=12)
    p.add_argument('--solver-backend',choices=['osqp','clarabel'],default='osqp')
    p.add_argument('--restore-feasibility',action='store_true',help='Explicit offline phaseI: minimize reported motor violations at unchanged caps; never a physical pass')
    a=p.parse_args()
    if a.out.exists() or not 1<=a.iterations<=20:p.error('New output and at most20 SQP iterations required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only diagnostic required')
    a.out.mkdir(parents=True);(a.out/'pid').write_text(str(os.getpid()))
    import mujoco
    import numpy as np
    import scipy.sparse as sp
    from scipy.optimize import nnls
    from scipy.spatial.transform import Rotation
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra',check_assets=True)
    from supermimic.data.physics_motion import differentiate_qpos
    from omni_contact_window_model import ContactWindow
    helper=ContactWindow(a.model,a.reference,a.scene,a.start_frame,a.end_frame)
    m,d=helper.m,helper.d;start,end=helper.start,helper.end;qs=helper.source.copy()
    count=end-start+1;n=end-start-1;pose_size=n*37
    sizes=[4*len(points) for points in helper.contacts];offsets=np.r_[pose_size,pose_size+np.cumsum(sizes)].astype(int)
    residual_offset=int(offsets[-1]);motor_violation_offset=residual_offset+count*6
    total=motor_violation_offset+(count*31 if a.restore_feasibility else 0)
    coefficients=[]
    for i,frame in enumerate(range(start,end+1)):
        required=helper.inverse(qs,frame);basis=helper.basis(helper.contacts[i])
        value=nnls(basis[:6]/helper.scales[:,None],required[:6]/helper.scales,maxiter=10000)[0]/helper.weight if basis.shape[1] else np.empty(0)
        coefficients.append(np.clip(value,0,4))
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    sources=[Path(__file__),Path(__file__).with_name('omni_contact_window_model.py'),a.solver_script,a.reference,a.model,a.scene]
    (a.out/'source-sha256.json').write_text(json.dumps({str(f):sha(f) for f in sources},indent=2))
    (a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    trust=1.;initial=helper.measure(qs,coefficients);history=[];started=time.monotonic()
    record=dict(pid=os.getpid(),source_reference_sha256=sha(a.reference),frames50hz=[start,end],time_scale=1.,
        phase='offline_feasibility_restoration' if a.restore_feasibility else 'strict_motor_linearization',
        physical_effort_caps_unchanged=True,motor_violation_variables_are_not_actuator_forces=True,
        approved_for_training=False,actual_controller_execution_verified=False,scope=__doc__,initial=initial)
    def progress(**extra):(a.out/'progress.json').write_text(json.dumps(dict(record,history=history,**extra),indent=2))
    progress(status='running');print(json.dumps(dict(initial=initial)),flush=True)
    termination='iteration_budget'
    try:
        for iteration in range(a.iterations):
            correction=helper.correction(qs);rows=[];cols=[];values=[];lower=[];upper=[];row=0
            def add(indices,data,lo,hi):
                nonlocal row
                indices=np.asarray(indices,dtype=int).ravel();data=np.asarray(data,dtype=float).ravel()
                mask=abs(data)>1e-14;rows.extend([row]*int(mask.sum()));cols.extend(indices[mask].tolist());values.extend(data[mask].tolist())
                lower.append(float(lo));upper.append(float(hi));row+=1
            low=np.full(total,-np.inf);high=np.full(total,np.inf)
            for frame in range(start+1,end):
                i=frame-start-1;bound=np.r_[np.full(3,.005),np.full(34,.02)]*trust
                cumulative=np.r_[np.full(3,.12),np.full(3,.15),helper.domain]
                lo=np.maximum(-bound,-cumulative-correction[frame]);hi=np.minimum(bound,cumulative-correction[frame])
                lo[6:]=np.maximum(lo[6:],helper.centre-helper.radius-qs[frame,7:])
                hi[6:]=np.minimum(hi[6:],helper.centre+helper.radius-qs[frame,7:])
                low[i*37:(i+1)*37]=lo;high[i*37:(i+1)*37]=hi
            low[pose_size:residual_offset]=0.;high[pose_size:residual_offset]=4.
            if a.restore_feasibility:low[motor_violation_offset:]=0.
            if np.any(low>high):raise ValueError('Current candidate left original source/soft-limit domain')
            for i,frame in enumerate(range(start,end+1)):
                required=helper.inverse(qs,frame);basis=helper.basis(helper.contacts[i]);velocity=d.qvel.copy()
                derivatives=[];indices=[];eps=1e-6
                for node in range(max(start+1,frame-1),min(end-1,frame+1)+1):
                    block=np.zeros((37,37));trial=qs.copy()
                    for j in range(37):
                        step=np.zeros(37);step[j]=eps
                        trial[node]=qs[node];mujoco.mj_integratePos(m,trial[node],step,1.)
                        plus=helper.inverse(trial,frame)
                        if node==frame:plus-=helper.basis(helper.contacts[i])@(coefficients[i]*helper.weight)
                        trial[node]=qs[node];mujoco.mj_integratePos(m,trial[node],step,-1.)
                        minus=helper.inverse(trial,frame)
                        if node==frame:minus-=helper.basis(helper.contacts[i])@(coefficients[i]*helper.weight)
                        block[:,j]=(plus-minus)/(2*eps)
                    derivatives.append(block);indices.extend(range((node-start-1)*37,(node-start)*37))
                derivative=np.concatenate(derivatives,axis=1) if derivatives else np.zeros((37,0))
                reaction=-basis*helper.weight;force_indices=np.arange(offsets[i],offsets[i+1])
                # Check the assembled derivative at an independent perturbation
                # before trusting a long sequence of native dynamics QPs.
                if iteration==0 and i==count//2:
                    rng=np.random.default_rng(42);delta=rng.normal(size=len(indices))*1e-7;df=rng.normal(size=len(force_indices))*1e-7
                    trial=qs.copy()
                    for k,node in enumerate(range(max(start+1,frame-1),min(end-1,frame+1)+1)):
                        mujoco.mj_integratePos(m,trial[node],delta[k*37:(k+1)*37],1.)
                    actual=helper.inverse(trial,frame)-helper.basis(helper.contacts[i])@((coefficients[i]+df)*helper.weight)
                    predicted=required-basis@(coefficients[i]*helper.weight)+derivative@delta+reaction@df
                    error=float(np.max(abs(actual-predicted)/np.r_[helper.scales,helper.effort]))
                    (a.out/'jacobian-check.json').write_text(json.dumps(dict(maximum_normalized_error=error,tolerance=1e-5),indent=2))
                    if error>1e-5:raise ValueError('Native dynamics/contact linearization failed independent perturbation test')
                for j in range(37):
                    scale=helper.scales[j] if j<6 else helper.effort[j-6]
                    ii=list(indices)+force_indices.tolist();dd=np.r_[derivative[j],reaction[j]]/scale
                    if j<6:
                        ii.append(residual_offset+i*6+j);dd=np.r_[dd,-1.];add(ii,dd,-required[j]/scale,-required[j]/scale)
                    elif a.restore_feasibility:
                        violation=motor_violation_offset+i*31+j-6
                        # The cap stays .95 of the original motor rating.
                        # Positive violation is an explicit unsatisfied offline
                        # constraint; it is never applied or called feasible.
                        add(ii+[violation],np.r_[dd,-1.],-np.inf,.95-required[j]/scale)
                        add(ii+[violation],np.r_[dd,1.],-.95-required[j]/scale,np.inf)
                    else:add(ii,dd,-.95-required[j]/scale,.95-required[j]/scale)
                # Hinge velocity is linear in neighboring hinge positions.
                for j in range(31):
                    ii=[];dd=[]
                    for node,sign in [(frame-1,-1),(frame+1,1)]:
                        if start<node<end:ii.append((node-start-1)*37+6+j);dd.append(sign*25/helper.speed[j])
                    add(ii,dd,-1.-velocity[6+j]/helper.speed[j],1.-velocity[6+j]/helper.speed[j])
                helper.geometry_state(qs[frame])
                if start<frame<end:
                    block_start=(frame-start-1)*37;pose_indices=np.arange(block_start,block_start+37)
                    for body,target_p,target_r in helper.targets[i]:
                        mujoco.mj_jacBody(m,d,helper.jp,helper.jr,body)
                        error_p=d.xpos[body]-target_p;error_r=(target_r.inv()*Rotation.from_matrix(d.xmat[body].reshape(3,3))).as_rotvec()
                        jac=np.r_[helper.jp,target_r.as_matrix().T@helper.jr];error=np.r_[error_p,error_r]
                        for j in range(6):add(pose_indices,jac[j],-error[j],-error[j])
                    witness=np.zeros(6)
                    for g in helper.robot:
                        for other in helper.terrain:
                            distance=float(mujoco.mj_geomDistance(m,d,g,other,.2,witness))
                            if distance>=.08:continue
                            vector=witness[:3]-witness[3:];length=float(np.linalg.norm(vector))
                            if length<1e-10:raise ValueError('Undefined native collision normal at a touching candidate')
                            normal=np.sign(distance)*vector/length
                            mujoco.mj_jac(m,d,helper.jp,helper.jr,witness[:3],int(m.geom_bodyid[g]))
                            add(pose_indices,normal@helper.jp,-.00085-distance,np.inf)
            # Native row matrices stay sparse; physics and the isolated solver
            # exchange arrays, never imported binary dependencies.
            A=sp.coo_matrix((values,(rows,cols)),shape=(row,total)).tocsc()
            A=sp.vstack([A,sp.eye(total,format='csc')],format='csc');l=np.r_[lower,low];u=np.r_[upper,high]
            weights=np.r_[np.full(3,10.),np.full(3,5.),np.ones(31)]
            diagonal=np.r_[np.tile(weights**2,n),np.full(residual_offset-pose_size,1e-4),np.full(count*6,1e4),
                np.full(count*31 if a.restore_feasibility else 0,1e5)]
            P=sp.diags(diagonal,format='csc');linear=np.zeros(total)
            linear[:pose_size]=(correction[start+1:end]*weights**2).ravel()
            sr=[];sc=[];sv=[];smooth_current=[];smooth_weights=np.r_[np.full(3,500.),np.full(3,50.),np.full(31,5.)]
            for i,frame in enumerate(range(start-1,end+2)):
                smooth_current.extend(correction[frame-1]-2*correction[frame]+correction[frame+1])
                for node,coefficient in [(frame-1,1.),(frame,-2.),(frame+1,1.)]:
                    if start<node<end:
                        sr.extend(range(i*37,(i+1)*37));sc.extend(range((node-start-1)*37,(node-start)*37));sv.extend(coefficient*smooth_weights)
            D=sp.coo_matrix((sv,(sr,sc)),shape=(len(smooth_current),total)).tocsc()
            P=P+D.T@D;linear+=np.asarray(D.T@(np.array(smooth_current)*np.tile(smooth_weights,len(smooth_current)//37))).ravel()
            directory=a.out/f'qp-{iteration:02d}';directory.mkdir();sp.save_npz(directory/'P.npz',sp.triu(P,format='csc'));sp.save_npz(directory/'A.npz',A)
            np.savez_compressed(directory/'vectors.npz',q=linear,lower=l,upper=u)
            solver_env=os.environ.copy();solver_env.pop('PYTHONPATH',None)
            with (directory/'solver.log').open('w') as log:
                completed=subprocess.run([str(a.solver_python),str(a.solver_script),str(directory),'--backend',a.solver_backend],env=solver_env,
                    stdout=log,stderr=subprocess.STDOUT,timeout=90)
            if completed.returncode:raise RuntimeError('Isolated QP solver process failed')
            solver=json.loads((directory/'solver.json').read_text())
            if solver['status_value'] not in [1,2] or solver['maximum_linear_constraint_violation'] is None or solver['maximum_linear_constraint_violation']>1e-5:
                termination='QP not numerically solved';history.append(dict(iteration=iteration,solver=solver,accepted=False));break
            with np.load(directory/'solution.npz') as z:solution=z['x'].copy()
            if solution.shape!=(total,) or not np.isfinite(solution).all():raise ValueError('Malformed QP solution')
            current=helper.measure(qs,coefficients);accepted=False;trials=[]
            for alpha in [1.,.5,.25,.125,.0625]:
                trial=qs.copy()
                for i,frame in enumerate(range(start+1,end)):mujoco.mj_integratePos(m,trial[frame],solution[i*37:(i+1)*37],alpha)
                forces=[old+alpha*(solution[offsets[i]:offsets[i+1]]-old) for i,old in enumerate(coefficients)]
                measured=helper.measure(trial,forces)
                feasible=(measured['environment_penetration_m']<=.001 and measured['self_penetration_m']<=.001
                    and measured['contact_position_error_m']<=.0001 and measured['contact_rotation_error_rad']<=.001
                    and measured['speed_max_ratio']<=1.000001 and measured['soft_joint_excess_rad']<=1e-7
                    and measured['root_translation_domain_max_m']<=.120001 and measured['root_rotation_domain_max_rad']<=.150001
                    and measured['joint_domain_excess_rad']<=1e-6)
                trials.append(dict(alpha=alpha,hard_geometry_contact_domain_passed=feasible,**measured))
                if feasible and measured['merit']<current['merit']*(1.-1e-6):
                    qs,coefficients=trial,forces;accepted=True;break
            item=dict(iteration=iteration,trust=trust,accepted=accepted,trials=trials,solver=solver,elapsed_seconds=time.monotonic()-started)
            if a.restore_feasibility:item['qp_motor_violation_max_ratio']=float(np.max(solution[motor_violation_offset:]))
            history.append(item)
            if not accepted:trust*=.5
            arrays=dict(qpos=qs,qvel=differentiate_qpos(m,qs,50.),fps=np.array([50.]),
                estimated_contact_coefficients=np.concatenate(coefficients),coefficient_offsets=np.r_[0,np.cumsum(sizes)])
            np.savez_compressed(a.out/f'candidate-{iteration:02d}.npz',**arrays)
            np.savez_compressed(a.out/'candidate-latest.npz',**arrays)
            progress(status='running');print(json.dumps(dict(iteration=iteration,accepted=accepted,trust=trust,last=trials[-1])),flush=True)
            if trust<.125:termination='Native hard-geometry line search could not improve dynamics';break
        np.testing.assert_array_equal(qs[:start+1],helper.source[:start+1]);np.testing.assert_array_equal(qs[end:],helper.source[end:])
        final=helper.measure(qs,coefficients);progress(status='completed' if termination=='iteration_budget' else 'stopped',
            termination_reason=termination,final=final,elapsed_seconds=time.monotonic()-started)
    except Exception as error:
        progress(status='failed',reason=str(error));raise


if __name__=='__main__':main()
