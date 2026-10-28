"""Nonlinear feasibility restoration, preserving X2 joint bounds and geometry."""
import mujoco
import numpy as np
from scipy.optimize import least_squares


def restore_pose(model,base,flat_targets,xy_targets,lower_act,upper_act,
                 penetration_tolerance=.0008,height_tolerance=.001,xy_tolerance=.001,max_nfev=120,
                 surface_pairs=(),surface_gap_upper=.001):
    base=np.asarray(base,dtype=float).copy();lower_act=np.asarray(lower_act);upper_act=np.asarray(upper_act)
    if base.shape!=(model.nq,) or len(lower_act)!=31 or model.nv not in (37,43):raise ValueError('Expected X2 pose layout')
    if not model.jnt_limited[1:32].all():raise ValueError('Expected explicit mechanical limits')
    if np.any(lower_act<model.jnt_range[1:32,0]-1e-12) or np.any(upper_act>model.jnt_range[1:32,1]+1e-12):raise ValueError('Restoration must not widen physical joint limits')
    lower=np.r_[np.full(3,-.5),np.full(3,-.6),lower_act-base[7:38]]
    upper=np.r_[np.full(3,.5),np.full(3,.6),upper_act-base[7:38]]
    obstacles=[g for g in range(model.ngeom) if model.geom(g).name=='ground' or model.geom(g).name.startswith('multi_boxes_')]
    robot=[g for g in range(model.ngeom) if g not in obstacles and (model.geom_contype[g] or model.geom_conaffinity[g])]
    if not obstacles or not(flat_targets or surface_pairs):raise ValueError('Explicit terrain and source contact task required')
    if any(g not in robot or o not in obstacles for g,o in surface_pairs):raise ValueError('Surface targets must use original robot and terrain geometry')
    data=mujoco.MjData(model);velocity=np.zeros(model.nv);witness=np.zeros(6)
    def evaluate(delta,forward=False):
        q=base.copy();velocity[:]=0.;velocity[:37]=delta;mujoco.mj_integratePos(model,q,velocity,1.)
        data.qpos[:]=q
        if forward:mujoco.mj_forward(model,data)
        else:mujoco.mj_kinematics(model,data)
        distances=np.array([mujoco.mj_geomDistance(model,data,g,o,.02,witness) for g in robot for o in obstacles])
        flat=np.array([data.xpos[body,2]-height for body,height in flat_targets.items()])
        xy=np.concatenate([data.xpos[body,:2]-point for body,point in xy_targets.items()]) if xy_targets else np.empty(0)
        surface=np.array([mujoco.mj_geomDistance(model,data,g,o,10.,witness) for g,o in surface_pairs])
        residual=np.r_[np.maximum(-distances-penetration_tolerance,0.)*1000,
            np.maximum(abs(flat)-height_tolerance,0.)*1000,np.maximum(abs(xy)-xy_tolerance,0.)*1000,
            np.maximum(surface-surface_gap_upper,0.)*1000,delta*.01]
        metrics={'maximum_penetration_m':float(max(0.,-distances.min())),
            'maximum_flat_height_error_m':float(abs(flat).max()) if len(flat) else 0.,
            'maximum_sticking_xy_error_m':float(abs(xy).max()) if len(xy) else 0.,
            'maximum_source_contact_gap_m':float(max(0.,surface.max())) if len(surface) else 0.,
            'joint_limit_violation_rad':float(np.maximum(np.maximum(lower_act-q[7:38],q[7:38]-upper_act),0).max())}
        return residual,metrics,q
    def derivative(x):
        columns=[]
        for j in range(37):
            plus=x.copy();minus=x.copy();plus[j]=min(upper[j],x[j]+1e-6);minus[j]=max(lower[j],x[j]-1e-6)
            columns.append((evaluate(plus)[0]-evaluate(minus)[0])/(plus[j]-minus[j]))
        return np.stack(columns,axis=1)
    fit=least_squares(lambda x:evaluate(x)[0],np.clip(np.zeros(37),lower+1e-10,upper-1e-10),
        jac=derivative,bounds=(lower,upper),max_nfev=max_nfev,ftol=1e-9,xtol=1e-9,gtol=1e-9)
    _,metrics,q=evaluate(fit.x,forward=True)
    feasible=(metrics['maximum_penetration_m']<=penetration_tolerance+5e-5
        and metrics['maximum_flat_height_error_m']<=height_tolerance+5e-5
        and metrics['maximum_sticking_xy_error_m']<=xy_tolerance+5e-6
        and metrics['maximum_source_contact_gap_m']<=surface_gap_upper+5e-5
        and metrics['joint_limit_violation_rad']<=1e-8)
    return q,{'geometry_and_task_feasible':bool(feasible),'solver_success':bool(fit.success),'solver_status':int(fit.status),
        'nfev':int(fit.nfev),'cost':float(fit.cost),'metrics':metrics,
        'maximum_joint_change_rad':float(abs(q[7:38]-base[7:38]).max()),
        'local_root_translation_domain_m':.5,'local_root_rotation_vector_component_domain_rad':.6,
        'scope':'Offline feasible pose initialization only; no dynamics, velocity, continuity or control certificate'}
