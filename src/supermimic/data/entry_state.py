"""Constrained incoming velocities for native-physics experiments."""
import mujoco
import numpy as np


def entry_velocity_state(model,qpos,qvel,target_vx,velocity_limits,*,com_velocity=None,angular_momentum=None):
    """Change incoming translation while preserving H and planted-foot velocity."""
    d=mujoco.MjData(model);d.qpos[:]=qpos;d.qvel[:]=qvel;mujoco.mj_forward(model,d)
    root=int(model.jnt_bodyid[0]);com=np.zeros((3,model.nv));momentum=np.zeros_like(com)
    mujoco.mj_jacSubtreeCom(model,d,com,root);mujoco.mj_angmomMat(model,d,momentum,root)
    target_com=np.array([target_vx,*(com@qvel)[1:]]) if com_velocity is None else np.asarray(com_velocity,dtype=float)
    target_h=momentum@qvel if angular_momentum is None else np.asarray(angular_momentum,dtype=float)
    if target_com.shape!=(3,) or target_h.shape!=(3,) or not np.isfinite(np.r_[target_com,target_h]).all():
        raise ValueError('Entry COM velocity and momentum must be finite three-vectors')
    if not np.isclose(target_com[0],target_vx):raise ValueError('Conflicting forward entry velocity targets')
    matrices=[com,momentum];rhs=[target_com-com@qvel,target_h-momentum@qvel]
    contacts=[]
    for g in range(model.ngeom):
        if 'foot_' not in model.geom(g).name:continue
        k=model.geom_dataid[g];a,n=model.mesh_vertadr[k],model.mesh_vertnum[k]
        world=model.mesh_vert[a:a+n]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
        if world[:,2].min()>.012:continue
        body=int(model.geom_bodyid[g])
        jp,jr=np.zeros_like(com),np.zeros_like(com)
        mujoco.mj_jacBody(model,d,jp,jr,body)
        jac=np.r_[jp,jr];matrices.append(jac);rhs.append(-jac@qvel)
        contacts.append(model.body(body).name)
    if not contacts:raise ValueError('Incoming-speed experiment requires a planted support foot')
    matrix=np.concatenate(matrices);target=np.concatenate(rhs)
    delta=np.linalg.lstsq(matrix,target,rcond=1e-10)[0];velocity=qvel+delta
    error=float(abs(matrix@delta-target).max())
    ratio=float((abs(velocity[6:])/velocity_limits).max())
    if error>1e-6 or ratio>1:raise ValueError(f'Invalid incoming state: constraint error={error}, joint speed ratio={ratio}')
    mass=np.zeros((model.nv,model.nv));mujoco.mj_fullM(model,d,mass)
    return velocity,{'target_com_vx':target_vx,'initial_com_velocity':(com@velocity).tolist(),
        'target_com_velocity':target_com.tolist(),'target_angular_momentum':target_h.tolist(),
        'original_com_velocity':(com@qvel).tolist(),'original_momentum':(momentum@qvel).tolist(),
        'initial_momentum':(momentum@velocity).tolist(),'stationary_support_bodies':contacts,
        'maximum_constraint_error':error,'maximum_joint_speed_ratio':ratio,
        'original_kinetic_energy_j':float(qvel@mass@qvel/2),'initial_kinetic_energy_j':float(velocity@mass@velocity/2),
        'scope':'Initial state only; no forces or state edits after reset'}
