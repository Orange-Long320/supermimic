"""Constrained incoming velocities for native-physics experiments."""
import mujoco
import numpy as np


def entry_velocity_state(model,qpos,qvel,target_vx,velocity_limits):
    """Change incoming translation while preserving H and planted-foot velocity."""
    d=mujoco.MjData(model);d.qpos[:]=qpos;d.qvel[:]=qvel;mujoco.mj_forward(model,d)
    root=int(model.jnt_bodyid[0]);com=np.zeros((3,model.nv));momentum=np.zeros_like(com)
    mujoco.mj_jacSubtreeCom(model,d,com,root);mujoco.mj_angmomMat(model,d,momentum,root)
    matrices=[com,momentum];rhs=[np.array([target_vx,*(com@qvel)[1:]])-com@qvel,np.zeros(3)]
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
        'original_com_velocity':(com@qvel).tolist(),'original_momentum':(momentum@qvel).tolist(),
        'initial_momentum':(momentum@velocity).tolist(),'stationary_support_bodies':contacts,
        'maximum_constraint_error':error,'maximum_joint_speed_ratio':ratio,
        'original_kinetic_energy_j':float(qvel@mass@qvel/2),'initial_kinetic_energy_j':float(velocity@mass@velocity/2),
        'scope':'Initial state only; no forces or state edits after reset'}
