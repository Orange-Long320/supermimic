"""Compile the project X2 mjlab training scene for CPU diagnostics only.

This preserves the configured geometry and actuators; it does not train or
claim that the model describes a particular physical robot.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('Output exists')
    import mujoco
    from mjlab.scene import Scene,SceneCfg
    from mjlab.terrains import TerrainEntityCfg
    from mjlab.sim import MujocoCfg
    from supermimic.robots.assets import get_robot_cfg
    from supermimic.robots.dimensions import PROFILE
    scene=Scene(SceneCfg(terrain=TerrainEntityCfg(terrain_type='plane'),entities={'robot':get_robot_cfg()}),device='cpu')
    m=scene.compile();cfg=MujocoCfg(timestep=.005,iterations=10,ls_iterations=20)
    m.opt.timestep=cfg.timestep;m.opt.iterations=cfg.iterations;m.opt.ls_iterations=cfg.ls_iterations
    m.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    if m.nu!=31:raise ValueError('Unexpected actuator count')
    joints=[]
    for name in PROFILE['joint_names']:
        jid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'robot/'+name)
        if jid<0:raise ValueError('Missing training joint '+name)
        motors=[i for i in range(m.nu) if m.actuator_trnid[i,0]==jid]
        if len(motors)!=1:raise ValueError('Missing training actuator '+name)
        i=motors[0];joints.append({'name':name,'actuator':int(i),'armature':float(m.dof_armature[m.jnt_dofadr[jid]]),
            'gain':m.actuator_gainprm[i].tolist(),'bias':m.actuator_biasprm[i].tolist(),
            'force_limited':bool(m.actuator_forcelimited[i]),'force_range':m.actuator_forcerange[i].tolist()})
    a.out.mkdir(parents=True);mujoco.mj_saveModel(m,str(a.out/'training-scene.mjb'))
    f=a.out/'training-scene.mjb';data={'scope':__doc__,'mujoco_version':mujoco.__version__,
        'model_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'timestep':m.opt.timestep,
        'nq':m.nq,'nv':m.nv,'nu':m.nu,'geoms':[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) for i in range(m.ngeom)],
        'joints':joints,'hardware_ready':False}
    (a.out/'build.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps({k:data[k] for k in ('model_sha256','timestep','nq','nv','nu')},indent=2))


if __name__=='__main__':main()
