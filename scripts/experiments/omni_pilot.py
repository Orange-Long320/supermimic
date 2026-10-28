"""User-authorized exploratory Omni multi-motion training, bounded to one hour.

This entry does not assert that inverse-force or complete-climb audits pass.
It retains original simulation limits and labels all resulting policies pending
full ascent, terminal standing and independent physical validation.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def warmstart_terrain_module(module,state,first,base_feature_dim):
    """Insert zero terrain columns without moving action/goal/xi weights."""
    state=dict(state)
    old=state[first];new=module.state_dict()[first].clone().zero_()
    if new.shape[1]!=old.shape[1]+18:raise ValueError('Unexpected terrain input expansion')
    new[:,:base_feature_dim]=old[:,:base_feature_dim]
    new[:,base_feature_dim+18:]=old[:,base_feature_dim:]
    state[first]=new;module.load_state_dict(state)


def prepare(inputs,out,include_task_variants=False):
    import mujoco
    import numpy as np
    from mjlab.scene import Scene
    from supermimic.envs.tracking_env_cfg import SuperMimicEnvCfgArgs,make_env_cfg
    from supermimic.robots.dimensions import PROFILE
    if len(inputs)<2:raise ValueError('Distinct motion families required')
    scenes=[json.loads((source/'scene.json').read_text()) for source in inputs]
    if len({s['source_motion'].split('_z_scale_')[0] for s in scenes})<2:
        raise ValueError('Task variants do not increase the distinct source family count')
    out.mkdir(parents=True,exist_ok=False);(out/'motions').mkdir()
    boxes=[];clips=[];families=set()
    for i,(source,scene) in enumerate(zip(inputs,scenes)):
        variant=scene.get('task_variant')
        task='source_full_motion'
        if not scene.get('complete_source_clip'):
            if not (include_task_variants and scene.get('complete_source_prefix') and scene.get('source_start_frame')==0
                and scene.get('dense_window_start_frame')==0 and isinstance(variant,dict)
                and variant.get('kind') in ['ascent_then_hold','traverse_then_hold','return_then_hold'] and variant.get('hold_seconds',0)>=2-1e-9):
                raise ValueError('A diagnostic window is not a complete source or verified task prefix')
            parent=Path(scene['source_reference_path'])
            parent_meta=json.loads((parent.parent/'result.json').read_text())
            with np.load(parent,allow_pickle=False) as z:parent_count=len(z['qpos']);parent_fps=float(z['fps'].reshape(-1)[0])
            if (sha(parent)!=scene['source_reference_sha256'] or not parent_meta.get('complete_source_clip')
                or parent_meta.get('source_start_frame')!=0 or parent_meta['source_motion']!=scene['source_motion']
                or parent_meta['model_sha256']!=scene['model_sha256'] or parent_fps!=200.
                or not 0<=variant['brake_frame']<variant['stop_frame']<parent_count
                or any(parent_meta[k]!=scene[k] for k in ['joint_limit_factor','wrist_reference_limit_deg','scene_boxes'])):
                raise ValueError('Task prefix parent/source/model binding failed')
            if variant['kind'] in ['traverse_then_hold','return_then_hold'] and variant['stop_frame']!=parent_count-1:
                raise ValueError('Landing task must cover the complete original source interval')
            task={'ascent_then_hold':'box_stand','traverse_then_hold':'vault_land','return_then_hold':'step_up_down_land'}[variant['kind']]
        family=scene['source_motion'].split('_z_scale_')[0]
        if family in families and not include_task_variants:raise ValueError('Scaled copies do not count as different motions')
        families.add(family);offset=np.array([0.,(i-(len(inputs)-1)/2)*6.,0.])
        with np.load(source/'motion-fk.npz',allow_pickle=False) as z:motion={k:z[k].copy() for k in z.files}
        if include_task_variants:
            exported=json.loads((source/'manifest.json').read_text())
            reference_path=Path(str(np.asarray(motion['source']).reshape(-1)[0]))
            if (sha(reference_path)!=exported['source_reference_sha256']
                or not exported.get('resampling_velocity',{}).get('passed')
                or exported['resampling_velocity']['maximum_dense_joint_velocity_ratio']>1.
                or exported['resampling_geometry']['model_sha256']!=scene['model_sha256']
                or not exported['resampling_geometry']['passed']):
                raise ValueError('Task dataset requires bound dense geometry and velocity export evidence')
        with np.load(source/'reference-qpos.npz',allow_pickle=False) as z:q=z['qpos'].copy();v=z['qvel'].copy()
        if float(motion['fps'][0])!=50. or q.shape!=(len(motion['joint_pos']),38):raise ValueError('Expected matching 50 Hz X2 reference')
        if list(motion['joint_names'])!=PROFILE['joint_names']:raise ValueError('Unexpected robot order')
        motion['body_pos_w']+=offset.astype(motion['body_pos_w'].dtype);q[:,:3]+=offset
        name=f'{i:02d}_{Path(scene["source_motion"]).stem}'+('' if task=='source_full_motion' else '_'+task)
        np.savez_compressed(out/'motions'/f'{name}.npz',**motion)
        np.savez_compressed(out/f'{name}-qpos.npz',qpos=q,qvel=v,fps=np.array([50.]))
        first=len(boxes)
        for box in scene['scene_boxes']:
            boxes.append({**box,'center':(np.array(box['center'])+offset).tolist()})
        clips.append({'name':name,'source':str(source),'source_motion':scene['source_motion'],'offset':offset.tolist(),
            'frames':len(q),'motion_sha256':sha(out/'motions'/f'{name}.npz'),'reference_sha256':sha(source/'reference-qpos.npz'),
            'box_indices':list(range(first,len(boxes))),'source_family':family,'task_type':task,
            'complete_source_clip':bool(scene.get('complete_source_clip')),'complete_source_prefix':bool(scene.get('complete_source_prefix')),
            'task_variant':variant})
    manifest=out/'terrain.json';manifest.write_text(json.dumps({'boxes':boxes,'clips':clips},indent=2))
    cfg=make_env_cfg(SuperMimicEnvCfgArgs(npz_files=[],terrain_manifest=manifest))
    m=Scene(cfg.scene,device='cpu').compile();cfg.sim.mujoco.apply(m);d=mujoco.MjData(m)
    if (m.nq,m.nv,m.nu)!=(38,37,31):raise ValueError('Unexpected training robot')
    effort=np.array([PROFILE['effort_limit'][n] for n in PROFILE['joint_names']])
    np.testing.assert_allclose(m.actuator_forcerange,np.c_[-effort,effort],atol=1e-9)
    root=int(m.jnt_bodyid[0]);robot=[g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]]==root and (m.geom_contype[g] or m.geom_conaffinity[g])]
    terrain=[g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]]!=root and (m.geom_contype[g] or m.geom_conaffinity[g])]
    witness=np.zeros(6);speed=np.array([PROFILE['velocity_limit'][n] for n in PROFILE['joint_names']])
    for clip in clips:
        with np.load(out/f'{clip["name"]}-qpos.npz') as z:q=z['qpos'];v=z['qvel']
        if not np.isfinite(q).all() or not np.isfinite(v).all():raise ValueError('Nonfinite training reference')
        hard=float(np.maximum(np.maximum(m.jnt_range[1:,0]-q[:,7:],q[:,7:]-m.jnt_range[1:,1]),0).max())
        peak=0.
        for pose in q:
            d.qpos[:]=pose;mujoco.mj_forward(m,d)
            peak=max(peak,-min(float(mujoco.mj_geomDistance(m,d,g,o,.02,witness)) for g in robot for o in terrain))
        clip.update(maximum_penetration_m=peak,hard_joint_violation_rad=hard,joint_velocity_ratio=float((abs(v[:,6:])/speed).max()))
        if peak>.001 or hard>1e-8 or clip['joint_velocity_ratio']>1.:raise ValueError(f'Gross input/scene check failed: {clip}')
        if clip['task_type']!='source_full_motion':
            if len(q)<101 or abs(v[-100:]).max()>1e-6 or abs(q[-100:]-q[-1]).max()>1e-6:
                raise ValueError('Task reference must retain two stationary terminal seconds')
            d.qpos[:]=q[-1];mujoco.mj_forward(m,d)
            root_up=float(d.xmat[root].reshape(3,3)[2,2])
            target=max(clip['box_indices'],key=lambda i:boxes[i]['center'][2]+boxes[i]['half_size'][2])
            top=boxes[target]['center'][2]+boxes[target]['half_size'][2]
            supported=[]
            for side in ['left','right']:
                body=m.body(f'robot/{side}_ankle_roll_link').id;points=[]
                for g in robot:
                    if m.geom_bodyid[g]!=body:continue
                    if m.geom_type[g]!=int(mujoco.mjtGeom.mjGEOM_MESH):raise ValueError('Expected original full foot meshes')
                    mesh=int(m.geom_dataid[g]);first,count=m.mesh_vertadr[mesh],m.mesh_vertnum[mesh]
                    points.extend(m.mesh_vert[first:first+count]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])
                if not points:raise ValueError('Missing physical foot')
                points=np.array(points);surfaces=set()
                if -.001<=points[:,2].min()<=.002:surfaces.add(-1)
                for i in clip['box_indices']:
                    box=boxes[i];local=(points-box['center'])@np.array(box['rotation_world']);half=np.array(box['half_size'])
                    if (abs(local[:,:2])<=half[:2]+1e-6).all() and -.001<=local[:,2].min()-half[2]<=.002:surfaces.add(i)
                supported.append(surfaces)
            shared=supported[0]&supported[1]
            if clip['task_type']=='box_stand':valid=target in shared
            else:valid=any(i==-1 or boxes[i]['center'][2]+boxes[i]['half_size'][2]<top-.05 for i in shared)
            if not valid or root_up<.9:raise ValueError('Terminal task/reference surface or upright geometry failed')
            clip['terminal_reference_kinematics']={'passed':True,'root_up_z':root_up,'shared_support_surfaces':sorted(shared),
                'scope':'Stationary full-foot reference geometry only, not actual contact force or policy success'}
    record={'clips':clips,'terrain':str(manifest.resolve()),'terrain_sha256':sha(manifest),'frames':sum(c['frames'] for c in clips),
        'distinct_source_families':len(families),'task_categories':sorted({c['task_type'] for c in clips}),
        'exploratory_training_authorized':True,'authorization':'User: if no particularly outrageous issues, train one hour then continue optimizing',
        'force_diagnostics_passed':False,'complete_climb_verified':False,'final_validation_approved':False,
        'scope':'Two or more motion-conditioned scenes spatially separated in each independent simulator world; no generic terrain perception claim'}
    (out/'manifest.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))


def train(a):
    pids=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if pids:raise RuntimeError(f'GPU {a.gpu} occupied: {pids}')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    from supermimic.config import training_config
    from supermimic.training.loop import Trainer
    torch.set_num_threads(2)
    manifest=json.loads((a.dataset/'manifest.json').read_text())
    if not manifest['exploratory_training_authorized'] or sha(manifest['terrain'])!=manifest['terrain_sha256']:raise ValueError('Dataset manifest changed')
    for clip in manifest['clips']:
        if sha(a.dataset/'motions'/f'{clip["name"]}.npz')!=clip['motion_sha256']:raise ValueError('Motion changed')
    cfg=training_config('smoke');cfg.robot='x2_ultra';cfg.run_name=a.out.name;cfg.log_root=a.out.parent
    cfg.data.npz_dir=a.dataset/'motions';cfg.env.terrain_manifest=Path(manifest['terrain'])
    cfg.env.num_envs=512;cfg.env.replay_capacity=256;cfg.env.anchor_obs=True;cfg.env.sampling_mode='uniform'
    cfg.env.domain_randomization=cfg.env.obs_noise=cfg.env.reset_noise=False;cfg.env.action_noise_std=0
    cfg.env.episode_length_s=16.;cfg.wm_trainer.batch_size=cfg.policy_trainer.batch_size=128
    cfg.policy_trainer.obs_noise=False;cfg.schedule.lr_final_ratio=1.
    cfg.schedule.total_iterations=100 if a.phase=='smoke' else 100_000_000
    cfg.schedule.max_hours=1/12 if a.phase=='smoke' else 1.
    cfg.schedule.checkpoint_interval=100 if a.phase=='smoke' else 2000
    cfg.schedule.log_interval=20 if a.phase=='smoke' else 100
    if a.phase=='train':
        proof=json.loads(a.capacity.read_text())
        if proof['status']!='completed' or proof['updates']<1 or proof['terrain_sha256']!=manifest['terrain_sha256'] or proof['checkpoint_sha256']!=sha(a.checkpoint):raise ValueError('Capacity smoke does not match')
        cfg.resume=a.checkpoint
    a.out.mkdir(parents=True,exist_ok=False)
    record={'phase':a.phase,'status':'initializing','pid':os.getpid(),'physical_gpu':a.gpu,'source_commit':a.source_commit,
        'dataset':str(a.dataset),'terrain_sha256':manifest['terrain_sha256'],'initial_checkpoint':str(a.checkpoint),
        'initial_checkpoint_sha256':sha(a.checkpoint),'configuration':asdict(cfg),'final_validation_approved':False,
        'authorization':'Explicit user request for one-hour exploratory training before continued optimization'}
    (a.out/'launch.json').write_text(json.dumps(record,indent=2,default=str));torch.cuda.reset_peak_memory_stats()
    trainer=None;start=time.monotonic()
    try:
        trainer=Trainer(cfg)
        if a.phase=='smoke':
            ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
            if ck['config'].get('robot')!='x2_ultra' or ck['config']['env'].get('box'):raise ValueError('Wrong warmstart robot/observation model')
            from supermimic.robots.dimensions import FEAT_DIM
            for key,module in [('wm',trainer.wm),('policy',trainer.policy),('critic',trainer.critic)]:
                state=ck[key]
                first={'wm':'dynamics.mlp.net.0.weight','critic':'mlp.net.0.weight'}.get(key)
                if first:
                    warmstart_terrain_module(module,state,first,FEAT_DIM)
                else:module.load_state_dict(state)
            del ck
            print('Warmstart weights only; fresh optimizer, replay and sampler for new scenes',flush=True)
        record['status']='running';(a.out/'launch.json').write_text(json.dumps(record,indent=2,default=str))
        trainer.run();checkpoint=a.out/'ckpt_latest.pt'
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if not all(torch.isfinite(t).all() for k in ['wm','policy','critic'] for t in ck[k].values() if torch.is_tensor(t)):raise RuntimeError('Nonfinite checkpoint')
        record.update(status='completed',updates=ck['num_updates'],iteration=ck['iteration'],checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint))
    finally:
        if trainer is not None:
            trainer.env.close()
            if trainer.writer:trainer.writer.close()
        if record['status']!='completed':record['status']='failed'
        record.update(elapsed_seconds=time.monotonic()-start,torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            torch_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        (a.out/'capacity.json').write_text(json.dumps(record,indent=2,default=str))


def main():
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['prepare','smoke','train'],required=True)
    p.add_argument('--inputs',nargs='+',type=Path)
    p.add_argument('--dataset',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--capacity',type=Path)
    p.add_argument('--gpu',type=int,default=3)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--include-task-variants',action='store_true',help='Prepare only: accept source-bound complete ascent/landing goals without counting variants as new families')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra',check_assets=True)
    if a.phase=='prepare':prepare(a.inputs,a.out,a.include_task_variants)
    else:train(a)


if __name__=='__main__':main()
