"""Rebind checkpoint paths only after exact training-data/scene equivalence.

No robot/model/optimizer/sampler values are changed. This handles provenance-only
reexports without weakening Trainer's full-resume scene binding or claiming a
new motion. All computation runs in the dedicated server environment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint','old-dataset','new-dataset','out']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('New output required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Server CPU-only equivalence check required')
    import numpy as np
    import torch
    torch.set_num_threads(2)
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    old=json.loads((a.old_dataset/'manifest.json').read_text())
    new=json.loads((a.new_dataset/'manifest.json').read_text())
    if [c['name'] for c in old['clips']]!=[c['name'] for c in new['clips']]:raise ValueError('Clip order differs')
    for key in ['frames','distinct_source_families','task_categories']:
        if old[key]!=new[key]:raise ValueError('Task coverage differs: '+key)
    old_scene,new_scene=json.loads(Path(old['terrain']).read_text()),json.loads(Path(new['terrain']).read_text())
    if set(old_scene)!=set(new_scene) or old_scene['boxes']!=new_scene['boxes']:raise ValueError('Physical scene differs')
    ignored={'source','motion_sha256'}
    for x,y in zip(old_scene['clips'],new_scene['clips']):
        if {k:v for k,v in x.items() if k not in ignored}!={k:v for k,v in y.items() if k not in ignored}:
            raise ValueError('Clip geometry/task semantics differ')
    comparisons=[]
    for x,y in zip(old['clips'],new['clips']):
        name=x['name'];left=a.old_dataset/'motions'/(name+'.npz');right=a.new_dataset/'motions'/(name+'.npz')
        if sha(left)!=x['motion_sha256'] or sha(right)!=y['motion_sha256']:raise ValueError('Dataset source hash changed')
        differences=[]
        with np.load(left,allow_pickle=False) as l,np.load(right,allow_pickle=False) as r:
            if set(l.files)!=set(r.files):raise ValueError('Motion schema differs')
            for key in l.files:
                equal=l[key].dtype==r[key].dtype and np.array_equal(l[key],r[key])
                if not equal:
                    if key!='source':raise ValueError(f'Training input differs: {name}/{key}')
                    differences.append(key)
        for key in ['offset','frames','box_indices','source_family','task_type','reference_sha256','task_variant']:
            if x[key]!=y[key]:raise ValueError('Reference/task binding differs: '+key)
        comparisons.append(dict(clip=name,training_arrays_identical=True,metadata_only_differences=differences,
            old_motion_sha256=x['motion_sha256'],new_motion_sha256=y['motion_sha256']))
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if Path(ck['config']['env']['terrain_manifest']).read_bytes()!=Path(old['terrain']).read_bytes():
        raise ValueError('Checkpoint does not bind the verified old scene')
    if Path(ck['config']['data']['npz_dir']).resolve()!=(a.old_dataset/'motions').resolve():
        raise ValueError('Checkpoint does not bind the verified old library')
    if not all(torch.isfinite(t).all() for key in ['wm','policy','critic'] for t in ck[key].values() if torch.is_tensor(t)):
        raise ValueError('Nonfinite weights')
    a.out.mkdir(parents=True);(a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    proof=dict(source_checkpoint=str(a.checkpoint),source_checkpoint_sha256=sha(a.checkpoint),
        old_manifest=str(a.old_dataset/'manifest.json'),new_manifest=str(a.new_dataset/'manifest.json'),
        old_manifest_sha256=sha(a.old_dataset/'manifest.json'),new_manifest_sha256=sha(a.new_dataset/'manifest.json'),
        old_terrain_sha256=sha(old['terrain']),new_terrain_sha256=sha(new['terrain']),
        physical_boxes_identical=True,clip_semantics_identical=True,clips=comparisons,
        changed_config_fields=['data.npz_dir','env.terrain_manifest','world_model.state_spec.terrain_manifest'],
        complete_task_verified=False,scope=__doc__)
    (a.out/'equivalence.json').write_text(json.dumps(proof,indent=2))
    ck['config']['data']['npz_dir']=a.new_dataset/'motions'
    ck['config']['env']['terrain_manifest']=Path(new['terrain'])
    ck['config']['world_model']['state_spec']['terrain_manifest']=Path(new['terrain'])
    ck['dataset_metadata_rebinding']=proof
    output=a.out/'checkpoint.pt';torch.save(ck,output)
    loaded=torch.load(output,map_location='cpu',weights_only=False)
    original=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    def exact(x,y):
        if torch.is_tensor(x):return torch.is_tensor(y) and x.dtype==y.dtype and torch.equal(x,y)
        if isinstance(x,dict):return isinstance(y,dict) and set(x)==set(y) and all(exact(v,y[k]) for k,v in x.items())
        if isinstance(x,(tuple,list)):return type(x)==type(y) and len(x)==len(y) and all(exact(l,r) for l,r in zip(x,y))
        return x==y
    checked=[k for k in original if k!='config']
    if not all(exact(original[k],loaded[k]) for k in checked):raise ValueError('Non-configuration checkpoint values changed')
    proof.update(checkpoint=str(output),checkpoint_sha256=sha(output),all_nonconfig_values_exact=True,
        verified_nonconfig_keys=checked,iteration=loaded['iteration'],updates=loaded['num_updates'])
    (a.out/'equivalence.json').write_text(json.dumps(proof,indent=2))
    print(json.dumps({k:v for k,v in proof.items() if k not in ['clips','scope']}),flush=True)


if __name__=='__main__':main()
