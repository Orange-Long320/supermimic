"""Evaluate an immutable WM on previously collected independent physics windows."""
import argparse,hashlib,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','windows','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    import torch
    from supermimic.world_model.world_model import WorldModel,WorldModelCfg
    from supermimic.world_model.state_spec import StateSpecCfg
    from supermimic.locomotion.task import measured_velocity
    torch.set_num_threads(2)
    ck=torch.load(a.checkpoint,map_location='cuda:0',weights_only=False)
    cfg=dict(ck['config']['world_model']);cfg['state_spec']=StateSpecCfg(**cfg['state_spec'])
    wm=WorldModel(WorldModelCfg(**cfg),None,'cuda:0').eval()
    missing,extra=wm.load_state_dict(ck['wm'],strict=False)
    assert not missing and all(k.startswith('heads.') for k in extra),(missing,extra)
    test=torch.load(a.windows,map_location='cuda:0',weights_only=False)
    rows={}
    with torch.no_grad():
        for name,data in test.items():
            pred=data['states'][0];row={}
            for j in range(32):
                pred=wm.step(pred,data['actions'][j],data['xi'])
                if j+1 in (1,8,32):
                    actual=data['states'][j+1]
                    delta=measured_velocity(pred)-measured_velocity(actual)
                    row[str(j+1)]=dict(velocity_rmse=delta.square().mean(0).sqrt().tolist(),velocity_bias=delta.mean(0).tolist(),nse=float(wm.spec.normalized_sq_error(pred,actual).mean()))
            rows[name]=row
    with a.checkpoint.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    result=dict(checkpoint=str(a.checkpoint),checkpoint_sha256=sha,independent_windows=str(a.windows),metrics=rows,scope='Matched held-out states/actions/context collected under fixed earlier policy; no fitting or policy update.')
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
