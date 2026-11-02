"""Score all completed offline variants on the same valid timestamps and fixed origin."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('output exists')
    manifest=json.loads((a.results/'manifest.json').read_text());base=Path(manifest['source'])
    with (base/'trial-record/odom.jsonl').open() as f: truth=[json.loads(s) for s in f]
    tt=np.array([r['stamp'][0]+r['stamp'][1]*1e-9 for r in truth]);pos=np.array([r['position'] for r in truth])
    interp=lambda t,v,g:np.column_stack([np.interp(g,t,v[:,i]) for i in range(v.shape[1])])
    runs={};excluded={}
    for name in manifest['variants']:
        summary=json.loads((a.results/(name+'.json')).read_text())
        if summary.get('runtime_failure') or summary.get('failure'):
            excluded[name]={'reason':summary.get('runtime_failure') or summary.get('failure'),'failure_time_s':summary.get('failure_time_s')};continue
        data=np.load(a.results/(name+'.npz'))
        if len(data['t'])<2:excluded[name]={'reason':'insufficient output'};continue
        runs[name]=(data['t'],data['p'],data['v'])
    if not runs:raise ValueError('no valid variants')
    start=max(tt[0]+.05,*(x[0][0] for x in runs.values()));end=min(tt[-1]-.05,*(x[0][-1] for x in runs.values()))
    grid=np.arange(start,end,.02);valid=np.ones(len(grid),bool)
    for t,_,_ in runs.values():
        idx=np.clip(np.searchsorted(t,grid,side='right'),1,len(t)-1)
        valid &= t[idx]-t[idx-1]<=.05
    grid=grid[valid]
    if len(grid)<2:raise ValueError('no common valid timestamps')
    actual=interp(tt,pos,grid)-interp(tt,pos,grid[:1]);truthv=(interp(tt,pos,grid+.05)-interp(tt,pos,grid-.05))/.1
    rmse=lambda x:float(np.sqrt(np.mean(x*x))); report={}
    for name,(t,p,v) in runs.items():
        pe=interp(t,p,grid)-interp(t,p,grid[:1])-actual;ve=interp(t,v,grid)-truthv
        report[name]={'xy_position_rmse_m':rmse(np.linalg.norm(pe[:,:2],axis=1)),'z_position_rmse_m':rmse(pe[:,2]),
            'xy_velocity_rmse_m_s':rmse(np.linalg.norm(ve[:,:2],axis=1)),'z_velocity_rmse_m_s':rmse(ve[:,2]),
            'max_xy_position_error_m':float(np.max(np.linalg.norm(pe[:,:2],axis=1))),
            'max_z_position_error_m':float(np.max(np.abs(pe[:,2]))),
            'max_xy_velocity_error_m_s':float(np.max(np.linalg.norm(ve[:,:2],axis=1))),
            'max_z_velocity_error_m_s':float(np.max(np.abs(ve[:,2]))),
            'final_position_error_m':pe[-1].tolist(),'final_velocity_error_m_s':ve[-1].tolist()}
    result={'scope':__doc__,'start_stamp':float(grid[0]),'end_stamp':float(grid[-1]),'duration_s':float(grid[-1]-grid[0]),
            'shared_coverage_of_source':float(len(grid)*.02/manifest['duration_s']),
            'scored_samples':len(grid),'excluded':excluded,'variants':report}
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
