"""Summarize continuous keyboard evaluations without dropping failed trials."""
import argparse,json,math
from pathlib import Path


def summarize(path):
    import numpy as np
    d=json.loads(path.read_text())
    result=dict(file=str(path),checkpoint=d['checkpoint'],seed=d['seed'],
        survived=d['survived_without_reset'],num_envs=d['num_envs'],first_failure_s=d['first_failure_s'],phases=[])
    passed=d['survived_without_reset']==d['num_envs'] and not any(d['resets'])
    for i,(start,target) in enumerate(d['schedule']):
        if start>=d['duration_s']:break
        end=min(d['duration_s'],d['schedule'][i+1][0] if i+1<len(d['schedule']) else d['duration_s'])
        rows=[r for r in d['samples'] if start+1<r['time_s']<=end]
        if not rows:continue
        v=np.asarray([r['velocity'] for r in rows])
        per_env=v.mean(0);mean=per_env.mean(0);target=np.asarray(target)
        # Engineering diagnostic tolerances, not a user-specified benchmark.
        zero=not target.any()
        error=abs(mean-target)
        if zero:
            planar=float(np.linalg.norm(v[:,:,:2],axis=-1).mean())
            ok=planar<=.06 and abs(mean[2])<=.08
        else:
            planar=None
            ok=error[0]<=.05 and error[1]<=.05 and error[2]<=.08
        row=dict(start_s=start,end_s=end,target=target.tolist(),mean_velocity=mean.tolist(),
            per_env_mean_velocity=per_env.tolist(),mean_stop_planar_speed=planar,within_diagnostic_tolerance=bool(ok))
        result['phases'].append(row);passed=passed and ok
    if 'stop_arm_metrics' in d:
        arm_max=max((max(v['joint_velocity_rms_rad_s']) for v in d['stop_arm_metrics'].values()),default=math.inf)
        result['max_stop_arm_rms_rad_s']=arm_max;passed=passed and arm_max<=.12
    result['passed_diagnostic_tolerances']=bool(passed)
    result['scope']='Deterministic simulation only; no filtering failed environments; mean command response is distinct from instantaneous RMSE.'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('files',nargs='+',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    rows=[summarize(f) for f in a.files]
    a.out.write_text(json.dumps(rows,indent=2)+'\n')
    for row in rows:
        print(row['file'],row['survived'],'/',row['num_envs'],'passed',row['passed_diagnostic_tolerances'])
        for r in row['phases']:print(r['start_s'],r['target'],[round(v,4) for v in r['mean_velocity']],r['within_diagnostic_tolerance'])

if __name__=='__main__':main()
