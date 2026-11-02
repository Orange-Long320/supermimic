"""Compare a known simulator slope geometry with odometry-only scoring."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--slope-deg',type=float,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('output exists')
    r=a.run;x=json.loads((r/'comparison.json').read_text());start=x['common_start_stamp'];end=x['common_end_stamp']
    f=[json.loads(line) for line in (r/'pelvis/packets.jsonl').open()];f=[z for z in f if 'position' in z]
    t=np.array([z['stamp_ns'] for z in f])*1e-9;est=np.array([z['position'] for z in f])
    o=[json.loads(line) for line in (r/'trial-record/odom.jsonl').open()]
    tt=np.array([z['stamp'][0]+z['stamp'][1]*1e-9 for z in o]);pos=np.array([z['position'] for z in o])
    grid=np.arange(start,end,.02);ix=np.clip(np.searchsorted(t,grid,side='right'),1,len(t)-1)
    grid=grid[(t[ix]-t[ix-1])<=.05]
    if len(grid)<2:raise ValueError('insufficient valid points')
    z=np.interp(grid,t,est[:,2]);z-=z[0];truthz=np.interp(grid,tt,pos[:,2]);truthz-=truthz[0]
    error=z-truthz;truthx=np.interp(grid,tt,pos[:,0]);truthx-=truthx[0]
    theory=np.tan(np.deg2rad(a.slope_deg))*truthx
    result={'scope':__doc__,'truth_use':'post-hoc scoring only','slope_deg':a.slope_deg,
            'global_x_displacement_final_m':float(truthx[-1]),'slope_plane_predict_final_m':float(theory[-1]),
            'z_error_final_m':float(error[-1]),'correlation_z_error_and_slope_x':float(np.corrcoef(error,theory)[0,1]),
            'slope_pred_residual_rmse_m':float(np.sqrt(np.mean((error-theory)**2)))}
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
