"""Prepare an isolated reset-free evaluation of the user-confirmed stop timing."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stop',type=Path,required=True);p.add_argument('--continuation',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--stand-seconds',type=float,default=20.);a=p.parse_args()
    if a.out.exists() or a.stand_seconds<3:p.error('New directory and at least 3 seconds required')
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.data.physics_motion import export_qpos_motion
    m=load_g1_model();meta=json.loads((a.stop/'manifest.json').read_text());release=meta['release_frame'];brake=meta['brake_start_frame'];stand=meta['standing_frame'];fps=50.
    if brake-release!=10 or stand-brake!=30:raise ValueError('Expected confirmed 0.6s stop with 10 committed frames')
    def poses(path):
        with np.load(path,allow_pickle=False) as z:
            if abs(float(z['fps'][0])-fps)>1e-6:raise ValueError('Unexpected reference frequency')
            root=list(z['body_names']).index('pelvis');return np.c_[z['body_pos_w'][:,root],z['body_quat_w'][:,root],z['joint_pos']].astype(float)
    source=a.stop/'natural-stop.npz'
    if sha(source)!=meta['stop_sha256']:raise ValueError('Source reference hash changed')
    q=poses(source);c=poses(a.continuation)
    if len(c)<=brake+10 or np.max(abs(q[:brake+1]-c[:brake+1]))>1e-5:raise ValueError('Continue and stop pose prefixes mismatch')
    c[:brake+1]=q[:brake+1]
    final=q[stand].copy()
    if np.max(abs(q[stand:]-final))>1e-5:raise ValueError('Standing reference not fixed')
    count=int(round(a.stand_seconds*fps));q=np.vstack([q[:stand],np.repeat(final[None,:],count+2,axis=0)])
    a.out.mkdir(parents=True);export_qpos_motion(m,q,a.out/'candidate.npz',fps,source=str(source));export_qpos_motion(m,c,a.out/'continue.npz',fps,source=str(a.continuation))
    phases=['walk']*release+['committed_walk']*(brake-release)+['brake']*(stand-brake)+['stand']*(len(q)-stand)
    result={'source_reference':str(source),'source_sha256':sha(source),'candidate_sha256':sha(a.out/'candidate.npz'),'continue_sha256':sha(a.out/'continue.npz'),'release_frame':release,'brake_start_frame':brake,'standing_frame':stand,'fps':fps,'stand_requested_seconds':a.stand_seconds,'phases':phases,'scope':'Evaluation only; no training approval'}
    (a.out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='phases'},indent=2))
if __name__=='__main__':main()
