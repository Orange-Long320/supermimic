"""Build an exploratory walk/loop/stop trajectory; no claim of contact validity."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation, Slerp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--loop-start', type=int, default=70)
    p.add_argument('--loop-end', type=int, default=123)
    p.add_argument('--join-frames', type=int, default=12)
    p.add_argument('--loops', type=int, default=3)
    p.add_argument('--stop-frames', type=int, default=75)
    p.add_argument('--hold-frames', type=int, default=250)
    p.add_argument('--release-lead', type=int, default=0, help='Commit continuation frames after release before braking')
    a = p.parse_args()
    if a.out.exists() or not (0 < a.loop_start < a.loop_end and a.join_frames >= 3 and a.loops >= 1 and a.stop_frames >= 4 and a.hold_frames >= 1 and 0 <= a.release_lead <= a.join_frames):
        p.error('Invalid bounds or output already exists')
    from supermimic.data.motion_convert import load_g1_model, convert_csv_to_npz
    model = load_g1_model()
    with np.load(a.source, allow_pickle=False) as d:
        root = list(d['body_names']).index(model.body(int(model.jnt_bodyid[0])).name)
        fps = float(d['fps'][0])
        original = np.c_[d['body_pos_w'][:, root], d['body_quat_w'][:, root][:, [1,2,3,0]], d['joint_pos']].astype(float)
    if a.loop_end + 1 >= len(original) or abs(fps-50)>1e-6:
        raise ValueError('Expected 50 Hz reference and an interior loop end')
    # Position derivatives use neighboring original samples. Rotation uses slerp;
    # resulting reconstructed velocities are regenerated from the assembled poses.
    def bridge(start, end, v0, v1, n):
        t=np.arange(1,n+1)/n
        spl=CubicHermiteSpline([0,n/fps],np.stack([start,end]),np.stack([v0,v1]),axis=0)
        x=spl(t*n/fps)
        x[:,3:7]=Slerp([0,1],Rotation.from_quat(np.stack([start[3:7],end[3:7]])))(t).as_quat()
        return x
    velocity=lambda i:(original[i+1]-original[i-1])*fps/2
    seq=[x.copy() for x in original[:a.loop_end+1]]
    phases=['start' if i<a.loop_start else 'loop' for i in range(len(seq))]
    for _ in range(a.loops):
        prev=seq[-1].copy(); dest=original[a.loop_start].copy()
        # Continue displacement using average source root speed over the bridge.
        dest[:2]=prev[:2]+(velocity(a.loop_end)[:2]+velocity(a.loop_start)[:2])*.5*a.join_frames/fps
        offset=dest[:2]-original[a.loop_start,:2]
        for row in bridge(prev,dest,velocity(a.loop_end),velocity(a.loop_start),a.join_frames):seq.append(row);phases.append('loop_join')
        for row in original[a.loop_start+1:a.loop_end+1]:
            x=row.copy();x[:2]+=offset;seq.append(x);phases.append('loop')
    release=len(seq)-1
    loop_tail=[]
    prev=seq[-1].copy();dest=original[a.loop_start].copy();dest[:2]=prev[:2]+(velocity(a.loop_end)[:2]+velocity(a.loop_start)[:2])*.5*a.join_frames/fps
    offset=dest[:2]-original[a.loop_start,:2]
    loop_tail.extend(bridge(prev,dest,velocity(a.loop_end),velocity(a.loop_start),a.join_frames))
    for row in original[a.loop_start+1:a.loop_end+1]:
        x=row.copy();x[:2]+=offset;loop_tail.append(x)
    for row in loop_tail[:a.release_lead]:
        seq.append(row.copy());phases.append('committed_walk')
    start=seq[-1].copy();end=original[10].copy()
    start_v=(seq[-1]-seq[-2])*fps if a.release_lead else velocity(a.loop_end)
    end[:2]=start[:2]+start_v[:2]*a.stop_frames/fps*.5
    # Bound joint tangents to a monotone cubic, avoiding Hermite limit overshoot.
    delta=end[7:]-start[7:]; duration=a.stop_frames/fps
    start_v[7:]=np.where(start_v[7:]*delta>0,np.sign(delta)*np.minimum(abs(start_v[7:]),3*abs(delta)/duration),0.)
    for i,row in enumerate(bridge(start,end,start_v,np.zeros_like(end),a.stop_frames)):
        seq.append(row);phases.append('brake' if i<a.stop_frames//2 else 'settle')
    for _ in range(a.hold_frames):seq.append(end.copy());phases.append('stand')
    a.out.mkdir(parents=True)
    for name,rows in [('candidate',seq),('continue',seq[:release+1]+loop_tail)]:
        csv=a.out/(name+'.csv');np.savetxt(csv,np.asarray(rows),delimiter=',',fmt='%.10f')
        convert_csv_to_npz(csv,a.out/(name+'.npz'),fps,fps,model)
    report=dict(source=str(a.source),source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
                parameters=vars(a)|{'source':str(a.source),'out':str(a.out)},release_frame=release,
                fps=fps,frames=len(seq),phases=phases,scope='Synthetic interpolation; contact and dynamic feasibility unverified',
                candidate_sha256=hashlib.sha256((a.out/'candidate.npz').read_bytes()).hexdigest())
    (a.out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='phases'},indent=2))

if __name__=='__main__':main()
