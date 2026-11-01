"""Render saved post-step X2 simulation poses without re-simulating dynamics."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('scene','candidate','run','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--label',required=True);p.add_argument('--stride',type=int,default=4);a=p.parse_args()
    if a.out.exists() or any(c in a.label for c in "':\\"):raise ValueError('Invalid new output or label')
    if a.stride not in (1,2,4):raise ValueError('Unsupported frame stride')
    report=json.loads((a.run/'result.json').read_text())
    if digest(a.scene)!=report['sha256']['scene'] or digest(a.candidate)!=report['sha256']['candidate']:
        raise ValueError('Replay model or candidate hash mismatch')
    names=json.loads(a.candidate.read_text())['joint_names'];m=mujoco.MjModel.from_xml_path(str(a.scene));d=mujoco.MjData(m)
    ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]
    if len(ids)!=31 or min(ids)<0:raise ValueError('Invalid joint order')
    qa=np.asarray([m.jnt_qposadr[j] for j in ids]);rows=[json.loads(s) for s in (a.run/'frames.jsonl').open()]
    if len(rows)<2 or any('root_quat_after_wxyz' not in r or 'joint_position_after' not in r for r in rows):
        raise ValueError('Recorded post-step poses missing')
    times=np.asarray([r['time_s'] for r in rows]);
    if not np.allclose(np.diff(times),.02,atol=1e-7):raise ValueError('Nonuniform recording')
    width,height=640,480;fps=50./a.stride;m.vis.global_.offwidth=max(width,m.vis.global_.offwidth);m.vis.global_.offheight=max(height,m.vis.global_.offheight)
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE;camera.distance=2.6;camera.azimuth=120;camera.elevation=-12
    camera.lookat[:]=[*np.mean([r['root'][:2] for r in rows],axis=0),.65]
    label=f"drawtext=text='{a.label} | SIMULATION':x=14:y=14:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.7"
    a.out.parent.mkdir(parents=True,exist_ok=True)
    proc=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
                           '-r',str(fps),'-i','pipe:0','-an','-vf',label,'-c:v','libx264','-pix_fmt','yuv420p',
                           '-crf','20','-threads','2','-movflags','+faststart','-n',str(a.out)],stdin=subprocess.PIPE)
    renderer=mujoco.Renderer(m,width=width,height=height)
    try:
        for row in rows[::a.stride]:
            d.qpos[:7]=[*row['root'],*row['root_quat_after_wxyz']];d.qpos[qa]=row['joint_position_after']
            mujoco.mj_forward(m,d);renderer.update_scene(d,camera=camera);proc.stdin.write(renderer.render().tobytes())
    finally:renderer.close();proc.stdin.close()
    if proc.wait()!=0:raise RuntimeError('FFmpeg failed')
    metadata={'scope':__doc__,'scene_sha256':digest(a.scene),'frames_sha256':digest(a.run/'frames.jsonl'),
              'candidate_sha256':digest(a.candidate),'video_sha256':digest(a.out),'frames':len(rows[::a.stride]),
              'fps':fps,'real_time':True,'label':a.label,'physics_rerun_during_render':False}
    a.out.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
