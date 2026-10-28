"""Render an audited OmniRetarget G1 source clip on the server for visual review."""
import argparse
import ast
import io
import json
import os
from pathlib import Path
import subprocess
import zipfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--motion',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('Output exists')
    os.environ.setdefault('MUJOCO_GL','egl')
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation
    screening=json.loads((a.audit/'screening.json').read_text())
    row=next(x for x in screening['motions'] if x['motion']==a.motion)
    options=[x for x in screening['candidate_windows'] if x['motion']==a.motion]
    chosen=min(options,key=lambda x:x['height_error_m'])
    with zipfile.ZipFile(a.source/'robot-terrain.zip') as archive:
        with np.load(io.BytesIO(archive.read(a.motion)),allow_pickle=False) as z:q=z['qpos'].copy();fps=float(np.asarray(z['fps']).reshape(-1)[0])
    end=min(len(q),chosen['end_frame']+round(fps)+1);q=q[:end]
    spec=mujoco.MjSpec.from_file(str(a.audit/'source-g1.urdf'))
    spec.worldbody.add_geom(name='source_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=(0,0,.1),rgba=(.75,.75,.75,1))
    spec.worldbody.add_light(pos=(1,-3,5),dir=(0,0,-1))
    for i,box in enumerate(row['boxes']):
        quat=Rotation.from_matrix(box['rotation_world']).as_quat()[[3,0,1,2]]
        spec.worldbody.add_geom(name=f'source_box_{i}',type=mujoco.mjtGeom.mjGEOM_BOX,pos=box['center'],size=box['half_size'],quat=quat,rgba=(.2,.45,.7,1))
    model=spec.compile();model.vis.global_.offwidth=800;model.vis.global_.offheight=600;data=mujoco.MjData(model)
    names=None
    config=a.source/'holosoma-sources/src/holosoma_retargeting/holosoma_retargeting/config_types/data_conversion.py'
    for node in ast.parse(config.read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_ROBOT_JOINT_NAMES_DEFAULT' for t in node.targets):names=ast.literal_eval(node.value)['g1']
    if names is None:raise ValueError('Missing verified source joint order')
    columns=[int(model.jnt_qposadr[model.joint(n).id]) for n in names]
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:2]=(q[:,4:6].min(0)+q[:,4:6].max(0))/2;camera.lookat[2]=.8
    camera.distance=max(3.5,float(np.ptp(q[:,4:6],axis=0).max())*1.4+1.5);camera.azimuth=135;camera.elevation=-20
    a.out.parent.mkdir(parents=True,exist_ok=True)
    label="drawtext=text='OmniRetarget G1 SOURCE | kinematic reference | not X2 policy':x=10:y=10:fontsize=17:fontcolor=white:box=1:boxcolor=black@0.7"
    command=['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','800x600','-r',str(fps),'-i','pipe:0','-an','-vf',label,'-c:v','libx264','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart','-n',str(a.out)]
    with mujoco.Renderer(model,width=800,height=600) as renderer:
        process=subprocess.Popen(command,stdin=subprocess.PIPE)
        try:
            for pose in q:
                data.qpos[:3]=pose[4:7];data.qpos[3:7]=pose[:4];data.qpos[columns]=pose[7:]
                mujoco.mj_forward(model,data);renderer.update_scene(data,camera=camera);process.stdin.write(renderer.render().tobytes())
        finally:process.stdin.close()
        if process.wait()!=0:raise RuntimeError('FFmpeg failed')
    a.out.with_suffix('.json').write_text(json.dumps({'source_motion':a.motion,'fps':fps,'frames':len(q),'selected_candidate':chosen,'scope':'Original G1 source trajectory, not X2 retargeting or physics rollout'},indent=2)+'\n')
    a.out.with_suffix('.as-run.py').write_bytes(Path(__file__).read_bytes())
    print('Rendered source frames',len(q),flush=True)


if __name__=='__main__':main()
