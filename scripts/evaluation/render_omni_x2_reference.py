"""Server CPU preview of an X2 Omni kinematic candidate; never policy footage."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--preview-fps',type=float,default=15.,help='Temporal sampling for bounded CPU previews; playback keeps source duration')
    a=p.parse_args()
    if a.out.exists():p.error('Output exists')
    os.environ.setdefault('MUJOCO_GL','osmesa')
    import mujoco
    import numpy as np
    from PIL import Image
    from scipy.spatial.transform import Rotation
    with np.load(a.reference,allow_pickle=False) as z:q=z['qpos'];fps=float(np.asarray(z['fps']).reshape(-1)[0])
    if not np.isfinite(a.preview_fps) or not 0<a.preview_fps<=fps:p.error('Preview rate must be positive and at most the source rate')
    scene=json.loads(a.scene.read_text())
    if a.model.suffix=='.mjb':
        # The native binary already contains the verified fixed scene. Do not
        # add duplicate boxes or manufacture a recorded-execution artifact.
        model=mujoco.MjModel.from_binary_path(str(a.model))
        binding=a.reference.parent/'resampling-geometry.json'
        if not binding.is_file():raise ValueError('Native reference requires its model-binding geometry report')
        proof=json.loads(binding.read_text())
        if proof.get('model_sha256')!=hashlib.sha256(a.model.read_bytes()).hexdigest():
            raise ValueError('Reference/native model hash mismatch')
    else:
        spec=mujoco.MjSpec.from_file(str(a.model))
        spec.worldbody.add_geom(name='preview_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=(0,0,.05),rgba=(.75,.75,.75,1))
        spec.worldbody.add_light(pos=(1,-3,5),dir=(0,0,-1))
        for i,box in enumerate(scene['scene_boxes']):
            quat=Rotation.from_matrix(box['rotation_world']).as_quat()[[3,0,1,2]]
            spec.worldbody.add_geom(name=f'preview_box_{i}',type=mujoco.mjtGeom.mjGEOM_BOX,pos=box['center'],size=box['half_size'],quat=quat,rgba=(.2,.45,.7,1))
        model=spec.compile()
    data=mujoco.MjData(model)
    if q.shape[1]!=model.nq:raise ValueError('Reference/model dimensions differ')
    model.vis.global_.offwidth=640;model.vis.global_.offheight=480
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:2]=(q[:,:2].min(0)+q[:,:2].max(0))/2;camera.lookat[2]=.8
    camera.distance=max(3.,float(np.ptp(q[:,:2],axis=0).max())*1.2+1.5);camera.azimuth=135;camera.elevation=-20
    a.out.parent.mkdir(parents=True,exist_ok=True)
    step=max(1,round(fps/a.preview_fps));label="drawtext=text='X2 KINEMATIC CANDIDATE | not policy or physics execution':x=8:y=8:fontsize=15:fontcolor=white:box=1:boxcolor=black@0.7"
    command=['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','640x480','-r',str(fps/step),'-i','pipe:0','-an','-vf',label,'-c:v','libx264','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart','-n',str(a.out)]
    previews=[];preview_indices=set(np.linspace(0,len(q)-1,4).astype(int).tolist())
    with mujoco.Renderer(model,width=640,height=480) as renderer:
        process=subprocess.Popen(command,stdin=subprocess.PIPE)
        try:
            for frame,pose in enumerate(q):
                if frame%step and frame not in preview_indices:continue
                data.qpos[:]=pose;mujoco.mj_forward(model,data);renderer.update_scene(data,camera=camera);pixels=renderer.render()
                if frame%step==0:process.stdin.write(pixels.tobytes())
                if frame in preview_indices:previews.append(pixels.copy())
        finally:process.stdin.close()
        if process.wait()!=0:raise RuntimeError('FFmpeg failed')
    Image.fromarray(np.concatenate([np.concatenate(previews[:2],axis=1),np.concatenate(previews[2:],axis=1)],axis=0)).save(a.out.with_suffix('.jpg'))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    a.out.with_suffix('.json').write_text(json.dumps({'reference':str(a.reference),'frames':len(q),'source_fps':fps,'render_stride':step,'renderer':'OSMesa CPU','policy_evaluation':False,'physics_execution':False,'reference_sha256':sha(a.reference),'model_sha256':sha(a.model),'scene_sha256':sha(a.scene),'video_sha256':sha(a.out)},indent=2)+'\n')
    a.out.with_suffix('.as-run.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':main()
