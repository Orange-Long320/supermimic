"""Server CPU comparison of isolated original/repaired terminal targets."""
import argparse
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attribution',type=Path,required=True)
    p.add_argument('--repair',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--baseline-repair',type=Path,help='Compare with a previous candidate instead of source terminal pose')
    p.add_argument('--task',help='Show this task from front and side')
    a=p.parse_args()
    if a.out.exists():p.error('New output required')
    os.environ.setdefault('MUJOCO_GL','osmesa')
    import mujoco
    import numpy as np
    from PIL import Image,ImageDraw
    rows=list(json.loads(a.attribution.read_text()).items())
    if a.task:
        selected=next((x for x in rows if x[0]==a.task),None)
        if selected is None:p.error('Unknown task')
        rows=[selected,selected]
    canvas=Image.new('RGB',(960,320*len(rows)),'white'); draw=ImageDraw.Draw(canvas)
    for i,(name,row) in enumerate(rows):
        m=mujoco.MjModel.from_binary_path(row['model']); d=mujoco.MjData(m)
        m.vis.global_.offwidth=480; m.vis.global_.offheight=300
        z=np.load(a.repair/name/'terminal-candidates.npz')
        baseline=np.load(a.baseline_repair/name/'terminal-candidates.npz')['candidate'] if a.baseline_repair else z['original']
        with mujoco.Renderer(m,width=480,height=300) as renderer:
            for j,label in enumerate(['original','candidate']):
                d.qpos[:]=baseline if label=='original' else z[label]; mujoco.mj_forward(m,d)
                camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
                camera.lookat[:]=d.qpos[:3]; camera.lookat[2]-=.25;camera.distance=2.2
                root=int(m.jnt_bodyid[0]); axis=d.xmat[root].reshape(3,3)[:,0]
                camera.azimuth=float(np.degrees(np.arctan2(axis[1],axis[0])))+180;camera.elevation=-20
                if a.task and i==1:camera.azimuth+=90
                renderer.update_scene(d,camera=camera)
                canvas.paste(Image.fromarray(renderer.render()),(j*480,i*320+20))
                draw.text((j*480+8,i*320+5),f'{name} | {"previous" if label=="original" and a.baseline_repair else label} | terminal target, not policy',fill='black')
    a.out.parent.mkdir(parents=True,exist_ok=True);canvas.save(a.out)
    a.out.with_suffix('.as-run.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':main()
