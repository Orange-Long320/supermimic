"""Kinematic pose illustration. No dynamics steps or policy-success claim."""
from pathlib import Path
import argparse,json,subprocess

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('checkpoint','reference','dataset','out'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 import torch,numpy as np,mujoco
 from PIL import Image,ImageDraw,ImageFont
 from supermimic.envs import SuperMimicEnvCfgArgs,make_env
 from supermimic.locomotion.config import VelocityCfg
 from supermimic.robots.dimensions import PROFILE
 cfg=torch.load(a.checkpoint,map_location='cpu',weights_only=False)['config']
 env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(a.dataset/'walk.npz')],num_envs=2,seed=0,future_frames=cfg['env']['future_frames'],episode_length_s=10.,action_noise_std=0.,obs_noise=False,reset_noise=False,domain_randomization=False,record_transitions=False,anchor_obs=cfg['env']['anchor_obs'],velocity=VelocityCfg(**cfg['velocity'])),device='cuda:0')
 cmd=env.command_manager.get_term('motion');cmd.is_velocity[:]=True;cmd.force_standing_reset=True;cmd.set_manual_command((0,0,0));env.reset()
 model=env.sim.mj_model;model.vis.global_.offwidth=960;model.vis.global_.offheight=720
 data=mujoco.MjData(model);data.qpos[:]=env.sim.data.qpos[:].cpu().numpy()[0];data.qvel[:]=0
 reference=np.load(a.reference);data.qpos[:3]=reference['body_pos_w'][0,0];data.qpos[3:7]=reference['body_quat_w'][0,0];data.qpos[7:]=reference['joint_pos'][0]
 mujoco.mj_forward(model,data)
 renderer=mujoco.Renderer(model,height=720,width=960);camera=mujoco.MjvCamera();camera.distance=2.9;camera.elevation=-12;camera.lookat[:]=data.qpos[:3]+np.array([0,0,.15])
 a.out.parent.mkdir(parents=True,exist_ok=True)
 writer=subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','960x720','-r','25','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(a.out)],stdin=subprocess.PIPE)
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
 try:
  for frame in range(125):
   camera.azimuth=135+30*np.sin(frame/124*np.pi*2)
   renderer.update_scene(data,camera);im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im);draw.rectangle((10,10,950,104),fill=(18,26,31));draw.text((24,20),'STATIC STANDING TRAINING REFERENCE | bent elbows',font=font,fill='white');draw.text((24,55),'Static checks passed | policy balance training is pending',font=font,fill=(255,205,110));writer.stdin.write(im.tobytes())
 finally:writer.stdin.close();code=writer.wait();renderer.close();env.close()
 if code:raise RuntimeError(code)
 a.out.with_suffix('.json').write_text(json.dumps(dict(kind='kinematic_pose_reference',physics_steps=0,reference=str(a.reference),checkpoint_assets=str(a.checkpoint),seconds=5),indent=2))
if __name__=='__main__':main()
