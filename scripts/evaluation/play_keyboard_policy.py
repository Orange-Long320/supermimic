"""Open an accepted five-gait-and-standing policy in a local browser via SSH."""
from pathlib import Path

COMMANDS={'stand':(0.,0.,0.),'forward':(.3,0.,0.),'backward':(-.2,0.,0.),'left':(0.,0.,.4),'right':(0.,0.,-.4),'jog':(.6,0.,0.)}


def start_controls(mailbox, port, viewer_port):
 import json,threading
 from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
 page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>X2 键盘控制</title>
 <style>body{margin:0;font:16px system-ui;background:#14231f;color:#edf4ef}main{max-width:1440px;margin:auto;padding:16px}h1{font-size:22px;margin:0 0 8px}#pad{padding:14px;background:#253a32;border-radius:12px;outline:none;margin-bottom:12px}#pad:focus{box-shadow:inset 0 0 0 2px #aed6bd}button{font:inherit;padding:9px 15px;border:0;border-radius:7px;margin:3px;background:#d5e8db;color:#183024;touch-action:none}iframe{border:0;width:100%;height:76vh;border-radius:12px;background:#fafafa}p{margin:8px 0}small{color:#c5d9ce}</style>
 <main><h1>X2 键盘控制</h1><div id="pad" tabindex="0"><p>点击此处后按住 W 前进、S 后退、A / D 原地转向；Shift + W 小跑。松键或空格站立。</p>
 <button data-mode="forward">W 前进</button><button data-mode="backward">S 后退</button><button data-mode="left">A 左转</button><button data-mode="right">D 右转</button><button data-mode="jog">Shift + W 小跑</button><button data-mode="stand">空格 站立</button>
 <p id="status">站立</p><small>按住按钮也可控制；切换窗口或断开连接后自动停止。</small></div><iframe id="view" title="X2 实时仿真"></iframe></main>
 <script>const keys=new Set(),pad=document.querySelector('#pad'),labels={stand:'站立',forward:'前进',backward:'后退',left:'左转',right:'右转',jog:'小跑'};let pointer=null;
 document.querySelector('#view').src=location.protocol+'//'+location.hostname+':VIEWER_PORT';
 function mode(){if(pointer)return pointer;if(keys.has('Space'))return 'stand';if(keys.has('KeyA')||keys.has('ArrowLeft'))return 'left';if(keys.has('KeyD')||keys.has('ArrowRight'))return 'right';if(keys.has('KeyS')||keys.has('ArrowDown'))return 'backward';if(keys.has('KeyW')||keys.has('ArrowUp'))return keys.has('ShiftLeft')||keys.has('ShiftRight')?'jog':'forward';return 'stand';}
 function send(){let name=mode();document.querySelector('#status').textContent=labels[name];fetch('/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(name)}).catch(()=>{document.querySelector('#status').textContent='连接中断，自动停止';});}
 function stop(){keys.clear();pointer=null;send();}
 const codes=['KeyW','KeyS','KeyA','KeyD','ArrowUp','ArrowDown','ArrowLeft','ArrowRight','Space','ShiftLeft','ShiftRight'];
 pad.addEventListener('keydown',e=>{if(codes.includes(e.code)){e.preventDefault();keys.add(e.code);send();}});pad.addEventListener('keyup',e=>{if(codes.includes(e.code)){e.preventDefault();keys.delete(e.code);send();}});
 document.querySelectorAll('button').forEach(b=>{b.onpointerdown=e=>{b.setPointerCapture(e.pointerId);pointer=b.dataset.mode;send();};b.onpointerup=stop;b.onpointercancel=stop;});
 window.addEventListener('blur',stop);pad.addEventListener('focusout',e=>{if(!pad.contains(e.relatedTarget))stop();});document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});setInterval(send,100);pad.focus();</script></html>'''.replace('VIEWER_PORT',str(viewer_port)).encode()
 class Handler(BaseHTTPRequestHandler):
  def do_GET(self):
   self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(page)));self.end_headers();self.wfile.write(page)
  def do_POST(self):
   try:
    n=int(self.headers.get('Content-Length','0'))
    if self.path!='/command' or not 0<n<=128:raise ValueError('Invalid request')
    name=json.loads(self.rfile.read(n))
    if not isinstance(name,str) or name not in COMMANDS:raise ValueError('Unknown command')
    mailbox.put(COMMANDS[name])
   except (ValueError,TypeError):self.send_error(400);return
   self.send_response(204);self.end_headers()
  def log_message(self,*args):pass
 server=ThreadingHTTPServer(('127.0.0.1',port),Handler);threading.Thread(target=server.serve_forever,daemon=True).start();return server


def main(acceptance:Path,npz_dir:Path,viewer_port:int=18080,keyboard_port:int=18081,device:str='cuda:0'):
 import json,hashlib,torch,viser
 from mjlab.rl import RslRlVecEnvWrapper
 from mjlab.viewer import ViserPlayViewer
 from supermimic.envs import SuperMimicEnvCfgArgs,make_env
 from supermimic.eval.benchmark import load_policy_from_checkpoint
 from supermimic.policy.standing import StandingContext,standing_action
 from supermimic.locomotion.config import VelocityCfg
 from supermimic.locomotion.keyboard import CommandMailbox
 report=json.loads(acceptance.read_text())
 if report['status']!='accepted':raise ValueError('Policy has not passed physical acceptance')
 checkpoint=Path(report['accepted_checkpoint'])
 if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=report['accepted_sha256']:raise ValueError('Accepted checkpoint has changed')
 policy,saved=load_policy_from_checkpoint(checkpoint,device)
 cfg=SuperMimicEnvCfgArgs(npz_files=[str(npz_dir/'walk.npz')],num_envs=2,future_frames=saved['env']['future_frames'],episode_length_s=1e5,action_noise_std=0.,obs_noise=False,reset_noise=False,domain_randomization=False,record_transitions=False,anchor_obs=saved['env']['anchor_obs'],velocity=VelocityCfg(**saved['velocity']))
 env=make_env(cfg,device=device);command=env.command_manager.get_term('motion');command.is_velocity[:]=True;command.force_standing_reset=True;command.set_manual_command(COMMANDS['stand']);mailbox=CommandMailbox();controls=start_controls(mailbox,keyboard_port,viewer_port);viewer_server=viser.ViserServer(host='127.0.0.1',port=viewer_port,label='X2 keyboard policy')
 if report.get('control_matmul_precision')=='ieee':
  torch.backends.cuda.matmul.fp32_precision='ieee';torch.backends.cudnn.fp32_precision='ieee'
 context=StandingContext(command,env)
 class Controller:
  @torch.no_grad()
  def __call__(self,observations):
   command.set_manual_command(mailbox.read(env.step_dt));return standing_action(policy,observations['policy'],command.velocity_command,context)
 print(f'Keyboard control and simulation: http://127.0.0.1:{keyboard_port}',flush=True)
 try:ViserPlayViewer(RslRlVecEnvWrapper(env),Controller(),viser_server=viewer_server).run()
 finally:controls.shutdown();controls.server_close();viewer_server.stop();env.close()
if __name__=='__main__':
 import tyro
 tyro.cli(main)
