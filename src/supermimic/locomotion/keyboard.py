"""Browser keyboard input with an expiring lease and bounded acceleration.

The HTTP worker only writes CPU values. The simulation thread owns all CUDA state.
"""

import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class CommandMailbox:
    def __init__(self, timeout=0.5, acceleration=(1.0, 1.0, 1.5)):
        self.timeout = timeout
        self.acceleration = acceleration
        self.target = (0., 0., 0.)
        self.current = (0., 0., 0.)
        self.updated = float("-inf")
        self.lock = threading.Lock()

    def put(self, values, now=None):
        values = tuple(float(x) for x in values)
        if len(values) != 3 or not all(math.isfinite(x) for x in values):
            raise ValueError("Expected three finite command values")
        with self.lock:
            self.target = values
            self.updated = time.monotonic() if now is None else now

    def read(self, dt, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            target = self.target if now - self.updated <= self.timeout else (0., 0., 0.)
        self.current = tuple(current + max(-rate * dt, min(rate * dt, goal - current)) for current, goal, rate in zip(self.current, target, self.acceleration))
        return self.current


def start_keyboard_server(mailbox, port, forward_speed, turn_speed):
    page = """<!doctype html><html lang="zh"><meta charset="utf-8">
<title>X2 行走控制</title><style>
body{font:20px system-ui;background:#16231e;color:#eef4ef;max-width:660px;margin:70px auto;padding:24px}
button{font:inherit;background:#bdd6c4;color:#15251b;border:0;border-radius:10px;padding:15px 24px;margin:8px;touch-action:none}
#pad{padding:24px;border:2px solid #739b81;border-radius:18px;outline:none}small{color:#bed0c3}
</style><h1>X2 行走控制</h1><p>按住 W 前进，A / D 左右转向；松键或空格停止。</p>
<div id="pad" tabindex="0"><p>点击这里启用键盘，再按住方向键。</p>
<button data-key="a">A 左转</button><button data-key="w">W 前进</button><button data-key="d">D 右转</button>
<p id="status">已停止</p></div><p><small>切换窗口或连接中断后会自动归零。仿真画面在查看器窗口。</small></p>
<script>const keys=new Set(),pad=document.querySelector('#pad');
function send(){const cmd=[keys.has('w')?FORWARD:0,0,((keys.has('a')?1:0)-(keys.has('d')?1:0))*TURN];
document.querySelector('#status').textContent=cmd.some(v=>v!==0)?'正在发送行走指令':'已停止';
fetch('/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(cmd)}).catch(()=>{});}
function stop(){keys.clear();send();}
pad.addEventListener('keydown',e=>{if(['w','a','d',' ','s'].includes(e.key.toLowerCase())){e.preventDefault();if(e.key===' '||e.key.toLowerCase()==='s')stop();else keys.add(e.key.toLowerCase());send();}});
pad.addEventListener('keyup',e=>{keys.delete(e.key.toLowerCase());send();});
window.addEventListener('blur',stop);document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
document.querySelectorAll('button').forEach(b=>{b.onpointerdown=e=>{b.setPointerCapture(e.pointerId);keys.add(b.dataset.key);send();};b.onpointerup=()=>{keys.delete(b.dataset.key);send();};b.onpointercancel=stop;});
setInterval(send,100);pad.focus();</script></html>""".replace("FORWARD", str(forward_speed)).replace("TURN", str(turn_speed)).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if self.path != "/command" or not 0 < length <= 256:
                    raise ValueError("Invalid command request")
                mailbox.put(json.loads(self.rfile.read(length)))
            except (ValueError, TypeError, OverflowError):
                self.send_error(400)
                return
            self.send_response(204)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
