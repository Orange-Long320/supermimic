"""Local-only keyboard input for an explicitly supervised X2 trial.

Heartbeat expiry selects the stand reference; it is not a motor or host-loss
failsafe. The listener accepts requests only from its own local web page.
"""
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

SKILLS = ('stand', 'walk', 'backward', 'jog', 'turn_left_inplace', 'turn_right_inplace')


class Keyboard:
    def __init__(self, port):
        self.mode, self.when, self.ready, self.ready_when = 'stand', 0., False, 0.
        self.lock = threading.Lock()
        self.token = secrets.token_hex(16)
        mailbox = self

        class Handler(BaseHTTPRequestHandler):
            def _local_host(self):
                try:
                    url = urlsplit('http://' + self.headers.get('Host', ''))
                    return url.hostname in ('127.0.0.1', 'localhost') and url.port == self.server.server_port
                except ValueError:
                    return False

            def _origin(self):
                try:
                    url = urlsplit(self.headers.get('Origin', ''))
                    return url.scheme == 'http' and url.hostname in ('127.0.0.1', 'localhost') and url.port == self.server.server_port
                except ValueError:
                    return False

            def do_GET(self):
                if not self._local_host() or self.path != '/':
                    self.send_error(404); return
                html = '''<!doctype html><meta charset="utf-8"><title>X2 keyboard</title>
<h1>X2 keyboard</h1><p>Focus this page. Hold W: forward, S: backward, A/D: turn, Shift+W: jog.</p>
<p>Release keys, leave the page or lose connection: request stand.</p><p id="mode">Waiting for policy</p>
<script>
const keys=new Set(), token=TOKEN;
let inflight=false, pending=false;
function current(){if(keys.has('KeyA'))return 'turn_left_inplace';if(keys.has('KeyD'))return 'turn_right_inplace';if(keys.has('KeyS'))return 'backward';if(keys.has('KeyW'))return keys.has('ShiftLeft')||keys.has('ShiftRight')?'jog':'walk';return 'stand'}
async function send(){pending=true;if(inflight)return;inflight=true;
 while(pending){pending=false;const mode=current();const controller=new AbortController();const timer=setTimeout(()=>controller.abort(),300);
 try{const r=await fetch('/mode',{method:'POST',body:mode,headers:{'X-Control-Token':token},signal:controller.signal,cache:'no-store'});document.querySelector('#mode').textContent=r.status===409?'Waiting for policy':r.ok?mode:'Connection error';}catch(_){document.querySelector('#mode').textContent='Connection error';}finally{clearTimeout(timer);}}inflight=false;}
onkeydown=e=>{if(['KeyW','KeyS','KeyA','KeyD','ShiftLeft','ShiftRight'].includes(e.code)){e.preventDefault();keys.add(e.code);send()}};
onkeyup=e=>{keys.delete(e.code);send()};function stop(){keys.clear();send()};onblur=stop;document.addEventListener('visibilitychange',()=>{if(document.hidden)stop()});setInterval(send,100);
</script>'''.replace('TOKEN', '"' + mailbox.token + '"').encode()
                self.send_response(200); self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Cache-Control', 'no-store'); self.send_header('Content-Length', str(len(html)))
                self.end_headers(); self.wfile.write(html)

            def do_POST(self):
                if not self._local_host() or not self._origin() or self.headers.get('X-Control-Token') != mailbox.token or self.path != '/mode':
                    self.send_error(403); return
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length < 32: raise ValueError('invalid length')
                    mode = self.rfile.read(length).decode('ascii')
                    if mode not in SKILLS: raise ValueError('invalid mode')
                except (ValueError, UnicodeError):
                    self.send_error(400); return
                with mailbox.lock:
                    if not mailbox.ready or time.monotonic()-mailbox.ready_when >= .5:
                        self.send_error(409); return
                    mailbox.mode, mailbox.when = mode, time.monotonic()
                self.send_response(204); self.end_headers()

            def log_message(self, *args): pass

        self.server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def set_ready(self, ready):
        if type(ready) is not bool: raise ValueError('ready must be bool')
        with self.lock:
            self.ready = ready
            self.ready_when = time.monotonic() if ready else 0.
            if not ready: self.mode, self.when = 'stand', 0.

    def read(self):
        with self.lock:
            now=time.monotonic()
            return self.mode if self.ready and now-self.ready_when < .5 and now-self.when < .5 else 'stand'

    def close(self):
        self.set_ready(False)
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.)
