"""Wait for a named project experiment and an idle GPU before one bounded run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--after-controller', type=Path, required=True)
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=2)
    p.add_argument('--wait-hours', type=float, default=2.)
    a = p.parse_args()
    expected = Path('/home/2025-Long.xh/HHD/venvs/supermimic')
    if Path(sys.prefix).resolve() != expected.resolve() or not 0 < a.wait_hours <= 4:
        p.error('Dedicated server environment and bounded wait required')
    request = json.loads(a.request.read_text())
    argv = request['argv']
    source = a.request.parent.resolve()
    if Path(argv[0]).resolve() != (expected/'bin/python').resolve() or Path(argv[1]).resolve() != source/'scripts/run_supermimic_stop.py':
        raise ValueError('Only the pinned SuperMimic controller may be queued')
    entry_sha = hashlib.sha256(Path(argv[1]).read_bytes()).hexdigest()
    state_file = source/'queue.json'
    if state_file.exists():
        raise FileExistsError(state_file)
    terminal = {'paired_evaluation_complete', 'not_accepted', 'accepted', 'failed',
                'training_failed', 'training_timeout', 'model_not_ready_no_actor_updates',
                'insufficient_actor_updates_for_ema'}
    started = time.monotonic()
    state = dict(status='waiting', pid=os.getpid(), gpu=a.gpu, argv=argv,
                 started_epoch=time.time(), after_controller=str(a.after_controller),
                 controller_sha256=entry_sha)
    def save():
        tmp = state_file.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2)+'\n'); tmp.replace(state_file)
    save()
    previous = None
    try:
        while time.monotonic()-started < a.wait_hours*3600:
            parent = json.loads(a.after_controller.read_text())
            occupied = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
                '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
            resources = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
                '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
            memory, utilization = map(float, resources.split(','))
            marker = (parent['status'], occupied)
            state.update(after_status=parent['status'], observed_gpu_pids=occupied, gpu_resources=resources)
            if marker != previous:
                save(); print(json.dumps(state), flush=True); previous = marker
            if parent['status'] in terminal and not occupied and memory <= 100 and utilization == 0:
                break
            time.sleep(15)
        else:
            state.update(status='wait_limit_no_launch', finished_epoch=time.time()); save(); return
        if hashlib.sha256(Path(argv[1]).read_bytes()).hexdigest() != entry_sha:
            raise RuntimeError('Queued source changed while waiting')
        with (source/'queued-controller.log').open('x') as log:
            child = subprocess.Popen(argv, cwd=source, stdout=log, stderr=subprocess.STDOUT)
            state.update(status='running', controller_pid=child.pid, launched_epoch=time.time()); save()
            state['returncode'] = child.wait()
        state.update(status='finished', finished_epoch=time.time()); save()
    except BaseException as error:
        state.update(status='queue_failed', error=repr(error), finished_epoch=time.time()); save(); raise


if __name__ == '__main__':
    main()
