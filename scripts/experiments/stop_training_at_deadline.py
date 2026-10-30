"""Stop one identified local training process at an authorized deadline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pid', type=int, required=True)
    p.add_argument('--started-epoch', type=float, required=True)
    p.add_argument('--hours', type=float, required=True)
    p.add_argument('--expected-command', required=True)
    p.add_argument('--checkpoint-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.hours <= 0 or a.out.exists():
        raise ValueError('Positive hours and a new result file are required')
    proc = Path('/proc') / str(a.pid)
    initial_stat = (proc / 'stat').read_text().split()
    start_ticks = initial_stat[21]
    if proc.stat().st_uid != os.getuid() or a.expected_command not in (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode():
        raise ValueError('Training process owner or command mismatch')
    deadline = a.started_epoch + a.hours * 3600
    outcome = {'pid': a.pid, 'start_ticks': start_ticks, 'deadline_epoch': deadline,
               'checkpoint_dir': str(a.checkpoint_dir), 'status': 'watching'}
    def save():
        tmp = a.out.with_suffix('.tmp')
        tmp.write_text(json.dumps(outcome, indent=2) + '\n')
        tmp.replace(a.out)
    save()
    while time.time() < deadline:
        if not proc.exists() or (proc / 'stat').read_text().split()[21] != start_ticks:
            outcome.update(status='process_exited_before_deadline', observed_epoch=time.time())
            save()
            return
        time.sleep(min(10., max(0., deadline-time.time())))
    if not proc.exists() or (proc / 'stat').read_text().split()[21] != start_ticks:
        outcome.update(status='process_exited_before_deadline', observed_epoch=time.time())
        save()
        return
    # A numbered snapshot could be in the middle of torch.save. Choose only a
    # numbered file that has been untouched for 30 seconds, before signalling.
    candidates = sorted(p for p in a.checkpoint_dir.glob('ckpt_[0-9]*.pt') if p.stat().st_mtime < time.time()-30)
    if not candidates:
        outcome.update(status='no_checkpoint_stop_not_sent', observed_epoch=time.time())
        save()
        raise RuntimeError('No saved checkpoint; did not stop training')
    checkpoint = candidates[-1]
    os.kill(a.pid, signal.SIGTERM)
    outcome.update(status='stop_signal_sent', signal='SIGTERM', signalled_epoch=time.time(), checkpoint=str(checkpoint),
                   checkpoint_note='Periodic snapshot; final in-memory updates may be unsaved')
    save()
    for _ in range(30):
        if not proc.exists() or (proc / 'stat').read_text().split()[2] == 'Z':
            break
        time.sleep(1)
    outcome['process_exited'] = not proc.exists() or (proc / 'stat').read_text().split()[2] == 'Z'
    if checkpoint.is_file():
        h = hashlib.sha256()
        with checkpoint.open('rb') as f:
            for chunk in iter(lambda: f.read(1024*1024), b''):
                h.update(chunk)
        outcome['checkpoint_sha256'] = h.hexdigest()
        outcome['checkpoint_mtime'] = checkpoint.stat().st_mtime
    outcome['status'] = 'stopped' if outcome['process_exited'] else 'stop_not_confirmed'
    save()


if __name__ == '__main__':
    main()
