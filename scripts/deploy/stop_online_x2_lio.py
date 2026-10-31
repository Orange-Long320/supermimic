"""Stop only an owned online diagnostic session, after checking process identity."""
import argparse
import json
import os
from pathlib import Path
import signal
import time


def argv(pid):
    path = Path('/proc')/str(pid)/'cmdline'
    return [s.decode() for s in path.read_bytes().split(b'\0') if s] if path.exists() else []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    args = parser.parse_args(); session = args.session.resolve()
    pid = int((session/'bridge.pid').read_text())
    command = argv(pid)
    if command:
        if not any(Path(s).name == 'online_x2_lio.py' for s in command):
            raise RuntimeError('PID no longer belongs to the bridge')
        if '--out' not in command or Path(command[command.index('--out')+1]).resolve() != session:
            raise RuntimeError('Bridge session mismatch')
        os.kill(pid, signal.SIGINT)
        deadline = time.monotonic()+10
        while argv(pid) and time.monotonic()<deadline: time.sleep(.1)
        if argv(pid): raise RuntimeError('Bridge did not exit; inspect the owned session')
    child = int((session/'fastlio.pid').read_text()); command = argv(child)
    if command:
        expected = str(session/'fastlio-live.yaml')
        if '--params-file' not in command or command[command.index('--params-file')+1] != expected:
            raise RuntimeError('Child PID no longer belongs to this session')
        if not any(Path(s).name == 'fastlio_mapping' for s in command):
            raise RuntimeError('Unexpected child executable')
        os.kill(child, signal.SIGTERM)
    print(json.dumps({'session': str(session), 'bridge_pid': pid, 'owned_child_pid': child,
                      'scope': 'diagnostic processes only; no robot control command'}))


if __name__ == '__main__': main()
