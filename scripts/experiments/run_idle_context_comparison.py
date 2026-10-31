"""Bounded official-simulator CPU comparison in an existing isolated fixture.

The fixture owns a copied simulator configuration, binary working directory,
ROS environment and startup script. Only child process groups started here are
stopped. This does not use a GPU or connect to hardware.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('run', 'dataset', 'policy'):
        p.add_argument('--'+n, type=Path, required=True)
    p.add_argument('--domain', type=int, default=201)
    p.add_argument('--state-port', type=int, default=19279)
    p.add_argument('--command-port', type=int, default=19280)
    p.add_argument('--seconds', type=float, default=55.)
    p.add_argument('--trials', nargs='+', default=['baseline', 'context-idle'],
                   choices=('baseline', 'context-idle', 'context-baseline-idle', 'context-baseline-motion',
                            'context-long-join', 'context-no-entry-skip'))
    a = p.parse_args()
    root = a.run/'source'
    if not 128 <= a.domain <= 232 or not 40 <= a.seconds <= 90 or len(set(a.trials)) != len(a.trials):
        raise ValueError('Invalid bounded simulation configuration')
    schedule = [[0., 'stand'], [3., 'walk'], [6.6, 'stand'], [13., 'walk'],
                [17., 'stand'], [23.4, 'walk'], [27.8, 'stand']]
    schedule_path = a.run/'schedule.json'
    with schedule_path.open('x') as f:
        json.dump(schedule, f)
    profile = root/'assets/robots/x2_ultra/profile.json'
    manifest = json.loads((a.dataset/'manifest.json').read_text())
    for clip in manifest['clips']:
        if sha(a.dataset/(clip['name']+'.npz')) != clip['sha256']:
            raise ValueError('Reference SHA mismatch: '+clip['name'])
    exported = json.loads(a.policy.with_name('manifest.json').read_text())
    if exported['checkpoint_sha256'] != 'ff1f91d33631530d9d7f90f9f0675359fa1bff18445df43449312b9cb85db34b':
        raise ValueError('Expected the original five-skill 11500 checkpoint')
    if sha(a.policy) != '4424f231fc1896da098b2fca366e8522df6791596638b2b9257ba682520bbe4c':
        raise ValueError('Frozen policy SHA mismatch')
    raw = subprocess.check_output(['bash', '-c', 'source "$1"; env -0', 'ros-env', str(a.run/'ros-env.sh')])
    env = dict(item.decode().split('=', 1) for item in raw.split(b'\0') if item)
    env.update(ROS_DOMAIN_ID=str(a.domain), ROS_LOCALHOST_ONLY='1', CUDA_VISIBLE_DEVICES='',
               LIBGL_ALWAYS_SOFTWARE='1', GALLIUM_DRIVER='llvmpipe', OMP_NUM_THREADS='2',
               LP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', SM_ROBOT='x2_ultra')
    record = dict(status='initializing', pid=os.getpid(), argv=sys.argv, domain=a.domain,
                  device='CPU; Mesa llvmpipe requested', policy_sha256=sha(a.policy),
                  dataset_manifest_sha256=sha(a.dataset/'manifest.json'), trials=[])
    result_path = a.run/'comparison.json'
    def save():
        tmp = result_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(record, indent=2)+'\n')
        tmp.replace(result_path)
    save()
    for trial_name in a.trials:
        planner = 'baseline' if trial_name == 'baseline' else 'context-idle'
        ablation = {'context-baseline-idle': 'baseline-idle', 'context-baseline-motion': 'baseline-motion'}.get(trial_name, 'none')
        diagnostic = ['--planner-ablation', ablation]
        if trial_name == 'context-long-join':
            diagnostic += ['--idle-join-frames', '30']
        if trial_name == 'context-no-entry-skip':
            diagnostic += ['--idle-entry-offset', '0']
        # Bind both unused UDP endpoints before launching this trial.
        for port in (a.state_port, a.command_port):
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.bind(('127.0.0.1', port))
        trial = a.run/trial_name
        trial.mkdir()
        children, handles = [], []
        entry = dict(planner=trial_name, controller=planner, ablation=ablation, started_epoch=time.time(), processes=[])
        record['trials'].append(entry)
        record['status'] = trial_name
        def launch(name, argv, child_env, cwd=None):
            log = (trial/(name+'.log')).open('x')
            handles.append(log)
            process = subprocess.Popen(argv, env=child_env, cwd=cwd, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            children.append(process)
            entry['processes'].append(dict(name=name, pid=process.pid, argv=argv))
            save()
            return process
        try:
            common = ['--sim-only', '--ros-domain', str(a.domain)]
            launch('command', ['/usr/bin/python3', str(root/'scripts/deploy/official_sim_command.py'),
                              '--profile', str(profile), '--port', str(a.command_port), *common], env)
            launch('state', ['/usr/bin/python3', str(root/'scripts/deploy/official_sim_state.py'),
                            '--port', str(a.state_port), *common], env)
            pyenv = dict(env, PYTHONPATH=str(root/'src'))
            policy = launch('policy', [sys.executable, str(root/'scripts/deploy/official_sim_policy.py'),
                '--profile', str(profile), '--dataset', str(a.dataset), '--policy', str(a.policy),
                '--out', str(trial/'policy'), '--state-port', str(a.state_port),
                '--command-port', str(a.command_port), '--seconds', str(a.seconds),
                '--schedule', str(schedule_path), '--planner', planner, '--compact-log', *diagnostic, *common], pyenv, str(root))
            deadline = time.monotonic()+40
            while 'policy ready for simulator' not in (trial/'policy.log').read_text():
                if policy.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('Policy initialization failed; see '+str(trial/'policy.log'))
                time.sleep(.2)
            simulator = launch('sim', ['bash', str(a.run/'start-sim.sh')], env)
            code = policy.wait(timeout=a.seconds+15)
            entry.update(exit_code=code, simulator_exit_code_at_policy_end=simulator.poll(),
                         result=json.loads((trial/'policy/result.json').read_text()))
        finally:
            # SIGTERM to this task's own process groups only. Every Popen above
            # starts a new session, so no caller or unrelated process is in it.
            for process in reversed(children):
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            for process in children:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            for handle in handles:
                handle.close()
            entry['finished_epoch'] = time.time()
            save()
        # Both results are retained even if one planner fails physically.
    record['status'] = 'completed'
    record['scope'] = 'one deterministic sequence per planner, official simulator; no policy training or physical acceptance'
    save()


if __name__ == '__main__':
    main()
