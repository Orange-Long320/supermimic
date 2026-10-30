"""Diagnostic standing-anchor latch after measured braking, using a fixed evaluator."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('evaluator', 'checkpoint', 'npz-dir', 'stand-reference', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--seconds', type=float, default=66.)
    p.add_argument('--num-envs', type=int, default=8)
    p.add_argument('--planar-threshold', type=float, default=.03)
    p.add_argument('--yaw-threshold', type=float, default=.05)
    p.add_argument('--joint-rms-threshold', type=float, default=.3)
    p.add_argument('--stable-seconds', type=float, default=.2)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--standing-only', action='store_true')
    modes.add_argument('--direct-switches', action='store_true')
    modes.add_argument('--extended-stop', action='store_true')
    a = p.parse_args()
    python = Path('/home/2025-Long.xh/HHD/venvs/supermimic/bin/python')
    if Path(sys.prefix).resolve() != python.parent.parent.resolve() or os.path.realpath(sys.executable) != os.path.realpath(python):
        p.error(f'Use the dedicated server interpreter: {python}')
    if (a.gpu < 0 or a.num_envs < 2 or a.seconds <= 0 or a.out.exists()
            or any(not math.isfinite(v) or v <= 0 for v in (a.planar_threshold, a.yaw_threshold, a.joint_rms_threshold, a.stable_seconds))):
        p.error('Invalid GPU, duration, thresholds or output path')
    for path in (a.evaluator, a.checkpoint, a.npz_dir / 'walk.npz', a.stand_reference):
        if not path.is_file():
            p.error(f'Missing input: {path}')
    active = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-compute-apps=pid',
                                      '--format=csv,noheader'], text=True).strip()
    gpu_info = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
        '--query-gpu=index,name,uuid,driver_version,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).strip()
    parts = [x.strip() for x in gpu_info.split(',')]
    if len(parts) != 6 or int(parts[0]) != a.gpu or active or float(parts[4]) > 100 or float(parts[5]) != 0:
        raise RuntimeError('Requested GPU is not idle; diagnostic not started')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    os.environ['SM_ROBOT'] = 'x2_ultra'
    os.environ['MUJOCO_GL'] = 'egl'
    for key in ('PYTHONHOME', 'SM_ROBOT_SPEC', 'SM_SOLE_BOX_CONFIG', 'SM_GPU_PINNED'):
        os.environ.pop(key, None)
    import torch
    from supermimic.kinematics.rotations import heading_yaw
    from supermimic.locomotion.task import measured_velocity
    from supermimic.robots.dimensions import NQ
    import supermimic.policy.standing as standing_module

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected one visible CUDA device')
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    reference_sha = sha(a.stand_reference)
    if ck.get('standing_reference', {}).get('sha256') != reference_sha:
        raise ValueError('Standing reference does not match checkpoint')
    if not ck.get('standing_skill', {}).get('feedback_dim'):
        raise ValueError('Checkpoint must use standing feedback')
    del ck
    base = standing_module.StandingContext
    instances = []

    class BrakeLatchContext(base):
        def __init__(self, command, env, radius=.1):
            super().__init__(command, env, radius)
            self.mode = self.stable_count = self.braking_steps = None
            self.started_s = None
            self.events = []
            self.required_steps = max(1, math.ceil(a.stable_seconds / env.step_dt - 1e-9))
            instances.append(self)

        def close_attempts(self, mask, reason):
            for i in mask.nonzero().flatten().tolist():
                self.events.append(dict(env_id=i, brake_start_s=float(self.started_s[i]), latch_s=None,
                                        braking_steps=int(self.braking_steps[i]), outcome=reason))

        @torch.no_grad()
        def reset(self, env_ids):
            if self.mode is not None:
                ids = torch.as_tensor(env_ids, device=self.mode.device, dtype=torch.long)
                mask = torch.zeros_like(self.mode, dtype=torch.bool)
                mask[ids] = self.mode[ids] == 2
                self.close_attempts(mask, 'reset_before_latch')
                self.mode[ids] = 0; self.stable_count[ids] = 0; self.braking_steps[ids] = 0
                self.started_s[ids] = 0
            super().reset(env_ids)

        @torch.no_grad()
        def features(self, command):
            state = self.command.state_io.read()
            yaw = heading_yaw(state[:, 3:7])
            age = self.env.episode_length_buf
            if self.anchor_xy is None:
                self.anchor_xy = state[:, :2].clone(); self.anchor_yaw = yaw.clone(); self.previous_episode = age.clone()
                self.mode = torch.zeros(len(state), dtype=torch.int8, device=state.device)
                self.stable_count = torch.zeros(len(state), dtype=torch.long, device=state.device)
                self.braking_steps = torch.zeros_like(self.stable_count)
                self.started_s = torch.zeros(len(state), dtype=state.dtype, device=state.device)
            auto_reset = age < self.previous_episode
            if auto_reset.any():
                self.close_attempts(auto_reset & (self.mode == 2), 'reset_before_latch')
                self.mode[auto_reset] = 0; self.stable_count[auto_reset] = 0
                self.braking_steps[auto_reset] = 0; self.started_s[auto_reset] = 0
                self.anchor_xy[auto_reset] = state[auto_reset, :2]; self.anchor_yaw[auto_reset] = yaw[auto_reset]
            moving = command.square().sum(-1) >= self.radius ** 2
            self.close_attempts(moving & (self.mode == 2), 'movement_before_latch')
            enter = ~moving & (self.mode == 1)
            self.mode[moving] = 1; self.stable_count[moving] = 0; self.braking_steps[moving] = 0
            self.mode[enter] = 2; self.stable_count[enter] = 0; self.braking_steps[enter] = 0
            self.started_s[enter] = age[enter] * self.env.step_dt
            braking = self.mode == 2
            refresh = moving | braking
            self.anchor_xy[refresh] = state[refresh, :2]; self.anchor_yaw[refresh] = yaw[refresh]
            self.braking_steps[braking] += 1
            velocity = measured_velocity(state)
            joint_rms = state[:, NQ+6:].square().mean(-1).sqrt()
            stable = ((velocity[:, :2].norm(dim=-1) < a.planar_threshold)
                      & (velocity[:, 2].abs() < a.yaw_threshold) & (joint_rms < a.joint_rms_threshold))
            self.stable_count[braking] = torch.where(stable[braking], self.stable_count[braking]+1, 0)
            latch = braking & (self.stable_count >= self.required_steps)
            for i in latch.nonzero().flatten().tolist():
                self.events.append(dict(env_id=i, brake_start_s=float(self.started_s[i]),
                    latch_s=float(age[i] * self.env.step_dt), braking_steps=int(self.braking_steps[i]), outcome='latched'))
            self.mode[latch] = 3
            self.previous_episode = age.clone()
            delta = self.anchor_xy-state[:, :2]; c = torch.cos(yaw); s = torch.sin(yaw)
            position = torch.stack((c*delta[:,0]+s*delta[:,1], -s*delta[:,0]+c*delta[:,1]), -1)
            heading = torch.atan2(torch.sin(self.anchor_yaw-yaw), torch.cos(self.anchor_yaw-yaw))[:,None]
            return torch.cat((velocity, position, heading), -1)

    spec = importlib.util.spec_from_file_location('brake_latch_fixed_evaluator', a.evaluator)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    standing_module.StandingContext = BrakeLatchContext
    try:
        evaluator.main(ckpt=a.checkpoint, out=a.out, seconds=a.seconds, num_envs=a.num_envs,
            seed=a.seed, npz_dir=a.npz_dir, ieee=True, standing_only=a.standing_only,
            direct_switches=a.direct_switches, extended_stop=a.extended_stop)
    finally:
        standing_module.StandingContext = base
    events = []
    for context in instances:
        if context.mode is not None:
            context.close_attempts(context.mode == 2, 'never_latched_before_end')
        events.extend(context.events)
    result = json.loads(a.out.read_text())
    result.update(context_override=True, accepted=False)
    result['brake_latch_diagnostic'] = dict(checkpoint_sha256=sha(a.checkpoint), reference_sha256=reference_sha,
        evaluator_sha256=sha(a.evaluator), entry_sha256=sha(__file__), gpu=a.gpu, gpu_info=gpu_info,
        argv=sys.argv, source_commit=os.environ.get('SM_SOURCE_COMMIT', 'unspecified'),
        planar_threshold=a.planar_threshold, yaw_threshold=a.yaw_threshold,
        joint_rms_threshold=a.joint_rms_threshold, stable_seconds=a.stable_seconds,
        required_stable_steps=[c.required_steps for c in instances], events=events,
        latch_count=sum(e['outcome']=='latched' for e in events),
        never_latched_count=sum(e['outcome']!='latched' for e in events),
        scope='context override diagnostic only; never an accepted checkpoint')
    result['scope'] += ' Brake-latch context override diagnostic; not accepted.'
    a.out.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
