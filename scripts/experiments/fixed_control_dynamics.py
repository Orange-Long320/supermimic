"""Fixed real control transitions for phase-balanced world-model fitting and gates.

Collection and source provenance must be verified by the caller. This module
never creates an environment, starts training, or changes a world model.
"""
from pathlib import Path
import hashlib
import json

import torch

from supermimic.kinematics.rotations import heading_yaw
from supermimic.locomotion.task import measured_velocity
from supermimic.robots.dimensions import NQ, STATE_DIM, XI_DIM


PHASES = ((0., 'initial_stop'), (3., 'forward'), (10., 'forward_stop'),
          (15., 'backward'), (22., 'backward_stop'), (27., 'left'),
          (34., 'left_stop'), (39., 'right'), (46., 'right_stop'),
          (51., 'jog'), (58., 'jog_stop'))


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class FixedControlDynamics:
    """CPU-backed 66s trajectory with disjoint, fixed environment splits.

    report_sha256 is an independently supplied digest, not a digest inferred
    from the report itself. Time/flags establish recorded continuity; the format
    does not contain an independent post-state for each transition, so it cannot
    separately prove physical state continuity beyond the collector's alignment.
    """

    def __init__(self, raw_path, retention_path, report_path, *, source_sha256,
                 reference_sha256, standing_pose_sha256, report_sha256, dt, device, seed=29):
        self.raw_path = Path(raw_path).resolve()
        self.retention_path = Path(retention_path).resolve()
        self.report_path = Path(report_path).resolve()
        if dt <= 0:
            raise ValueError('dt must be positive')
        for name, value in (('source', source_sha256), ('reference', reference_sha256),
                            ('pose', standing_pose_sha256), ('report', report_sha256)):
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError(f'Invalid {name} SHA256')
        if _sha(self.report_path) != report_sha256:
            raise ValueError('Collector report SHA256 mismatch')
        report = json.loads(self.report_path.read_text())
        raw_hash, retention_hash = _sha(self.raw_path), _sha(self.retention_path)
        if report.get('status') != 'collected' or report.get('raw_sha256') != raw_hash or report.get('retention_sha256') != retention_hash:
            raise ValueError('Raw/retention files do not match a completed collector report')
        expected = {'source_sha256': source_sha256, 'reference_sha256': reference_sha256,
                    'standing_pose_sha256': standing_pose_sha256}
        raw = torch.load(self.raw_path, map_location='cpu', weights_only=False)
        retention = torch.load(self.retention_path, map_location='cpu', weights_only=False)
        if (_sha(self.report_path) != report_sha256 or _sha(self.raw_path) != raw_hash
                or _sha(self.retention_path) != retention_hash):
            raise RuntimeError('A collector artifact changed while loading')
        for name, item in (('report', report), ('raw', raw), ('retention', retention)):
            if not isinstance(item, dict) or any(item.get(key) != value for key, value in expected.items()):
                raise ValueError(f'{name} source/reference/pose SHA mismatch')
        mode = report.get('preserve_moving_feedback')
        if not isinstance(mode, bool) or raw.get('preserve_moving_feedback') is not mode or retention.get('preserve_moving_feedback') is not mode:
            raise ValueError('Control-policy collection movement modes differ')
        if (raw.get('moving_projection_sha256') != report.get('moving_projection_sha256')
                or retention.get('moving_projection_sha256') != report.get('moving_projection_sha256')
                or (mode and not report.get('moving_projection_sha256'))):
            raise ValueError('Moving projection provenance differs')
        if raw.get('walk_sha256') != report.get('walk_sha256') or not report.get('walk_sha256'):
            raise ValueError('Dataset provenance differs')
        if abs(float(raw.get('dt', 0)) - dt) > 1e-8 or abs(float(report.get('dt', 0)) - dt) > 1e-8:
            raise ValueError('Collection and requested time steps differ')
        needed = ('state_before', 'final_state', 'action', 'executed_action', 'command', 'feedback',
                  'xi', 'phase_start_s', 'time_s', 'terminated', 'timeout')
        if any(not isinstance(raw.get(k), torch.Tensor) for k in needed):
            raise ValueError('Raw collection lacks required tensors')
        states, actions, xi = raw['state_before'], raw['executed_action'], raw['xi']
        if states.ndim != 3 or actions.ndim != 3 or actions.shape[:2] != states.shape[:2]:
            raise ValueError('State/action transitions are not aligned')
        count, envs = states.shape[:2]
        if states.shape[-1] != STATE_DIM or actions.shape[-1] != NQ - 7 or xi.shape != (envs, XI_DIM):
            raise ValueError('State, action or context dimensions differ from the robot profile')
        if (raw['action'].shape != (count, envs, NQ - 7) or raw['command'].shape != (count, envs, 3)
                or raw['feedback'].shape != (count, envs, 6)):
            raise ValueError('Policy actions, commands or feedback do not align with recorded states')
        if not isinstance(raw.get('env_id'), torch.Tensor) or not torch.equal(raw['env_id'], torch.arange(envs)):
            raise ValueError('Raw environment identifiers are inconsistent')
        if envs < 4 or count != round(66. / dt) or abs(count * dt - 66.) > 1e-6:
            raise ValueError('Need at least four environments and an entire 66-second trajectory')
        if raw['final_state'].shape != states.shape[1:] or xi.ndim != 2 or xi.shape[0] != envs:
            raise ValueError('Final state or environment contexts have invalid dimensions')
        if raw['terminated'].shape != (count, envs) or raw['timeout'].shape != (count, envs):
            raise ValueError('Terminal flags do not align with transitions')
        if raw['terminated'].any() or raw['timeout'].any() or report.get('actual_steps') != count or report.get('raw_transitions') != count:
            raise ValueError('Trajectory includes a termination, timeout or incomplete collection')
        if raw['time_s'].shape != (count,) or raw['phase_start_s'].shape != (count,):
            raise ValueError('Time and phase vectors do not align with transitions')
        time = raw['time_s'].to(torch.float64)
        if not torch.allclose(time, torch.arange(count, dtype=torch.float64) * dt, atol=1e-7, rtol=0):
            raise ValueError('Trajectory timestamps contain a gap or do not start at zero')
        expected_phase = torch.zeros_like(time)
        for start, _ in PHASES:
            expected_phase[time >= start] = start
        if not torch.equal(raw['phase_start_s'].to(torch.float64), expected_phase):
            raise ValueError('Recorded phase labels do not match the fixed command schedule')
        if any(not torch.isfinite(t).all() for t in (states, raw['final_state'], actions, raw['action'], raw['command'], raw['feedback'], xi, time)):
            raise ValueError('Non-finite trajectory data')
        if any(t.dtype != torch.float32 for t in (states, raw['final_state'], actions, raw['action'], raw['command'], raw['feedback'], xi)):
            raise ValueError('World-model and policy transition data must be float32')
        if report.get('num_envs') != envs:
            raise ValueError('Report environment count does not match raw data')
        self.states = torch.cat((states, raw['final_state'][None]), dim=0).contiguous()
        self.actions, self.xi = actions.contiguous(), xi.contiguous()
        self.policy_actions = raw['action'].contiguous()
        self.commands = raw['command'].contiguous()
        self.feedback = raw['feedback'].contiguous()
        self.dt, self.device, self.horizon = float(dt), torch.device(device), 32
        self.train_env = torch.arange(envs)[torch.arange(envs) % 4 != 3]
        self.heldout_env = torch.arange(envs)[torch.arange(envs) % 4 == 3]
        if not len(self.train_env) or not len(self.heldout_env):
            raise ValueError('Both training and held-out environment splits must be nonempty')
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        zero = self.commands.square().sum(-1) < 1e-6
        self.stop_ages = torch.zeros(count, envs, dtype=torch.float32)
        consecutive = torch.zeros(envs, dtype=torch.long)
        for t in range(count):
            consecutive = torch.where(zero[t], consecutive + 1, torch.zeros_like(consecutive))
            self.stop_ages[t] = consecutive * self.dt
        self.stop_pairs = []
        for phase_start in (10., 22., 34., 46., 58.):
            phase_mask = (expected_phase == phase_start)[:, None] & zero
            # Runtime history length is checked again by sample_stop_starts.
            indices = phase_mask[:, self.train_env].nonzero()
            self.stop_pairs.append(torch.stack((self.train_env[indices[:, 1]], indices[:, 0]), -1))
        self.train_pairs = []
        self.heldout_pairs = []
        for i, (phase_start, name) in enumerate(PHASES):
            end = PHASES[i + 1][0] if i + 1 < len(PHASES) else 66.
            # Assign by starting phase, including windows that cross a command boundary.
            starts = torch.arange(count - self.horizon + 1)
            starts = starts[(time[starts] >= phase_start) & (time[starts] < end)]
            if not len(starts):
                raise ValueError(f'No complete 32-step windows in {name}')
            def pairs(env_ids, frame_starts):
                return torch.stack((env_ids.repeat_interleave(len(frame_starts)), frame_starts.repeat(len(env_ids))), dim=-1)
            self.train_pairs.append(pairs(self.train_env, starts))
            held_starts = starts[expected_phase[starts + self.horizon - 1] == phase_start]
            if not len(held_starts):
                raise ValueError(f'No within-phase held-out windows in {name}')
            held = pairs(self.heldout_env, held_starts)
            if len(held) > 32:
                index = torch.linspace(0, len(held) - 1, 32).round().long()
                held = held[index]
            self.heldout_pairs.append(held)
        self.provenance = {'raw_sha256': raw_hash, 'retention_sha256': retention_hash,
                           'report_sha256': report_sha256, **expected, 'dt': dt,
                           'train_env_ids': self.train_env.tolist(), 'heldout_env_ids': self.heldout_env.tolist(),
                           'train_windows_by_phase': {name: len(self.train_pairs[i]) for i, (_, name) in enumerate(PHASES)},
                           'heldout_windows_by_phase': {name: len(self.heldout_pairs[i]) for i, (_, name) in enumerate(PHASES)},
                           'source_post_stop_samples_by_phase': {str(start): len(self.stop_pairs[i]) for i, start in enumerate((10., 22., 34., 46., 58.))},
                           'source_stop_sampling': 'real zero-command source states; train envs only; balanced by post-movement stop phase',
                           'phase_assignment': 'train by window start, including boundaries; held-out windows stay within their phase',
                           'collection_seed': report.get('seed'), 'sampler_seed': seed,
                           'preserve_moving_feedback': mode,
                           'continuity_scope': 'collector time, termination and timeout records; post-state is not independently recorded per step'}

    def _windows(self, pairs):
        env_ids, starts = pairs[:, 0], pairs[:, 1]
        times = starts[None, :] + torch.arange(self.horizon + 1)[:, None]
        env_index = env_ids[None, :].expand(self.horizon + 1, -1)
        states = self.states[times, env_index]
        actions = self.actions[times[:-1], env_index[:-1]]
        return states.to(self.device), actions.to(self.device), self.xi[env_ids].to(self.device)

    def sample_train(self, batch_size):
        """Return states (33,N,S), executed actions (32,N,A), xi (N,X)."""
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        # Random phase order makes the count per phase differ by at most one.
        phase_order = torch.randperm(len(PHASES), generator=self.generator)
        selected = []
        for j in range(batch_size):
            phase = int(phase_order[j % len(PHASES)])
            pool = self.train_pairs[phase]
            index = int(torch.randint(len(pool), (1,), generator=self.generator))
            selected.append(pool[index])
        return self._windows(torch.stack(selected))

    def sample_stop_starts(self, batch_size, history):
        """Return true source post-movement stop starts for StopImaginationTrainer.

        Ages and anchors come from the recorded command/feedback context. These
        source states do not demonstrate that an updated policy passes physics.
        """
        if batch_size < 1 or history < 1:
            raise ValueError('batch_size and history must be positive')
        pools = [pairs[pairs[:, 1] >= history + 1] for pairs in self.stop_pairs]
        if any(not len(pool) for pool in pools):
            raise ValueError('At least one post-movement phase has no eligible source starts')
        phase_order = torch.randperm(len(pools), generator=self.generator)
        selected = []
        for j in range(batch_size):
            pool = pools[int(phase_order[j % len(pools)])]
            selected.append(pool[int(torch.randint(len(pool), (1,), generator=self.generator))])
        pairs = torch.stack(selected)
        env_ids, starts = pairs[:, 0], pairs[:, 1]
        times = starts[None, :] + torch.arange(history)[:, None] - (history - 1)
        env_index = env_ids[None, :].expand(history, -1)
        hist_states = self.states[times, env_index]
        hist_actions = self.policy_actions[times - 1, env_index]
        state = self.states[starts, env_ids]
        feedback = self.feedback[starts, env_ids]
        yaw = heading_yaw(state[:, 3:7])
        c, s = torch.cos(yaw), torch.sin(yaw)
        dx, dy = feedback[:, 3], feedback[:, 4]
        anchor_xy = state[:, :2] + torch.stack((c * dx - s * dy, s * dx + c * dy), -1)
        anchor_yaw = torch.atan2(torch.sin(yaw + feedback[:, 5]), torch.cos(yaw + feedback[:, 5]))
        return {'hist_states': hist_states.to(self.device), 'hist_actions': hist_actions.to(self.device),
                'xi': self.xi[env_ids].to(self.device), 'anchor_xy': anchor_xy.to(self.device),
                'anchor_yaw': anchor_yaw.to(self.device), 'stop_age': self.stop_ages[starts, env_ids].to(self.device),
                'is_cold': torch.zeros(batch_size, dtype=torch.bool, device=self.device)}

    @torch.no_grad()
    def heldout_gate(self, wm):
        """Measure fixed per-phase errors and apply the original 8/32-step thresholds."""
        if abs(float(wm.spec.dt) - self.dt) > 1e-8:
            raise ValueError('World-model dt differs from collection')
        phases = {}
        passed = True
        for i, (_, name) in enumerate(PHASES):
            states, actions, xi = self._windows(self.heldout_pairs[i])
            pred = states[0]
            velocity, pose = {}, {}
            for step in range(self.horizon):
                pred = wm.step(pred, actions[step], xi)
                if not torch.isfinite(pred).all():
                    raise FloatingPointError(f'Non-finite world-model prediction in {name}')
                if step + 1 in (1, 8, 32):
                    actual = states[step + 1]
                    velocity[str(step + 1)] = (measured_velocity(pred) - measured_velocity(actual)).square().mean(0).sqrt().cpu().tolist()
                    dh = heading_yaw(pred[:, 3:7]) - heading_yaw(actual[:, 3:7])
                    dh = torch.atan2(torch.sin(dh), torch.cos(dh))
                    pose[str(step + 1)] = [float((pred[:, :2] - actual[:, :2]).square().sum(-1).mean().sqrt()),
                                           float(dh.square().mean().sqrt())]
            ok = (all(x < y for x, y in zip(velocity['8'], (.10, .10, .15)))
                  and all(x < y for x, y in zip(velocity['32'], (.20, .20, .30)))
                  and all(x < y for x, y in zip(pose['32'], (.03, .08))))
            phases[name] = {'windows': len(self.heldout_pairs[i]), 'velocity_rmse': velocity,
                            'pose_rmse': pose, 'passed': ok}
            passed = passed and ok
        return {'passed': passed, 'phases': phases,
                'thresholds': {'velocity_8': [.10, .10, .15], 'velocity_32': [.20, .20, .30],
                               'position_heading_32': [.03, .08]},
                'scope': 'fixed held-out real transitions by phase; model gate only, not physical policy acceptance'}
