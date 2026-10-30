"""Audit a new recorded control trajectory with a frozen world model on CPU.

No simulation, model update, or policy acceptance is performed.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint', 'raw', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    expected = Path('/home/2025-Long.xh/HHD/venvs/supermimic').resolve()
    if Path(sys.prefix).resolve() != expected:
        p.error('Use the dedicated server environment')
    if a.out.exists() or not a.checkpoint.is_file() or not a.raw.is_file():
        p.error('Inputs must exist and output must be a new path')
    import numpy as np
    import torch
    from supermimic.kinematics.rotations import heading_yaw
    from supermimic.locomotion.task import measured_velocity
    from supermimic.robots.dimensions import NQ, STATE_DIM, XI_DIM
    from supermimic.world_model.state_spec import StateSpecCfg
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    torch.set_num_threads(2)
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    raw = torch.load(a.raw, map_location='cpu', weights_only=False)
    meta = ck.get('supermimic_stop') or ck.get('supermimic_velocity_feedback') or ck.get('supermimic_composed_feedback') or {}
    source_sha, ref_sha, pose_sha = raw.get('source_sha256'), raw.get('reference_sha256'), raw.get('standing_pose_sha256')
    for key, value in (('source_sha256', source_sha), ('reference_sha256', ref_sha), ('standing_pose_sha256', pose_sha)):
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value.lower()):
            raise ValueError(f'Invalid raw provenance: {key}')
    raw_report_path = a.raw.parent / 'report.json'
    raw_report_sha = None
    raw_seed = None
    if raw_report_path.is_file():
        raw_report = json.loads(raw_report_path.read_text())
        if (raw_report.get('raw_sha256') != sha(a.raw)
                or any(raw_report.get(k) != raw.get(k) for k in
                       ('source_sha256', 'reference_sha256', 'standing_pose_sha256'))):
            raise ValueError('Raw collection report does not match raw SHA or input provenance')
        raw_seed = raw_report.get('seed')
        raw_report_sha = sha(raw_report_path)
    expected_ref = meta.get('reference_sha256') or ck.get('standing_reference', {}).get('sha256')
    if expected_ref != ref_sha:
        raise ValueError('Raw and checkpoint reference SHA differ')
    expected_pose = meta.get('standing_pose_sha256')
    if expected_pose and expected_pose != pose_sha:
        raise ValueError('Raw and checkpoint standing-pose SHA differ')
    wc = dict(ck['config']['world_model'])
    wc['state_spec'] = StateSpecCfg(**wc['state_spec'])
    wm = WorldModel(WorldModelCfg(**wc), None, 'cpu')
    missing, extra = wm.load_state_dict(ck['wm'], strict=False)
    if missing or any(not k.startswith('heads.') for k in extra):
        raise ValueError(f'Unexpected world model keys: {missing}, {extra}')
    wm.eval().requires_grad_(False)
    if not isinstance(raw.get('dt'), (int, float)) or not np.isfinite(raw['dt']) or raw['dt'] <= 0 or abs(raw['dt'] - wm.spec.dt) > 1e-8:
        raise ValueError('Raw control dt differs from world-model dt')
    required = ('state_before', 'final_state', 'executed_action', 'command', 'feedback', 'phase_start_s',
                'time_s', 'terminated', 'timeout', 'xi', 'env_id')
    if any(raw.get(k) is None for k in required):
        raise ValueError('Raw trajectory lacks required aligned data')
    states, final, actions, command, feedback, xi = (raw[k] for k in
        ('state_before', 'final_state', 'executed_action', 'command', 'feedback', 'xi'))
    phase, times, term, timeout, env_ids = (raw[k] for k in ('phase_start_s', 'time_s', 'terminated', 'timeout', 'env_id'))
    if states.ndim != 3:
        raise ValueError('Expected state_before [time, environment, state]')
    T, E, S = states.shape
    if (T < 32 or S != STATE_DIM or final.shape != (E, S) or actions.shape != (T, E, NQ-7)
            or command.shape != (T, E, 3) or feedback.shape != (T, E, 6) or xi.shape != (E, XI_DIM)
            or phase.shape != (T,) or times.shape != (T,) or term.shape != (T, E) or timeout.shape != (T, E)
            or env_ids.shape != (E,) or not torch.equal(env_ids, torch.arange(E))):
        raise ValueError('Raw data shapes or environment identifiers are inconsistent')
    if any(not torch.isfinite(x).all() for x in (states, final, actions, command, feedback, xi, phase, times)):
        raise ValueError('Raw trajectory contains nonfinite data')
    if term.bool().any() or timeout.bool().any():
        raise ValueError('Raw trajectory contains a terminal or reset boundary')
    if not torch.allclose(times.double(), torch.arange(T, dtype=torch.float64) * raw['dt'], atol=1e-7, rtol=0):
        raise ValueError('Raw timestamps are not contiguous')
    if not torch.all(phase[1:] >= phase[:-1]) or not torch.all(phase <= times + 1e-7):
        raise ValueError('Invalid phase timestamps')
    if not torch.isfinite(states[..., 3:7]).all() or ((states[..., 3:7].norm(dim=-1)-1).abs() > 1e-3).any() or ((final[..., 3:7].norm(dim=-1)-1).abs() > 1e-3).any():
        raise ValueError('Invalid state quaternion')
    all_states = torch.cat((states, final[None]), dim=0)
    source_paths = [Path(inspect.getsourcefile(obj)).resolve() for obj in (WorldModel, StateSpecCfg, measured_velocity, heading_yaw)]
    report = dict(status='auditing', checkpoint=str(a.checkpoint.resolve()), checkpoint_sha256=sha(a.checkpoint),
        raw=str(a.raw.resolve()), raw_sha256=sha(a.raw), entry_sha256=sha(__file__),
        source_code_sha256={str(path): sha(path) for path in source_paths}, source_commit=os.environ.get('SM_SOURCE_COMMIT', 'unspecified'),
        raw_source_sha256=source_sha, checkpoint_initial_source_sha256=meta.get('source_sha256'),
        raw_reference_sha256=ref_sha, checkpoint_reference_sha256=expected_ref,
        raw_pose_sha256=pose_sha, checkpoint_pose_sha256=expected_pose,
        source_matches_checkpoint_initial=source_sha == meta.get('source_sha256') if meta.get('source_sha256') else None,
        checkpoint_iteration=meta.get('iteration', ck.get('iteration')), checkpoint_training_seed=meta.get('config', {}).get('seed'),
        raw_seed=raw_seed, raw_report_sha256=raw_report_sha, dt=raw['dt'], num_envs=E, transitions=T,
        wm_error_thresholds={'velocity_8':[.10,.10,.15], 'velocity_32':[.20,.20,.30], 'xy_heading_32':[.03,.08]},
        window_selection='at most 32 evenly spaced (time, environment) starts per phase, 32 executed actions within that phase',
        trajectory='new recorded physical trajectory; statistical independence from training is unverified',
        scope='new raw trajectory diagnostic only; statistical independence unverified; no physical policy acceptance', phases={})

    def quantiles(t):
        return {'p95': float(torch.quantile(t, .95)), 'max': float(t.max())}

    def statistics(errors):
        out = {}
        for horizon, (dv, xy, dyaw) in errors.items():
            speed_norm = dv.norm(dim=-1)
            out[str(horizon)] = dict(velocity_rmse=dv.square().mean(0).sqrt().tolist(),
                xy_rmse=float(xy.square().mean().sqrt()), heading_rmse=float(dyaw.square().mean().sqrt()),
                position_error=quantiles(xy), speed_error_norm=quantiles(speed_norm),
                per_sample_velocity_error=dv.tolist(),
                velocity_axis_abs_p95=torch.quantile(dv.abs(), .95, dim=0).tolist(),
                velocity_axis_abs_max=dv.abs().amax(0).tolist())
        return out

    def gate(metrics):
        return (all(x < y for x,y in zip(metrics['8']['velocity_rmse'], (.10,.10,.15))) and
                all(x < y for x,y in zip(metrics['32']['velocity_rmse'], (.20,.20,.30))) and
                metrics['32']['xy_rmse'] < .03 and metrics['32']['heading_rmse'] < .08)

    aggregate = {h: [[], [], []] for h in (1, 8, 32)}
    with torch.no_grad():
        unique_phase = torch.unique_consecutive(phase)
        for phase_value in unique_phase.tolist():
            starts = [i for i in range(T-31) if float(phase[i]) == float(phase_value) and float(phase[i+31]) == float(phase_value)]
            combinations = [(t, e) for t in starts for e in range(E)]
            if not combinations:
                report['phases'][str(phase_value)] = dict(window_count=0, model_gate_passed=None, reason='no 32-step window')
                continue
            chosen = np.linspace(0, len(combinations)-1, min(32, len(combinations)), dtype=int)
            pairs = [combinations[int(i)] for i in chosen]
            ts = torch.tensor([t for t,e in pairs], dtype=torch.long)
            es = torch.tensor([e for t,e in pairs], dtype=torch.long)
            pred = all_states[ts, es].clone(); context = xi[es]
            errors = {}
            for j in range(32):
                pred = wm.step(pred, actions[ts+j, es], context)
                if j+1 in (1,8,32):
                    actual = all_states[ts+j+1, es]
                    dv = measured_velocity(pred)-measured_velocity(actual)
                    xy = (pred[:,:2]-actual[:,:2]).norm(dim=-1)
                    dyaw = heading_yaw(pred[:,3:7])-heading_yaw(actual[:,3:7])
                    dyaw = torch.atan2(torch.sin(dyaw),torch.cos(dyaw))
                    if not all(torch.isfinite(v).all() for v in (dv, xy, dyaw)):
                        raise FloatingPointError('Nonfinite world-model prediction')
                    errors[j+1] = (dv, xy, dyaw)
                    for bucket,v in zip(aggregate[j+1],errors[j+1]): bucket.append(v)
            metrics = statistics(errors)
            report['phases'][str(phase_value)] = dict(window_count=len(pairs), available_windows=len(combinations),
                selected=[{'frame':t,'time_s':float(times[t]),'env_id':e} for t,e in pairs], metrics=metrics,
                model_gate_passed=gate(metrics))
    if aggregate[1][0]:
        combined = {h: tuple(torch.cat(v) for v in groups) for h,groups in aggregate.items()}
        report['global'] = dict(window_count=len(combined[1][0]), metrics=statistics(combined))
        report['global']['model_gate_passed'] = gate(report['global']['metrics'])
    else:
        report['global'] = dict(window_count=0, model_gate_passed=None)
    report['status'] = 'completed'
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as f:
        json.dump(report,f,indent=2,allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
