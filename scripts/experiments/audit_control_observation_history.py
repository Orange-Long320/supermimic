"""Compare recorded policy observations with native analytic history reconstruction on CPU."""
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
    for key in ('raw', 'retention', 'report', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    expected = Path('/home/2025-Long.xh/HHD/venvs/supermimic').resolve()
    if Path(sys.prefix).resolve() != expected:
        p.error('Use the dedicated server interpreter')
    if a.out.exists() or any(not x.is_file() for x in (a.raw, a.retention, a.report)):
        p.error('Require existing inputs and a new output path')
    import torch
    from mjlab.envs import mdp
    from supermimic.envs.mdp.commands import MultiClipMotionCommand
    from supermimic.envs.tracking_env_cfg import make_env_cfg
    from supermimic.robots.dimensions import NJ, NQ, OBS_DIM, OBS_TERMS, STATE_DIM, XI_DIM
    from supermimic.world_model.heads import analytic_observation, flatten_history_term_major

    torch.set_num_threads(2)
    raw = torch.load(a.raw, map_location='cpu', weights_only=False)
    retention = torch.load(a.retention, map_location='cpu', weights_only=False)
    report = json.loads(a.report.read_text())
    if report.get('raw_sha256') != sha(a.raw) or report.get('retention_sha256') != sha(a.retention):
        raise ValueError('Collection report does not bind both input files')
    for key in ('source_sha256', 'reference_sha256', 'standing_pose_sha256'):
        if not raw.get(key) or raw[key] != retention.get(key) or raw[key] != report.get(key):
            raise ValueError(f'Input provenance mismatch: {key}')
    if report.get('status') != 'collected':
        raise ValueError('Collection did not complete without failure')
    required_raw = ('state_before', 'action', 'command', 'feedback', 'time_s', 'terminated', 'timeout', 'xi', 'env_id', 'dt')
    required_ret = ('obs', 'command', 'feedback', 'source_action', 'age', 'env_id')
    if any(raw.get(k) is None for k in required_raw) or any(retention.get(k) is None for k in required_ret):
        raise ValueError('Missing aligned input tensors')
    states, actions, xi, times = (raw[k] for k in ('state_before', 'action', 'xi', 'time_s'))
    observations = retention['obs']
    H = 10
    if states.ndim != 3:
        raise ValueError('Expected states with shape [time, environment, state]')
    T, E, S = states.shape
    n = len(observations)
    if (S != STATE_DIM or actions.shape != (T, E, NJ) or xi.shape != (E, XI_DIM) or times.shape != (T,)
            or raw['command'].shape != (T,E,3) or raw['feedback'].shape != (T,E,6)
            or observations.shape != (n, H * OBS_DIM) or retention['command'].shape != (n,3)
            or retention['feedback'].shape != (n,6) or retention['source_action'].shape != (n,NJ)
            or retention['age'].shape != (n,) or retention['env_id'].shape != (n,)
            or raw['terminated'].shape != (T,E) or raw['timeout'].shape != (T,E)
            or not torch.equal(raw['env_id'], torch.arange(E))):
        raise ValueError('Invalid raw or sampled dimensions')
    if any(not torch.isfinite(x).all() for x in (states, actions, xi, times, observations, retention['age'],
                                                  raw['command'], raw['feedback'], retention['command'], retention['feedback'])):
        raise ValueError('Nonfinite inputs')
    dt = float(raw['dt'])
    if dt <= 0 or not torch.allclose(times.double(), torch.arange(T, dtype=torch.float64) * dt, atol=1e-7, rtol=0):
        raise ValueError('Noncontiguous time axis')
    if raw['terminated'].bool().any() or raw['timeout'].bool().any():
        raise ValueError('Trace contains a termination or reset boundary')
    indices = torch.round(retention['age'].double() / dt).long()
    envs = retention['env_id'].long()
    valid = (indices >= H+2) & (indices < T-1) & (envs >= 0) & (envs < E)
    indices, envs, observed = indices[valid], envs[valid], observations[valid]
    if len(indices) == 0:
        raise ValueError('No samples have sufficient non-reset history')
    if not torch.allclose(retention['age'][valid].double(), times[indices].double(), atol=1e-5, rtol=0):
        raise ValueError('Sample ages do not match raw frame times')
    for key, field in (('command','command'), ('feedback','feedback'), ('source_action','action')):
        if not torch.equal(retention[key][valid], raw[field][indices,envs]):
            raise ValueError(f'Sampled {key} differs from corresponding raw row')
    # Native term-major layout: q history first, each frame NJ entries.
    q_observed_latest = observed[:, (H-1)*NJ:H*NJ]
    q0_samples = states[indices,envs,7:NQ] + xi[envs,1:NJ+1] - q_observed_latest
    q0 = q0_samples.median(dim=0).values
    q0_diff = q0_samples-q0

    def describe(x):
        return dict(max_abs=float(x.abs().max()), rmse=float(x.square().mean().sqrt()))

    def reconstruct(state_offset=0, action_offset=0, reverse=False, time_major=False):
        offsets = torch.arange(-(H-1), 1)
        state_times = indices[None,:] + offsets[:,None] + state_offset
        action_times = state_times - 1 + action_offset
        ee = envs[None,:].expand(H,-1)
        hist = analytic_observation(states[state_times,ee], actions[action_times,ee], xi[envs][None,:,:], q0)
        if reverse:
            hist = hist.flip(0)
        return hist.permute(1,0,2).reshape(len(indices), H*OBS_DIM) if time_major else flatten_history_term_major(hist)

    native = reconstruct()
    diff = native-observed
    terms = {}; pos = 0
    for name, width in OBS_TERMS:
        end = pos + H*width
        terms[name] = describe(diff[:, pos:end])
        pos = end
    if pos != H*OBS_DIM:
        raise ValueError('Native term widths do not match observation dimension')
    alternatives = {}
    if float(diff.abs().max()) > 1e-5:
        for name, kwargs in (('reverse_history', {'reverse':True}), ('state_t_minus_1', {'state_offset':-1}),
                             ('state_t_plus_1', {'state_offset':1}), ('previous_action_minus_1', {'action_offset':-1}),
                             ('previous_action_plus_1', {'action_offset':1}), ('time_major', {'time_major':True})):
            alternatives[name] = describe(reconstruct(**kwargs)-observed)
    source_files = [Path(inspect.getsourcefile(obj)).resolve() for obj in
                    (analytic_observation, flatten_history_term_major, make_env_cfg, MultiClipMotionCommand,
                     mdp.joint_pos_rel, mdp.joint_vel_rel, mdp.last_action)]
    result = dict(status='completed', raw_sha256=sha(a.raw), retention_sha256=sha(a.retention),
        collection_report_sha256=sha(a.report), source_sha256=raw['source_sha256'],
        reference_sha256=raw['reference_sha256'], standing_pose_sha256=raw['standing_pose_sha256'],
        entry_sha256=sha(__file__), source_code_sha256={str(x):sha(x) for x in source_files},
        source_commit=os.environ.get('SM_SOURCE_COMMIT','unspecified'), history=H, dt=dt,
        num_samples=len(indices), sample_frame_min=int(indices.min()), sample_frame_max=int(indices.max()),
        expected_term_order=[{'name':name,'width_per_frame':width} for name,width in OBS_TERMS],
        inferred_default_joint_pos=q0.tolist(), default_position_consistency=describe(q0_diff),
        default_position_per_joint_max_abs=q0_diff.abs().amax(0).tolist(),
        default_position_per_joint_std=q0_samples.std(0, unbiased=False).tolist(),
        default_joint_pos_basis='median of physical q + measured xi encoder bias - latest recorded joint_pos observation; library is initialized from robot.data.default_joint_pos',
        native_total=describe(diff), native_terms=terms, alternative_diagnostics=alternatives,
        scope='offline observation alignment diagnostic; no automatic training or policy changes')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
