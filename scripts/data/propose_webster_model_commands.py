"""Propose bounded preparation changes with a frozen short-horizon world model.

Uses recorded actions and bias-corrected, four-step predictions. This is an
open-loop proposal only: no actual state, policy, physics or checkpoint is edited.
Every exported parameter row still requires independent native simulation.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial import ConvexHull

from supermimic.data.command_basis import preparation_command_basis
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.kinematics.rotations import quat_apply
from supermimic.robots.dimensions import PROFILE, XI_DIM
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--solution', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--trust', type=float, default=.1, help='Maximum normalized effort coefficient change')
    args = parser.parse_args()
    if args.ckpt.name == 'ckpt_latest.pt' or args.steps <= 0 or not 0 < args.trust <= .2:
        parser.error('Use an immutable checkpoint, positive steps and trust in (0,.2]')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1); started = time.monotonic()
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    configuration = json.loads((args.trace/'configuration.json').read_text())
    if configuration['checkpoint_sha256'] != digest(args.ckpt):
        raise ValueError('Prediction model must match the controller generating this calibration trace')
    if configuration['warmstart_sha256'] != digest(args.solution):
        raise ValueError('Calibration trace must use the supplied exact parameter solution')
    p = np.load(args.solution)['parameters'].copy()
    groups = configuration['groups']; count = len(groups)
    if p.shape != (1+4*count,):
        raise ValueError('Expected preparation-only parameters with an initial command knot')
    labels = [g[0][0] for g in groups]
    joints = ['left_hip_pitch_joint', 'left_knee_joint']
    columns = [1+(k-1)*count+labels.index(n) for k in [2, 3] for n in joints]
    model = mujoco.MjModel.from_binary_path(str(args.trace/'native-model.mjb'))
    fk = G1Kinematics(model, 'cpu'); nq = model.nq
    names = PROFILE['joint_names']; joint_indices = [names.index(n) for n in joints]
    data = np.load(args.trace/'baseline-trajectory.npz')
    assert float(data['fps'][0]) == 50 and data['qpos'].ndim == 2
    states = torch.tensor(np.concatenate([data['qpos'], data['qvel']], axis=-1), dtype=torch.float32)
    actions = torch.tensor(data['actions'], dtype=torch.float32)
    # Saved states follow action[t]; transition from state[t] uses action[t+1].
    starts = torch.tensor([17, 21, 25, 29])  # .36/.44/.52/.60s, four-step windows
    if starts[-1]+4 >= len(states): raise ValueError('Calibration trace is too short')
    wm_data = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = dict(wm_data['config']['world_model']); cfg['state_spec'] = StateSpecCfg(**cfg['state_spec'])
    wm = WorldModel(WorldModelCfg(**cfg), None, 'cpu')
    wm.load_state_dict(wm_data['wm']); wm.eval(); wm.requires_grad_(False)
    foot_geoms = [g for g in range(model.ngeom) if 'foot_' in model.geom(g).name]
    assert foot_geoms and np.allclose(model.geom_friction[foot_geoms, 0], 1)
    xi = torch.zeros((len(starts), XI_DIM)); xi[:, 0] = 1
    as_tensor = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)
    mass = as_tensor(model.body_mass[1:]); ipos = as_tensor(model.body_ipos[1:])
    points, bodies, kinds = [], [], []
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        if not b or not (model.geom_contype[g] or model.geom_conaffinity[g]): continue
        mesh = int(model.geom_dataid[g]); assert mesh >= 0
        vertices = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh]+model.mesh_vertnum[mesh]]
        vertices = vertices[ConvexHull(vertices).vertices]
        rotation = np.zeros(9); mujoco.mju_quat2Mat(rotation, model.geom_quat[g])
        points.extend(vertices@rotation.reshape(3, 3).T+model.geom_pos[g])
        bodies.extend([b-1]*len(vertices)); kinds.extend([g in foot_geoms]*len(vertices))
    points = as_tensor(points); bodies = torch.tensor(bodies); feet = torch.tensor(kinds)
    def physical_features(s):
        out = fk(s[:, :nq], s[:, nq:]); offset = quat_apply(out['quat'], ipos.expand_as(out['pos']))
        com_z = ((out['pos']+offset)[:, :, 2]*mass).sum(-1)/mass.sum()
        vel = out['lin_vel']+torch.cross(out['ang_vel'], offset, dim=-1)
        vz = (vel[:, :, 2]*mass).sum(-1)/mass.sum()
        height = (out['pos'][:, bodies]+quat_apply(out['quat'][:, bodies], points.expand(len(s), -1, -1)))[:, :, 2]
        return torch.stack([com_z, vz, height[:, feet].amin(-1), height[:, ~feet].amin(-1)], -1)
    with torch.no_grad():
        native_features = torch.stack([physical_features(states[starts+h]) for h in range(1, 5)])
        s = states[starts]; baseline_states = []
        for h in range(1, 5):
            s = wm.step(s, actions[starts+h], xi); baseline_states.append(s)
        baseline_states = torch.stack(baseline_states)
        baseline_features = torch.stack([physical_features(s) for s in baseline_states])
    # Native servo conversion; bounds and torque clipping remain in actual simulation.
    scale = PROFILE['action_scale']
    conversion = torch.tensor([PROFILE['effort_limit'][n]/PROFILE['stiffness'][n]/(scale[n] if isinstance(scale, dict) else scale) for n in joints])
    basis = preparation_command_basis((starts[None, :]+torch.arange(1, 5)[:, None])*.02, float(p[0]))
    lo = as_tensor(model.jnt_range[1:, 0]); hi = as_tensor(model.jnt_range[1:, 1])
    lower = torch.maximum(torch.full((4,), -args.trust), as_tensor(-1.5-p[columns]))
    upper = torch.minimum(torch.full((4,), args.trust), as_tensor(1.5-p[columns]))
    delta = torch.zeros(4, requires_grad=True); opt = torch.optim.Adam([delta], lr=.01)
    best = None; history = []
    for iteration in range(args.steps):
        s = states[starts]; predicted = []
        for h in range(4):
            change = (basis[h, :, 2:4]@delta.reshape(2, 2))*conversion
            action = actions[starts+h+1].clone(); action[:, joint_indices] += change
            s = wm.step(s, action, xi); predicted.append(s)
        predicted = torch.stack(predicted)
        features = torch.stack([physical_features(s) for s in predicted])
        corrected = native_features+(features-baseline_features)
        # This vertical-energy proxy also scores supported states; it is not a flight certificate.
        proxy = corrected[..., 0]+corrected[..., 1].clamp(min=0).square()/(2*abs(float(model.opt.gravity[2])))
        original_proxy = native_features[..., 0]+native_features[..., 1].clamp(min=0).square()/(2*abs(float(model.opt.gravity[2])))
        q = states[starts[None, :]+torch.arange(1, 5)[:, None], 7:nq]+predicted[..., 7:nq]-baseline_states[..., 7:nq]
        joint_penalty = ((lo+.025-q).clamp(min=0).square()+(q-hi+.025).clamp(min=0).square()).sum(-1).mean()
        floor_penalty = ((-.004-corrected[..., 2]).clamp(min=0).square()+(.005-corrected[..., 3]).clamp(min=0).square()).mean()
        gain = (proxy-original_proxy).mean()
        loss = -gain+100*joint_penalty+200*floor_penalty+.001*(delta/args.trust).square().mean()
        if not torch.isfinite(loss): raise ValueError('Nonfinite prediction; proposal rejected')
        value = float(loss.detach())
        if best is None or value < best[0]: best = (value, delta.detach().clone())
        history.append({'iteration': iteration, 'loss': value, 'predicted_proxy_gain_m': float(gain.detach()), 'delta': delta.detach().tolist()})
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): delta.clamp_(min=lower, max=upper)
    scales = [0., .25, .5, 1., -1.]; candidates = []
    for factor in scales:
        row = p.copy(); row[columns] = np.clip(row[columns]+factor*best[1].numpy(), -1.5, 1.5); candidates.append(row)
    np.savez(args.out/'parameter-grid.npz', parameters=np.asarray(candidates))
    result = {'approved_for_training': False, 'actual_state_modified': False, 'policy_trained': False,
        'scope': 'Bias-corrected four-step, recorded-action model proposals; native replay remains mandatory.',
        'columns': columns, 'joints': joints, 'scales': scales, 'best_delta': best[1].tolist(), 'history': history,
        'elapsed_s': time.monotonic()-started, 'input_sha256': {'ckpt': digest(args.ckpt), 'solution': digest(args.solution),
        'trajectory': digest(args.trace/'baseline-trajectory.npz'), 'model': digest(args.trace/'native-model.mjb')}}
    (args.out/'result.json').write_text(json.dumps(result, indent=2)); print(json.dumps({k:v for k,v in result.items() if k!='history'}), flush=True)


if __name__ == '__main__': main()
