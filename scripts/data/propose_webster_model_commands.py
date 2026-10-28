"""Propose bounded preparation changes with a frozen short-horizon world model.

Uses recorded actions or optional frozen-policy feedback and bias-corrected,
four-step predictions. No actual state, policy, physics or checkpoint is edited.
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
from supermimic.data.motion_convert import load_g1_model
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
    parser.add_argument('--policy-feedback', action='store_true', help='Recompute frozen policy actions and observation histories through predicted states; requires captured native observations')
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
    asset = load_g1_model()
    assert (asset.nq, asset.nv) == (model.nq, model.nv)
    assert [model.joint(i).name.rsplit('/', 1)[-1] for i in range(model.njnt)] == [asset.joint(i).name for i in range(asset.njnt)]
    fk = G1Kinematics(asset, 'cpu'); nq = model.nq
    # The scene has a separate fixed terrain body before the robot's free root.
    native_body_ids = np.array([model.body('robot/'+n).id for n in fk.body_names])
    native_to_fk = {int(b): i for i, b in enumerate(native_body_ids)}
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
    feedback_validation = None
    if args.policy_feedback:
        unsupported = [k for k in ('landing_ckpt', 'bounded_target_joints', 'joint_stop_margin', 'physics_command_projection')
                       if configuration['arguments'].get(k)]
        if unsupported:
            raise ValueError(f'Feedback prediction does not implement these captured controllers: {unsupported}')
        from supermimic.envs.mdp.motion_library import MotionLibrary
        from supermimic.eval.benchmark import load_policy_from_checkpoint
        from supermimic.world_model.heads import OBS_TERMS, analytic_observation, anchor_relative_obs, flatten_history_term_major
        captured = np.load(args.trace/'baseline-observations.npz')
        policy, policy_cfg = load_policy_from_checkpoint(args.ckpt, 'cpu')
        policy.requires_grad_(False)
        q0 = torch.tensor([PROFILE['default_joint_pos'][n] for n in PROFILE['joint_names']], dtype=states.dtype)
        tracked = tuple(PROFILE['tracked_bodies']); anchor = tracked.index(PROFILE['anchor_body'])
        source = Path(configuration['arguments']['source'])
        if digest(source) != configuration['source_sha256']:
            raise ValueError('Reference source changed after calibration capture')
        lib = MotionLibrary([source], tracked, q0, device='cpu')
        frames = torch.tensor(captured['reference_frame'], dtype=torch.long)
        native_goals = torch.tensor(captured['goal']); native_obs = torch.tensor(captured['policy'])
        history_length = native_obs.shape[-1]//sum(d for _, d in OBS_TERMS)
        if len(native_obs) != len(states): raise ValueError('Observation/state capture timelines differ')
        histories = []; offset = 0
        for _, size in OBS_TERMS:
            histories.append(native_obs[:, offset:offset+history_length*size].reshape(len(states), history_length, size))
            offset += history_length*size
        histories = torch.cat(histories, -1).permute(1, 0, 2)
        anchor_index = fk.body_index(PROFILE['anchor_body'])
        future_frames = policy_cfg['env'].get('future_frames', 10)
        def feedback_goal(state, indices):
            frame = frames[indices]
            goal = lib.goal_window(frame, future_frames)
            if policy_cfg['env'].get('anchor_obs', False):
                out = fk(state[:, :nq]); reference = lib.bodies(frame)
                rel = anchor_relative_obs(out['pos'][:, anchor_index], out['quat'][:, anchor_index],
                    reference.pos_w[:, anchor], reference.quat_w[:, anchor])
                goal = torch.cat([goal, rel], -1)
            return goal
        with torch.no_grad():
            before = torch.cat([torch.tensor(np.r_[captured['initial_qpos'], captured['initial_qvel']], dtype=states.dtype)[None], states[:-1]])
            reproduced_goals = feedback_goal(before, torch.arange(len(states)))
            goal_error = float((reproduced_goals-native_goals).abs().max())
            # Independently verify the full rolling input history, not just its newest sample.
            history = histories[:, :1].clone(); observation_error = 0.
            for t in range(1, len(states)):
                current = analytic_observation(states[t-1:t], actions[t-1:t], torch.zeros(1, XI_DIM), q0)
                history = torch.cat([history[1:], current[None]], 0)
                observation_error = max(observation_error, float((flatten_history_term_major(history)[0]-native_obs[t]).abs().max()))
            if max(goal_error, observation_error) >= 3e-5:
                raise ValueError(f'Native feedback reconstruction failed: goal={goal_error}, history={observation_error}')
            native_base_actions = policy(native_obs, native_goals)
            residuals = actions-native_base_actions
        feedback_validation = {'maximum_goal_error': goal_error, 'maximum_history_error': observation_error,
            'observations_sha256': digest(args.trace/'baseline-observations.npz'), 'reference_frames_from_native_capture': True}
    foot_geoms = [g for g in range(model.ngeom) if 'foot_' in model.geom(g).name]
    assert foot_geoms and np.allclose(model.geom_friction[foot_geoms, 0], 1)
    xi = torch.zeros((len(starts), XI_DIM)); xi[:, 0] = 1
    as_tensor = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)
    mass = as_tensor(model.body_mass[native_body_ids]); ipos = as_tensor(model.body_ipos[native_body_ids])
    points, bodies, kinds = [], [], []
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        if b not in native_to_fk or not (model.geom_contype[g] or model.geom_conaffinity[g]): continue
        mesh = int(model.geom_dataid[g]); assert mesh >= 0
        vertices = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh]+model.mesh_vertnum[mesh]]
        vertices = vertices[ConvexHull(vertices).vertices]
        rotation = np.zeros(9); mujoco.mju_quat2Mat(rotation, model.geom_quat[g])
        points.extend(vertices@rotation.reshape(3, 3).T+model.geom_pos[g])
        bodies.extend([native_to_fk[b]]*len(vertices)); kinds.extend([g in foot_geoms]*len(vertices))
    points = as_tensor(points); bodies = torch.tensor(bodies); feet = torch.tensor(kinds)
    def physical_features(s):
        out = fk(s[:, :nq], s[:, nq:]); offset = quat_apply(out['quat'], ipos.expand_as(out['pos']))
        com_z = ((out['pos']+offset)[:, :, 2]*mass).sum(-1)/mass.sum()
        vel = out['lin_vel']+torch.cross(out['ang_vel'], offset, dim=-1)
        vz = (vel[:, :, 2]*mass).sum(-1)/mass.sum()
        height = (out['pos'][:, bodies]+quat_apply(out['quat'][:, bodies], points.expand(len(s), -1, -1)))[:, :, 2]
        return torch.stack([com_z, vz, height[:, feet].amin(-1), height[:, ~feet].amin(-1)], -1)
    with torch.no_grad():
        check = physical_features(states[starts]); check_pose = fk(states[starts, :nq])['pos']
        native_data = mujoco.MjData(model); pose_error = 0.; com_error = 0.
        for i, state in enumerate(states[starts].numpy()):
            native_data.qpos[:] = state[:nq]; native_data.qvel[:] = state[nq:]
            mujoco.mj_forward(model, native_data); mujoco.mj_subtreeVel(model, native_data)
            root = int(model.jnt_bodyid[0])
            pose_error = max(pose_error, float(abs(check_pose[i].numpy()-native_data.xpos[native_body_ids]).max()))
            com_error = max(com_error, abs(float(check[i, 0])-native_data.subtree_com[root, 2]),
                            abs(float(check[i, 1])-native_data.subtree_linvel[root, 2]))
        assert max(pose_error, com_error) < 3e-5, 'Asset FK differs from the recorded native scene; proposal rejected'
        native_features = torch.stack([physical_features(states[starts+h]) for h in range(1, 5)])
    # Native servo conversion; bounds and torque clipping remain in actual simulation.
    scale = PROFILE['action_scale']
    conversion = torch.tensor([PROFILE['effort_limit'][n]/PROFILE['stiffness'][n]/(scale[n] if isinstance(scale, dict) else scale) for n in joints])
    basis = preparation_command_basis((starts[None, :]+torch.arange(1, 5)[:, None])*.02, float(p[0]))
    def predict(change_parameters):
        s = states[starts]; predicted = []
        history = histories[:, starts+1].clone() if args.policy_feedback else None
        for h in range(4):
            indices = starts+h+1
            change = (basis[h, :, 2:4]@change_parameters.reshape(2, 2))*conversion
            action = (policy(flatten_history_term_major(history), feedback_goal(s, indices))+residuals[indices]
                      if args.policy_feedback else actions[indices].clone())
            action = action.clone(); action[:, joint_indices] += change
            s = wm.step(s, action, xi); predicted.append(s)
            if args.policy_feedback:
                current = analytic_observation(s, action, xi, q0)
                history = torch.cat([history[1:], current[None]], 0)
        return torch.stack(predicted)
    with torch.no_grad():
        baseline_states = predict(torch.zeros(4))
        baseline_features = torch.stack([physical_features(s) for s in baseline_states])
    lo = as_tensor(model.jnt_range[1:, 0]); hi = as_tensor(model.jnt_range[1:, 1])
    lower = torch.maximum(torch.full((4,), -args.trust), as_tensor(-1.5-p[columns]))
    upper = torch.minimum(torch.full((4,), args.trust), as_tensor(1.5-p[columns]))
    delta = torch.zeros(4, requires_grad=True); opt = torch.optim.Adam([delta], lr=.01)
    best = None; history = []
    for iteration in range(args.steps):
        predicted = predict(delta)
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
        'scope': ('Bias-corrected four-step frozen-policy feedback proposals; native replay remains mandatory.' if args.policy_feedback
                  else 'Bias-corrected four-step, recorded-action model proposals; native replay remains mandatory.'),
        'policy_feedback': args.policy_feedback, 'feedback_validation': feedback_validation,
        'native_fk_validation': {'maximum_link_position_error_m': pose_error, 'maximum_COM_height_or_velocity_error': com_error},
        'columns': columns, 'joints': joints, 'scales': scales, 'best_delta': best[1].tolist(), 'history': history,
        'elapsed_s': time.monotonic()-started, 'input_sha256': {'ckpt': digest(args.ckpt), 'solution': digest(args.solution),
        'trajectory': digest(args.trace/'baseline-trajectory.npz'), 'model': digest(args.trace/'native-model.mjb')}}
    (args.out/'result.json').write_text(json.dumps(result, indent=2)); print(json.dumps({k:v for k,v in result.items() if k!='history'}), flush=True)


if __name__ == '__main__': main()
