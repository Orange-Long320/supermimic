"""CPU forward shooting for a physically simulated single-leg takeoff.

This is a reference-design experiment, not a learned policy. The isolated CPU
plant uses the project mass, collision meshes, PD gains, armatures and effort
limits. Agreement with mjlab and a complete flip/landing remain separate gates.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import least_squares


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--force-report", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=60)
    p.add_argument("--population", type=int, default=24)
    p.add_argument("--max-seconds", type=float, default=900)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmstart", type=Path)
    p.add_argument("--soft-limit-weight", type=float, default=100.)
    p.add_argument("--follow-through-seconds", type=float, default=0.)
    p.add_argument("--flight-seconds", type=float, default=0., help="Replay the aerial joint targets after physical contact loss")
    p.add_argument("--extra-balance-controls", action="store_true")
    p.add_argument("--joint-margin", type=float, default=0., help="Reserve an interior margin in the initial pose and pose prior; never clips simulated states")
    p.add_argument("--control-units", choices=("angle", "effort"), default="angle")
    p.add_argument("--inverse-torque-scale", type=float, default=1.)
    p.add_argument("--initial-pose", choices=("source", "neutral"), default="source")
    p.add_argument("--stance-prior", choices=("video", "balanced"), default="video")
    p.add_argument("--flatten-neutral-feet", action="store_true")
    p.add_argument("--feedback-stiffness", type=float, default=0., help="Optional outer controller, expressed through ordinary position commands; does not change the actuator model")
    p.add_argument("--feedback-damping", type=float, default=0.)
    args = p.parse_args()
    from supermimic.robots.dimensions import PROFILE, robot_xml_path
    args.out.mkdir(parents=True, exist_ok=False)
    spec = mujoco.MjSpec.from_file(str(robot_xml_path()))
    spec.worldbody.add_geom(name="design_ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[10, 10, .1], contype=1, conaffinity=1,
                            friction=[1, .005, .0001])
    names = PROFILE["joint_names"]
    kp = np.array([PROFILE["stiffness"][n] for n in names])
    kd = np.array([PROFILE["damping"][n] for n in names])
    effort = np.array([PROFILE["effort_limit"][n] for n in names])
    velocity_limits = np.array([PROFILE["velocity_limit"][n] for n in names])
    for k, name in enumerate(names):
        spec.add_actuator(name=f"design_{name}", target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                          gaintype=mujoco.mjtGain.mjGAIN_FIXED, gainprm=[kp[k]] + [0.] * 9,
                          biastype=mujoco.mjtBias.mjBIAS_AFFINE, biasprm=[0, -kp[k], -kd[k]] + [0.] * 7,
                          ctrllimited=False, inheritrange=0.,
                          forcelimited=True, forcerange=[-effort[k], effort[k]])
    spec.option.timestep = .005
    spec.option.iterations = 10
    spec.option.ls_iterations = 20
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model = spec.compile()
    assert model.nv == len(names) + 6 and model.nu == len(names)
    for k, name in enumerate(names):
        model.dof_armature[model.jnt_dofadr[model.joint(name).id]] = PROFILE["armature"][name]
    data = mujoco.MjData(model)
    centre = model.jnt_range[1:].mean(axis=1)
    half = .45 * np.diff(model.jnt_range[1:], axis=1).ravel()
    lower, upper = centre - half, centre + half
    assert 0 <= args.joint_margin < half.min()
    root = int(model.jnt_bodyid[0])
    z = np.load(args.source)
    r = z["body_names"].tolist().index(model.body(root).name)
    poses = np.c_[z["body_pos_w"][:, r], z["body_quat_w"][:, r], z["joint_pos"]].astype(float)
    initial = poses[0].copy()
    if args.initial_pose == "neutral":
        initial[:7] = [0,0,PROFILE["standing_height"],1,0,0,0]
        initial[7:] = [PROFILE["default_joint_pos"][name] for name in names]
    initial[7:] = np.clip(initial[7:], lower + args.joint_margin, upper - args.joint_margin)
    feet = []
    foot_ids = set()
    for g in range(model.ngeom):
        if model.geom(g).name.startswith("foot_"):
            mesh = model.geom_dataid[g]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            vertices = model.mesh_vert[start:start + count].astype(float)
            vertices = vertices[ConvexHull(vertices).vertices]
            feet.append((g, vertices))
            foot_ids.add(g)
            model.geom_priority[g] = 1
    if args.flatten_neutral_feet:
        assert args.initial_pose == "neutral"
        data.qpos[:] = initial
        mujoco.mj_forward(model,data)
        sole_normals = []
        for g,vertices in feet:
            normals = ConvexHull(vertices).equations[:,:3]
            world_normals = normals @ data.geom_xmat[g].reshape(3,3).T
            sole_normals.append(normals[np.argmin(world_normals[:,2])])
        columns = np.array([model.jnt_qposadr[model.joint(n).id] for n in (
            "left_ankle_pitch_joint","left_ankle_roll_joint","right_ankle_pitch_joint","right_ankle_roll_joint")])
        def flat_error(angles):
            data.qpos[:] = initial
            data.qpos[columns] = angles
            mujoco.mj_kinematics(model,data)
            return np.concatenate([(data.geom_xmat[g].reshape(3,3) @ normal)[:2] for (g,_),normal in zip(feet,sole_normals)])
        fit = least_squares(flat_error,initial[columns],bounds=(lower[columns-7],upper[columns-7]),
                            ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=30)
        assert np.linalg.norm(fit.fun)<1e-5,"Neutral feet could not be flattened within the project soft limits"
        initial[columns] = fit.x
    def foot_heights():
        return np.array([(v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])[:, 2].min()
                         for g, v in feet])
    data.qpos[:] = initial
    mujoco.mj_forward(model, data)
    initial[2] += .001 - foot_heights().min()
    force_rows = json.loads(args.force_report.read_text())["rows"]
    inverse_torque = np.zeros((len(poses), len(names)))
    for row in force_rows:
        inverse_torque[row["frame"]] = row["required_joint_torque_Nm"]
    inverse_torque = np.clip(inverse_torque, -effort, effort)
    source_fps = float(z["fps"].reshape(-1)[0])
    support_frames = np.arange(31)
    joint_prior = np.clip(z["joint_pos"], lower + args.joint_margin, upper - args.joint_margin)
    if args.initial_pose == "neutral":
        transition = np.minimum(np.arange(len(poses))/20,1.)
        transition = transition**2 * (3-2*transition)
        joint_prior = initial[None,7:] * (1-transition[:,None]) + joint_prior * transition[:,None]
    prior_velocity = np.gradient(joint_prior, 1/source_fps, axis=0)
    full_target = joint_prior + args.inverse_torque_scale * inverse_torque / kp + np.clip(prior_velocity, -velocity_limits, velocity_limits) * kd / kp
    reference_target = full_target[:31]
    gravity_torque = np.zeros(len(names))
    if args.stance_prior == "balanced":
        assert args.initial_pose == "neutral"
        from supermimic.data.contact_wrench import solve_contact_effort
        data.qpos[:] = initial
        data.qvel[:] = 0
        data.qacc[:] = 0
        mujoco.mj_inverse(model,data)
        required = data.qfrc_inverse.copy()
        rays = np.array([[a/np.sqrt(2),b/np.sqrt(2),1] for a,b in [(1,1),(1,-1),(-1,1),(-1,-1)]]).T
        columns = []
        jp,jr = np.zeros((3,model.nv)),np.zeros((3,model.nv))
        for g,vertices in feet:
            world = vertices @ data.geom_xmat[g].reshape(3,3).T + data.geom_xpos[g]
            for point in world[world[:,2] < world[:,2].min()+.003]:
                mujoco.mj_jac(model,data,jp,jr,point,int(model.geom_bodyid[g]))
                columns.append(jp.T @ rays)
        basis = np.concatenate(columns,axis=1)
        fit = solve_contact_effort(basis,required,effort)
        assert fit.success and fit.x[-1] < 1, "Neutral pose has no motor-bounded static contact equilibrium"
        gravity_torque = required[6:] - basis[6:] @ fit.x[:-1]
        reference_target = np.tile(initial[7:] + gravity_torque/kp,(31,1))
    forward_axis = data.xmat[root].reshape(3,3)[:,0].copy()
    forward_axis[2] = 0
    forward_axis /= np.linalg.norm(forward_axis)
    # Independent pitch controls plus coordinated arm swing and lateral balance.
    groups = [[(name, 1)] for name in (
        "left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint",
        "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint",
        "waist_pitch_joint", "waist_roll_joint")]
    groups += [[("left_shoulder_pitch_joint", 1), ("right_shoulder_pitch_joint", 1)],
               [("left_elbow_joint", 1), ("right_elbow_joint", 1)],
               [("left_hip_roll_joint", 1), ("right_hip_roll_joint", -1)]]
    if args.extra_balance_controls:
        groups += [[("left_ankle_roll_joint",1)], [("right_ankle_roll_joint",1)], [("right_elbow_joint",1)]]
        groups += [[("left_hip_roll_joint",1),("right_hip_roll_joint",1)]]
    control_map = np.zeros((len(groups), len(names)))
    for k, group in enumerate(groups):
        for name, sign in group:
            control_map[k, names.index(name)] = sign
    group_scale = np.array([min(effort[names.index(n)] / kp[names.index(n)] for n,_ in group) for group in groups])
    if args.control_units == "effort":
        control_map *= group_scale[:,None]
    mass = float(model.body_subtreemass[root])
    ground = model.geom("design_ground").id
    left_ids = {g for g in foot_ids if model.body(int(model.geom_bodyid[g])).name.startswith("left")}
    right_ids = foot_ids - left_ids
    log = args.out / "progress.jsonl"
    rng = np.random.default_rng(args.seed)
    dimension = 1 + 3 * len(groups)
    mean = np.zeros(dimension)
    mean[0] = .75
    if args.warmstart:
        warm_data = np.load(args.warmstart)
        warm = warm_data["parameters"].copy()
        previous_groups = (len(warm)-1)//3
        assert previous_groups <= len(groups) and 1+3*previous_groups == len(warm)
        previous_units = str(warm_data["control_units"]) if "control_units" in warm_data else "angle"
        if previous_units != args.control_units:
            warm[1:].reshape(3,previous_groups)[:] *= group_scale[:previous_groups] if previous_units == "effort" else 1/group_scale[:previous_groups]
        mean[0] = warm[0]
        mean[1:].reshape(3,len(groups))[:,:previous_groups] = warm[1:].reshape(3,previous_groups)
    std = np.full(dimension, .55)
    std[0] = .12
    bounds_lower = np.r_[.45, np.full(dimension - 1, -1.8)]
    bounds_upper = np.r_[1.2, np.full(dimension - 1, 1.8)]
    started = time.monotonic()
    best = None
    evaluations = 0

    def rollout(parameters, record=False):
        nonlocal evaluations
        evaluations += 1
        duration = float(parameters[0])
        offsets = np.r_[np.zeros((1, len(groups))), parameters[1:].reshape(3, len(groups))]
        mujoco.mj_resetData(model, data)
        data.qpos[:] = initial
        data.ctrl[:] = z["joint_pos"][0]
        mujoco.mj_forward(model, data)
        rows, q_history, controls = [], [initial.copy()], []
        best_cost, best_row = 1e6, None
        nonfoot_contacts = 0
        max_soft = max_speed = max_effort = 0.
        soft_energy = 0.
        left_impulse = right_impulse = 0.
        force_history = []
        airborne_detected = False
        flight_start = None
        flight_offset = None
        pitch_samples = []
        for step in range(int(round((duration + args.follow_through_seconds + args.flight_seconds) / .02))):
            u = min(step * .02 / duration, 1.)
            frame = u * 30
            target = np.array([np.interp(frame, support_frames, reference_target[:, j]) for j in range(len(names))])
            correction = np.array([np.interp(u, [0, 1/3, 2/3, 1], offsets[:, k]) for k in range(len(groups))])
            target += correction @ control_map
            if args.stance_prior == "balanced":
                desired, desired_velocity = initial[7:], np.zeros(len(names))
            else:
                desired = np.array([np.interp(frame,np.arange(len(poses)),joint_prior[:,j]) for j in range(len(names))])
                desired_velocity = np.array([np.interp(frame,np.arange(len(poses)),prior_velocity[:,j]) for j in range(len(names))]) * .6/duration
            if flight_start is not None and args.flight_seconds > 0:
                elapsed_air = step * .02 - flight_start
                air_frame = min(30 + max(elapsed_air,0) * source_fps, len(poses)-1)
                target = np.array([np.interp(air_frame,np.arange(len(poses)),full_target[:,j]) for j in range(len(names))])
                target += flight_offset * max(1 - elapsed_air / .12, 0)
                desired = np.array([np.interp(air_frame,np.arange(len(poses)),joint_prior[:,j]) for j in range(len(names))])
                desired_velocity = np.array([np.interp(air_frame,np.arange(len(poses)),prior_velocity[:,j]) for j in range(len(names))])
            if args.feedback_stiffness > 0:
                target += ((args.feedback_stiffness-kp)*(desired-data.qpos[7:])
                           +(args.feedback_damping-kd)*(desired_velocity-data.qvel[6:]))/kp
            data.ctrl[:] = target
            foot_force = {"left": 0., "right": 0.}
            for substep in range(4):
                mujoco.mj_step(model, data)
                max_effort = max(max_effort, float(np.max(np.abs(data.actuator_force) / effort)))
                for contact_index, contact in enumerate(data.contact):
                    pair = set(map(int, contact.geom))
                    if ground not in pair:
                        continue
                    other = next(iter(pair - {ground}))
                    if other not in foot_ids:
                        nonfoot_contacts += 1
                    else:
                        wrench = np.zeros(6)
                        mujoco.mj_contactForce(model, data, contact_index, wrench)
                        side = "left" if other in left_ids else "right"
                        foot_force[side] += abs(float(wrench[0])) / 4
            mujoco.mj_forward(model, data)
            mujoco.mj_subtreeVel(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                return 1e9, {"nonfinite": True}, None
            violation = np.maximum(np.maximum(lower - data.qpos[7:], data.qpos[7:] - upper), 0)
            max_soft = max(max_soft, float(violation.max()))
            soft_energy += float(violation @ violation) * .02
            max_speed = max(max_speed, float(np.max(np.abs(data.qvel[6:]) / velocity_limits)))
            left_impulse += foot_force["left"] * .02
            right_impulse += foot_force["right"] * .02
            force_history.append([foot_force["left"], foot_force["right"]])
            recent_impulse = np.sum(force_history[-10:], axis=0) * .02
            velocity = data.subtree_linvel[root].copy()
            momentum = data.subtree_angmom[root].copy()
            heights = foot_heights()
            up = data.xmat[root].reshape(3,3)[:,2]
            pitch_samples.append(float(np.arctan2(up @ forward_axis, up[2])))
            row = {"time": float(data.time), "com_velocity": velocity.tolist(), "angular_momentum": momentum.tolist(),
                   "foot_heights": heights.tolist(), "left_force_N": foot_force["left"], "right_force_N": foot_force["right"],
                   "com_height": float(data.subtree_com[root, 2])}
            row["recent_left_impulse_Ns"], row["recent_right_impulse_Ns"] = map(float, recent_impulse)
            # Continuous surrogate drives the search toward lift and forward
            # rotation; separate hard gates decide whether a takeoff exists.
            if data.time >= .25:
                cost = ((velocity[2] - 2.5)**2 * 4 + (velocity[0] - .8)**2 + velocity[1]**2 * 2
                        + np.sum(((momentum - np.array([0., 25., 0.])) / 8)**2))
                cost += float(np.maximum(.025 - heights, 0).sum()) * 80
                cost += max(.65 - data.subtree_com[root, 2], 0)**2 * 30
                cost += foot_force["right"] / (mass * 9.81) * 3
                cost += max(35 - recent_impulse[0], 0) / 15 + recent_impulse[1] / 20
                if cost < best_cost:
                    best_cost, best_row = float(cost), row
            if record:
                rows.append(row)
                q_history.append(data.qpos.copy())
                controls.append(target.copy())
            if data.time >= .35 and heights.min() > .025 and velocity[2] > .5:
                airborne_detected = True
                if args.flight_seconds == 0:
                    break
            if args.flight_seconds > 0 and flight_start is None and data.time >= .25 and heights.min() > .001 and velocity[2] > .5 and sum(foot_force.values()) < 1:
                flight_start = float(data.time)
                flight_offset = data.qpos[7:].copy() - z["joint_pos"][30]
            if flight_start is not None and data.time >= flight_start + args.flight_seconds:
                break
            if data.qpos[2] < .2:
                break
        cost = best_cost + (max_soft**2 + soft_energy) * args.soft_limit_weight + max(max_speed - 1, 0)**2 * 20 + min(nonfoot_contacts, 20) * 5
        report = {"cost": float(cost), "best_event": best_row, "max_soft_limit_violation_rad": max_soft,
                  "max_joint_speed_to_limit": max_speed, "max_actuator_effort_to_limit": max_effort,
                  "nonfoot_ground_contacts": nonfoot_contacts, "left_normal_impulse_Ns": left_impulse,
                  "right_normal_impulse_Ns": right_impulse, "duration": duration,
                  "airborne_detected": airborne_detected,
                  "flight_start_seconds": flight_start,
                  "forward_pitch_change_rad": float(np.unwrap(pitch_samples)[-1] - pitch_samples[0]) if pitch_samples else 0.,
                  "final_up_z": float(data.xmat[root].reshape(3,3)[2,2]),
                  "approved_for_training": False, "complete_flip_verified": False}
        report["takeoff_candidate_passed"] = bool(airborne_detected and best_row is not None
            and min(best_row["foot_heights"]) > .025 and best_row["com_velocity"][2] > 2.1
            and np.linalg.norm(np.array(best_row["angular_momentum"]) - [0,25,0]) < 7
            and best_row["recent_left_impulse_Ns"] > 25 and best_row["recent_right_impulse_Ns"] < 10
            and max_soft < 1e-3 and max_speed <= 1 and max_effort <= 1.000001 and nonfoot_contacts == 0)
        payload = (np.array(q_history), np.array(controls), rows) if record else None
        return float(cost), report, payload

    (args.out / "configuration.json").write_text(json.dumps({"arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        "source_sha256":hashlib.sha256(args.source.read_bytes()).hexdigest(), "groups":groups,
        "robot":PROFILE["robot"], "mass_kg":mass, "timestep":model.opt.timestep,
        "effort_limits":effort.tolist(), "initial_qpos":initial.tolist(),
        "static_gravity_torque_Nm":gravity_torque.tolist(),
        "limitations":"Isolated CPU forward-physics design, not mjlab policy validation; full single-leg flight/flip/landing gates remain."},indent=2))
    for iteration in range(args.iterations):
        population = np.clip(rng.normal(mean, std, (args.population, dimension)), bounds_lower, bounds_upper)
        population[0] = mean
        if best is not None:
            population[1] = best[1]
        outcomes = []
        for parameters in population:
            cost, report, _ = rollout(parameters)
            outcomes.append((cost, parameters.copy(), report))
            if best is None or cost < best[0]:
                best = (cost, parameters.copy(), report)
            if time.monotonic() - started > args.max_seconds:
                break
        outcomes.sort(key=lambda x:x[0])
        elite = np.array([item[1] for item in outcomes[:max(2, len(outcomes)//4)]])
        mean = .3 * mean + .7 * elite.mean(axis=0)
        std = np.maximum(.3 * std + .7 * elite.std(axis=0), np.r_[.015, np.full(dimension-1,.08)])
        cost, report, payload = rollout(best[1], record=True)
        q, controls, rows = payload
        np.savez(args.out / "best-forward.npz", parameters=best[1], control_units=np.array(args.control_units), qpos=q, controls=controls, fps=np.array([50.]))
        record = {"iteration":iteration,"evaluations":evaluations,"elapsed_s":time.monotonic()-started, **report}
        (args.out / "latest.json").write_text(json.dumps(record,indent=2))
        with log.open("a") as f:
            f.write(json.dumps(record)+"\n")
        print(json.dumps(record),flush=True)
        if time.monotonic() - started > args.max_seconds:
            break
    (args.out / "completed.json").write_text(json.dumps({"evaluations":evaluations,"elapsed_s":time.monotonic()-started,"approved_for_training":False},indent=2))


if __name__ == "__main__":
    main()
