"""Export GEM's actual SOMA FK to the existing retargeter's Y-up/cm BVH.

Run in the isolated GEM environment on the server. This preserves the recovered
motion, including airborne translation. It does not make a motion physically
feasible or approve it for policy training.
"""

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.spatial.transform import Rotation
import torch


def canonical_hierarchy(path):
    names, parents, stack = [], [], []
    pending = None
    for line in path.read_text().split("MOTION")[0].splitlines():
        line = line.strip()
        if line.startswith(("ROOT ", "JOINT ")):
            names.append(line.split()[1])
            parents.append(-1 if not stack else stack[-1])
            pending = len(names) - 1
        elif line == "End Site":
            pending = None
        elif line == "{":
            stack.append(pending)
        elif line == "}":
            stack.pop()
    return names, parents


def local_transforms(world, parents):
    result = world.copy()
    for i, parent in enumerate(parents):
        if parent >= 0:
            result[:, i] = np.linalg.inv(world[:, parent]) @ world[:, i]
    return result


def write_bvh(path, names, parents, world, fps):
    local = local_transforms(world, parents)
    # Full position channels preserve per-frame translations exactly; offsets
    # only describe a reference skeleton and are not added to those channels.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        eulers = Rotation.from_matrix(local[..., :3, :3].reshape(-1, 3, 3)).as_euler(
            "ZYX", degrees=True
        ).reshape(len(world), len(names), 3)
    children = {i: [] for i in range(len(names))}
    for i, parent in enumerate(parents):
        if parent >= 0:
            children[parent].append(i)
    order, lines = [], ["HIERARCHY"]

    def emit(i, depth):
        indent = "  " * depth
        order.append(i)
        lines.extend([f"{indent}{'ROOT' if parents[i] < 0 else 'JOINT'} {names[i]}", indent + "{"])
        offset = local[0, i, :3, 3] * 100
        lines.append(indent + "  OFFSET " + " ".join(f"{v:.10f}" for v in offset))
        lines.append(indent + "  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation")
        for child in children[i]:
            emit(child, depth + 1)
        if not children[i]:
            lines.extend([indent + "  End Site", indent + "  {", indent + "    OFFSET 0 0 0", indent + "  }"])
        lines.append(indent + "}")

    roots = [i for i, parent in enumerate(parents) if parent < 0]
    if len(roots) != 1:
        raise ValueError("BVH requires one virtual Root")
    emit(roots[0], 0)
    positions = local[..., :3, 3]
    values = np.concatenate([positions[:, order, :] * 100, eulers[:, order]], axis=-1)
    with path.open("x") as stream:
        stream.write("\n".join(lines) + f"\nMOTION\nFrames: {len(world)}\nFrame Time: {1 / fps:.12f}\n")
        np.savetxt(stream, values.reshape(len(world), -1), fmt="%.10f")
    # Euler wrap/gimbal choices must leave all FK unchanged.
    restored = np.tile(np.eye(4), (len(world), len(names), 1, 1))
    restored[..., :3, :3] = Rotation.from_euler("ZYX", eulers.reshape(-1, 3), degrees=True).as_matrix().reshape(len(world), len(names), 3, 3)
    restored[..., :3, 3] = local[..., :3, 3]
    for i in order:
        if parents[i] >= 0:
            restored[:, i] = restored[:, parents[i]] @ restored[:, i]
    error = float(np.max(np.abs(restored - world)))
    if error > 1e-5:
        raise ValueError(f"BVH Euler/FK roundtrip failed: {error}")
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--soma-assets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from gem.common.soma import SomaFixedGs1BodyAdapter, SomaRuntimeConfig
    from gem.inference.timing import prediction_fps

    source = args.input.resolve(strict=True)
    result = torch.load(source, map_location="cpu", weights_only=False)
    params = result["pred_soma_params_global"]
    if params["body_pose"].shape[0] != 1:
        raise ValueError("Export one recovered person at a time")
    body = SomaFixedGs1BodyAdapter(SomaRuntimeConfig(data_root=args.soma_assets, device="cpu", lod="xlo", apply_correctives=False))
    with torch.inference_mode():
        output = body(**params, return_joints_only=True)
    world = output["transforms"][0].numpy().astype(np.float64)
    names = [str(x) for x in body.runtime.public_joint_names]
    parents = body.runtime.output_joint_parent_ids.cpu().tolist()
    # SOMA's virtual Root uses a self-parent sentinel; BVH uses -1.
    if names[0] != "Root" or parents[0] not in (-1, 0):
        raise ValueError("Unexpected SOMA virtual Root convention")
    parents[0] = -1
    if not np.isfinite(world).all():
        raise ValueError("Recovered FK contains non-finite values")
    root = Path(__file__).resolve().parents[2]
    cn, cp = canonical_hierarchy(root / "vendor/soma-retargeter/soma_retargeter/assets/soma/soma_zero_frame0.bvh")
    expected = {name: None if cp[i] < 0 else cn[cp[i]] for i, name in enumerate(cn)}
    actual = {name: None if parents[i] < 0 else names[parents[i]] for i, name in enumerate(names)}
    if expected != actual:
        raise ValueError("GEM and pinned retargeter have different named skeleton hierarchies")
    args.out.mkdir(parents=True, exist_ok=False)
    fps = prediction_fps(result)
    error = write_bvh(args.out / "webster.bvh", names, parents, world, fps)
    np.savez_compressed(args.out / "expected_fk.npz", world=world, names=names, parents=parents, fps=fps)
    hips = world[:, names.index("Hips"), :3, 3]
    chest = world[:, names.index("Chest"), :3, 3]
    torso = chest - hips
    torso /= np.linalg.norm(torso, axis=-1, keepdims=True)
    report = {
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "frames": len(world), "fps": fps, "duration_s": (len(world) - 1) / fps,
        "coordinates": "Y up, positions in meters internally; BVH positions in centimeters; retargeter's Mujoco SpaceConverter applies Y-to-Z once",
        "euler_fk_roundtrip_max_abs_error": error,
        "torso_up_y_min": float(torso[:, 1].min()), "torso_up_y_final": float(torso[-1, 1]),
        "hips_height_range_m": [float(hips[:, 1].min()), float(hips[:, 1].max())],
        "torso_inversion_detected": bool(torso[:, 1].min() < -0.5),
        "status": "needs_visual_and_physical_review; not approved for policy training",
    }
    (args.out / "export.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
