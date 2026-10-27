"""Build a diagnostic world-translation candidate for a near-stationary camera.

Replaces learned velocity integration with camera-space translation transformed
by the recorded background rotations. Camera translation is unobserved and is
explicitly assumed negligible; this is not a generic moving-camera correction.
The original prediction is never overwritten. Ground/contact correction remains
a separate official GEM postprocessing step.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    result = torch.load(args.input, map_location="cpu", weights_only=False)
    camera = result["pred_soma_params_incam"]
    world = result["pred_soma_params_global"]
    if camera["global_orient"].shape[0] != 1:
        raise ValueError("one person required")
    reference = (
        Rotation.from_rotvec(camera["global_orient"][0, 0].numpy()).as_matrix()
        @ Rotation.from_rotvec(world["global_orient"][0, 0].numpy()).as_matrix().T
    )
    relative = result["camera"]["R_w2c"].numpy()
    relative = relative @ relative[0].T
    world_to_camera = relative @ reference
    translation = np.einsum("tji,tj->ti", world_to_camera, camera["transl"][0].numpy())
    translation -= translation[:1]
    translation += world["transl"][0, 0].numpy()
    if not np.isfinite(translation).all():
        raise ValueError("non-finite candidate translation")
    replacement = dict(world, transl=torch.from_numpy(translation.astype(np.float32))[None])
    result["pred_soma_params_global"] = replacement
    result["pred_body_params_global"] = replacement
    result["trajectory_candidate"] = {
        "method": "camera_translation_with_background_rotation",
        "source": str(args.input.resolve()),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "assumption": "camera translation is negligible; background relative rotations are used",
        "camera_rotation_max_deg": float(np.rad2deg(Rotation.from_matrix(relative).magnitude()).max()),
        "limitation": "camera-space depth and gravity alignment remain estimates; requires contact and physical review",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(result["trajectory_candidate"])


if __name__ == "__main__":
    main()
