"""Check exported BVH through the pinned retargeter's actual importer (CPU)."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from soma_retargeter.io.bvh import load_bvh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    skeleton, animation = load_bvh(str(args.directory / "webster.bvh"))
    expected = np.load(args.directory / "expected_fk.npz")
    order = [expected["names"].tolist().index(name) for name in skeleton.joint_names]
    if animation.num_frames != len(expected["world"]) or not np.isclose(animation.sample_rate, expected["fps"]):
        raise ValueError("Imported clock differs from recovered motion")
    position_error = rotation_error = 0.0
    for frame in range(animation.num_frames):
        actual = np.asarray(animation.compute_global_transforms(frame))
        if not np.isfinite(actual).all():
            raise ValueError("Non-finite imported transforms")
        positions = actual[:, :3]
        rotations = Rotation.from_quat(actual[:, 3:7]).as_matrix()
        target = expected["world"][frame][order]
        position_error = max(position_error, float(np.max(np.abs(positions - target[:, :3, 3]))))
        rotation_error = max(rotation_error, float(np.max(np.abs(rotations - target[:, :3, :3]))))
    report = {"frames": animation.num_frames, "fps": animation.sample_rate,
              "max_position_error_m": position_error, "max_rotation_matrix_error": rotation_error}
    if position_error >= 1e-5 or rotation_error >= 1e-5:
        raise ValueError(f"Retarget importer changed the reference: {report}")
    (args.directory / "import_roundtrip.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
