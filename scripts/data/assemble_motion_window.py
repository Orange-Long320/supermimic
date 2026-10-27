"""Reinsert a fitted pose window, regenerating velocities across both joins.

This does not repair or approve the result. Run full-clip independent audits.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--window-csv", type=Path, required=True)
    p.add_argument("--offset", type=int, required=True)
    p.add_argument("--contacts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    from supermimic.data.motion_convert import load_g1_model, convert_csv_to_npz
    m = load_g1_model()
    z = np.load(args.source)
    root = z["body_names"].tolist().index(m.body(int(m.jnt_bodyid[0])).name)
    original = np.c_[z["body_pos_w"][:, root], z["body_quat_w"][:, root][:, [1, 2, 3, 0]], z["joint_pos"]].astype(float)
    window = np.loadtxt(args.window_csv, delimiter=",", ndmin=2)
    assert 0 <= args.offset < len(original)
    assert window.shape[1] == original.shape[1]
    assert args.offset + len(window) <= len(original)
    assert np.isfinite(window).all()
    assert np.max(np.abs(np.linalg.norm(window[:, 3:7], axis=1) - 1)) < 1e-5
    combined = original.copy()
    combined[args.offset:args.offset + len(window)] = window
    args.out.mkdir(parents=True, exist_ok=False)
    csv = args.out / "webster.csv"
    np.savetxt(csv, combined, delimiter=",", fmt="%.10f")
    fps = float(z["fps"].reshape(-1)[0])
    convert_csv_to_npz(csv, args.out / "webster.npz", fps, fps)
    (args.out / "contacts.json").write_text(args.contacts.read_text())
    report = {"source": str(args.source.resolve()), "window_csv": str(args.window_csv.resolve()),
              "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
              "window_sha256": hashlib.sha256(args.window_csv.read_bytes()).hexdigest(),
              "offset": args.offset, "window_frames": len(window), "total_frames": len(combined),
              "prefix_poses_preserved": bool(np.array_equal(combined[:args.offset], original[:args.offset])),
              "velocities_regenerated_across_joins": True,
              "approved_for_training": False,
              "remaining": "Full-clip geometry, gravity, momentum, contact and torque audits; actual closed-loop validation."}
    (args.out / "assembly.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
