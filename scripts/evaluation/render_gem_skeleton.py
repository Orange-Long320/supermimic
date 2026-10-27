"""Render recovered camera-space joints over the source clip, on CPU."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--soma-assets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from gem.common.soma import SomaFixedGs1BodyAdapter, SomaRuntimeConfig
    from gem.inference.video import normalize_video, VideoFrames
    from gem.inference.render import RGBVideoWriter
    from gem.inference.timing import prediction_fps

    result = torch.load(args.input, map_location="cpu", weights_only=False)
    body = SomaFixedGs1BodyAdapter(SomaRuntimeConfig(data_root=args.soma_assets, device="cpu", lod="xlo", apply_correctives=False))
    with torch.inference_mode():
        points = body(**result["pred_soma_params_incam"], return_joints_only=True)["transforms"][0, :, :, :3, 3].numpy()
    names = [str(n) for n in body.runtime.public_joint_names]
    parents = body.runtime.output_joint_parent_ids.tolist()
    camera = result["conditions"]["K_fullimg"][0].numpy()
    projected = np.einsum("tij,tkj->tki", camera, points)
    projected = projected[..., :2] / projected[..., 2:3]
    fps = prediction_fps(result)
    args.out.mkdir(parents=True, exist_ok=False)
    source = result["video"]["source"]
    snapshots = set(np.linspace(0, len(points) - 1, min(12, len(points)), dtype=int).tolist())
    thumbs = []
    with TemporaryDirectory(prefix="webster-overlay-") as temp:
        video = Path(temp) / "normalized.mp4"
        normalize_video(source["path"], video, start=source["requested_start"], duration=source["selected_duration"])
        with VideoFrames(video) as frames:
            if len(frames) != len(points) or not np.isclose(frames.fps, fps):
                raise ValueError("Recovered motion and source video have different clocks")
            with RGBVideoWriter(args.out / "overlay.mp4", frames.width, frames.height, fps) as writer:
                for frame in range(len(points)):
                    rgb = frames[frame].copy()
                    for joint in range(1, len(names)):
                        parent = parents[joint]
                        if parent <= 0:
                            continue
                        if names[joint].startswith(("LeftHand", "RightHand")) and names[joint] not in ("LeftHand", "RightHand"):
                            continue
                        pair = projected[frame, [joint, parent]]
                        if not np.isfinite(pair).all() or min(points[frame, [joint, parent], 2]) <= 0:
                            continue
                        if np.max(np.abs(pair)) > 4 * max(frames.width, frames.height):
                            continue
                        a, b = [tuple(np.rint(v).astype(int)) for v in pair]
                        cv2.line(rgb, a, b, (0, 255, 200), 4, cv2.LINE_AA)
                        cv2.circle(rgb, a, 5, (255, 150, 0), -1)
                    cv2.putText(rgb, f"Recovered 3D skeleton | {frame / fps:.2f}s | quality review", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    writer.write(rgb)
                    if frame in snapshots:
                        thumbs.append(cv2.resize(rgb, (480, 270)))
    while len(thumbs) % 4:
        thumbs.append(np.zeros_like(thumbs[0]))
    sheet = np.concatenate([np.concatenate(thumbs[i:i + 4], axis=1) for i in range(0, len(thumbs), 4)], axis=0)
    cv2.imwrite(str(args.out / "overview.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    (args.out / "preview.json").write_text(json.dumps({"frames": len(points), "fps": fps, "description": "Recovered camera-space FK projection; not ground truth or robot policy execution"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
