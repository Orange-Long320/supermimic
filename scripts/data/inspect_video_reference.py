"""CPU-only source-video inspection; does not infer or retarget motion.

Run on the project server with its explicit Python interpreter and FFmpeg.
Preserves source timing and refuses to overwrite an existing inspection.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--step", type=float, default=1)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--clip", action="store_true")
    args = parser.parse_args()
    if args.start < 0 or args.step <= 0 or args.columns < 1 or args.width < 64:
        parser.error("invalid time or thumbnail layout")
    source = args.input.resolve(strict=True)
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(source)
    ]))
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    duration = float(video.get("duration", probe["format"]["duration"]))
    end = args.end if args.end is not None else duration
    if not args.start < end <= duration:
        parser.error(f"require 0 <= start < end <= {duration}")
    count = math.ceil((end - args.start) / args.step)
    if count > 150:
        parser.error("at most 150 thumbnails per inspection; split the time range")
    args.out.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    commands = []
    # Timestamp is printed before fps sampling, so it refers to source time.
    # select includes the first decoded frame and then samples every `step` seconds.
    filters = (
        f"trim=start={args.start}:end={end},"
        f"select=isnan(prev_selected_t)+gte(t-prev_selected_t\\,{max(0, args.step - 1e-6)}),"
        f"scale={args.width}:-2,"
        "drawtext=text='%{pts\\:hms}':x=6:y=6:fontsize=18:"
        "fontcolor=white:box=1:boxcolor=black@0.7,"
        f"tile={args.columns}x{math.ceil(count / args.columns)}"
    )
    overview_command = [
        "ffmpeg", "-nostdin", "-v", "error", "-threads", "2", "-i", str(source),
        "-vf", filters, "-frames:v", "1", "-threads", "2", "-n", str(args.out / "overview.jpg")
    ]
    commands.append(overview_command)
    subprocess.run(overview_command, check=True)
    if args.clip:
        clip_command = [
            "ffmpeg", "-nostdin", "-v", "error", "-threads", "2", "-i", str(source),
            "-ss", str(args.start), "-t", str(end - args.start), "-map", "0:v:0",
            "-an", "-c:v", "libx264", "-crf", "16", "-preset", "fast", "-threads", "2",
            "-vsync", "0", "-movflags", "+faststart", "-n", str(args.out / "clip.mp4")
        ]
        commands.append(clip_command)
        subprocess.run(clip_command, check=True)
    manifest = {
        "source": str(source), "source_sha256": digest.hexdigest(),
        "source_probe": probe, "source_start_seconds": args.start,
        "source_end_seconds": end, "thumbnail_step_seconds": args.step,
        "timing": "source speed preserved; actual physical speed requires visual review",
        "status": "video inspection only; no 3D motion or training data generated",
        "commands": commands,
    }
    (args.out / "inspection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "start": args.start, "end": end}))


if __name__ == "__main__":
    main()
