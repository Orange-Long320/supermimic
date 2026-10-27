"""Download only pinned SOMA X2 LFS assets locally; verify size and SHA256."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "vendor/soma-retargeter/soma_retargeter/assets/robotics/agibot/x2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--repository", default="NVIDIA/soma-retargeter")
    parser.add_argument("--include", default="**/*")
    args = parser.parse_args()
    base = args.base
    pending = []
    for path in base.glob(args.include):
        if not path.is_file() or path.stat().st_size > 1024:
            continue
        content = path.read_bytes()
        if not content.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
            continue
        fields = dict(line.split(" ", 1) for line in content.decode().splitlines())
        pending.append((path, fields["oid"].split(":")[1], int(fields["size"])))
    request = urllib.request.Request(
        f"https://github.com/{args.repository}.git/info/lfs/objects/batch",
        data=json.dumps({"operation": "download", "transfers": ["basic"],
                         "objects": [{"oid": oid, "size": size}
                                     for _, oid, size in pending]}).encode(),
        headers={"Accept": "application/vnd.git-lfs+json",
                 "Content-Type": "application/vnd.git-lfs+json"},
    )
    response = json.load(urllib.request.urlopen(request, timeout=60))
    objects = {obj["oid"]: obj for obj in response["objects"]}
    for index, (path, oid, size) in enumerate(pending, 1):
        action = objects[oid]["actions"]["download"]
        req = urllib.request.Request(action["href"], headers=action.get("header", {}))
        temporary = path.with_name(path.name + ".download")
        digest = hashlib.sha256()
        count = 0
        with urllib.request.urlopen(req, timeout=120) as stream, temporary.open("wb") as out:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                count += len(chunk)
                out.write(chunk)
        if count != size or digest.hexdigest() != oid:
            raise RuntimeError(f"LFS verification failed: {path}")
        temporary.replace(path)
        print(f"{index}/{len(pending)} verified {path.relative_to(base)}", flush=True)


if __name__ == "__main__":
    main()
