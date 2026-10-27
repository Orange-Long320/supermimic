"""Download pinned foreign GEM assets locally, checking upstream sizes/hashes.

Run locally before transferring the output to the project's server data directory.
Uses curl for HTTPS; never stores authentication or temporary signed URLs.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url, output, byte_range=None):
    command = [
        "curl", "--fail", "--location", "--silent", "--show-error",
        "--retry", "8", "--retry-all-errors", "--retry-delay", "3", "--connect-timeout", "20", "--max-time", "3600",
    ]
    if byte_range:
        # Separate resolve cache entries for differently signed range requests.
        url += "?download=true&part=" + byte_range
        command += ["--range", byte_range]
    subprocess.run(command + ["--output", str(output), url], check=True)


def ranged_fetch(item, part):
    """Keep a contiguous prefix and fetch the remaining ranges concurrently."""
    prefix = part.stat().st_size if part.exists() else 0
    if prefix > item["size"]:
        raise ValueError(f"Oversized partial file: {part}")
    ranges = []
    chunk_size = 128 * 1024 * 1024
    for start in range(prefix, item["size"], chunk_size):
        end = min(start + chunk_size, item["size"]) - 1
        ranges.append((start, end, part.with_name(f"{part.name}.{start}-{end}")))

    def get_chunk(row):
        start, end, chunk = row
        if not chunk.exists() or chunk.stat().st_size != end - start + 1:
            fetch(item["url"], chunk, f"{start}-{end}")
        if chunk.stat().st_size != end - start + 1:
            raise ValueError(f"Server did not return the requested byte range: {chunk.name}")
        print(f"Range ready {item['path']} {start}-{end}", flush=True)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(get_chunk, ranges))
    with part.open("ab") as output:
        for _, _, chunk in ranges:
            with chunk.open("rb") as source:
                shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
    for _, _, chunk in ranges:
        chunk.unlink()


def download(item, dest):
    path = dest / item["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        part = path.with_name(path.name + ".part")
        print(f"Downloading {item['path']} ({item['size']} bytes)", flush=True)
        if item["size"] > 256 * 1024 * 1024:
            ranged_fetch(item, part)
        else:
            fetch(item["url"], part)
        if part.stat().st_size != item["size"]:
            raise ValueError(f"Size mismatch: {path.name}")
        digest = sha256(part)
        if item.get("sha256") and digest != item["sha256"]:
            raise ValueError(f"Hash mismatch: {path.name}")
        part.rename(path)
    else:
        if path.stat().st_size != item["size"]:
            raise ValueError(f"Existing file size mismatch: {path.name}")
        digest = sha256(path)
        if item.get("sha256") and digest != item["sha256"]:
            raise ValueError(f"Existing file hash mismatch: {path.name}")
    print(f"Verified {item['path']} SHA256 {digest}", flush=True)
    result = dict(item, sha256=digest)
    # Per-file records survive an interrupted download of another asset.
    path.with_name(path.name + ".provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("workers must be in [1, 4]")
    manifest = json.loads(args.manifest.read_text())
    dest = args.dest.resolve()
    for item in manifest["files"]:
        if not (dest / item["path"]).resolve().is_relative_to(dest):
            parser.error("asset path must stay within destination")
    dest.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        manifest["files"] = list(pool.map(lambda item: download(item, dest), manifest["files"]))
    (dest / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
