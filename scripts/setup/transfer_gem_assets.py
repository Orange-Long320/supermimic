"""Transfer only verified GEM downloads, then atomically signal server readiness."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time


REMOTE = r'''
import hashlib,json,sys
from pathlib import Path
request=json.load(sys.stdin)
root=Path(request['root']).resolve()
mode=request['mode']
if mode == 'prepare':
    path=root/request['path']
    assert path.resolve().is_relative_to(root)
    path.parent.mkdir(parents=True,exist_ok=True)
elif mode == 'verify':
    item=request['item']; path=root/item['path']; staged=Path(str(path)+'.incoming')
    assert path.resolve().is_relative_to(root)
    source=path if path.exists() else staged
    assert source.stat().st_size == item['size'], 'remote size mismatch'
    digest=hashlib.sha256()
    with source.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''): digest.update(block)
    assert digest.hexdigest()==item['sha256'], 'remote SHA256 mismatch'
    if source == staged: staged.rename(path)
    path.with_name(path.name+'.provenance.json').write_text(json.dumps(item,indent=2)+'\n')
elif mode == 'ready':
    manifest=request['manifest']
    for item in manifest['files']:
        record=json.loads((root/(item['path']+'.provenance.json')).read_text())
        assert record==item
        assert (root/item['path']).stat().st_size==item['size']
    assert (root/'weights/yolox_humanart.onnx').is_file(), 'person detector missing'
    staged=root/'READY.json.incoming'
    staged.write_text(json.dumps(manifest,indent=2)+'\n')
    staged.rename(root/'READY.json')
else: raise ValueError(mode)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--host", default="2025-Long-robot")
    parser.add_argument("--remote", default="/home/2025-Long.xh/HHD/data/supermimic/assets/gem-x")
    parser.add_argument("--timeout", type=float, default=7200)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())

    def remote(**payload):
        command = "/home/2025-Long.xh/HHD/venvs/supermimic/bin/python -c " + shlex.quote(REMOTE)
        subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", args.host, command],
                       input=json.dumps(dict(payload, root=args.remote)), text=True, check=True)

    verified, remaining = {}, {row["path"]: row for row in manifest["files"]}
    deadline = time.monotonic() + args.timeout
    print("Waiting for checksum-verified local files", flush=True)
    while remaining:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Missing verified assets: {list(remaining)}")
        for path, expected in list(remaining.items()):
            record = args.local / (path + ".provenance.json")
            if not record.is_file():
                continue
            item = json.loads(record.read_text())
            if any(item.get(key) != expected[key] for key in ("path", "url", "size")):
                raise ValueError(f"Provenance differs: {path}")
            if expected.get("sha256") and item["sha256"] != expected["sha256"]:
                raise ValueError(f"Expected hash differs: {path}")
            remote(mode="prepare", path=path)
            print(f"Uploading {path}", flush=True)
            staged = args.remote + "/" + path + ".incoming"
            subprocess.run(["scp", "-q", str(args.local / path), f"{args.host}:{staged}"], check=True)
            remote(mode="verify", item=item)
            print(f"Verified on server {path}", flush=True)
            verified[path] = item
            del remaining[path]
        if remaining:
            time.sleep(10)
    manifest["files"] = [verified[row["path"]] for row in manifest["files"]]
    remote(mode="ready", manifest=manifest)
    print("All assets verified; server READY.json created", flush=True)


if __name__ == "__main__":
    main()
