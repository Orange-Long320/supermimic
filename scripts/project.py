"""Inspect isolated project workspaces and export immutable source releases.

Standard library only. This tool does not train, deploy, merge, or install dependencies.
Run from the management checkout: python3 scripts/project.py status
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *args], text=True, stderr=subprocess.PIPE
    ).strip()


def registry(root: Path) -> dict:
    data = json.loads((root / "projects/workspaces.json").read_text())
    if data["schema_version"] != 1:
        raise ValueError("Unsupported workspace registry version")
    return data["workspaces"]


def inspect(root: Path, name: str) -> dict:
    entries = registry(root)
    if name not in entries:
        raise ValueError(f"Unknown workspace {name!r}; choose {', '.join(entries)}")
    entry = entries[name]
    path = Path(entry["path"]).expanduser()
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    result = dict(entry, name=name, path=str(path), errors=[])
    if not path.is_dir():
        result["errors"].append("Workspace does not exist")
        return result
    try:
        top = Path(git(path, "rev-parse", "--show-toplevel")).resolve()
        if top != path:
            result["errors"].append("Path is not a worktree root")
            return result
        result["commit"] = git(path, "rev-parse", "HEAD")
        result["actual_branch"] = git(path, "branch", "--show-current")
        result["dirty"] = bool(git(path, "status", "--porcelain", "--untracked-files=normal"))
        if result["actual_branch"] != entry["branch"]:
            result["errors"].append(f"Expected branch {entry['branch']}")
        subprocess.run(
            ["git", "-C", str(path), "merge-base", "--is-ancestor", entry["baseline"], "HEAD"],
            check=True, capture_output=True,
        )
        # Cross-worktree links would defeat source isolation. Asset/data links are
        # intentionally outside this source-only check and need dataset manifests.
        for folder in ("src", "scripts", "configs"):
            for item in [path / folder, *(path / folder).rglob("*")]:
                if item.is_symlink() and not item.resolve().is_relative_to(path):
                    result["errors"].append(f"External source link: {item.relative_to(path)}")
    except (subprocess.CalledProcessError, OSError) as exc:
        result["errors"].append(f"Git/workspace check failed: {exc}")
    return result


def export(root: Path, name: str, destination: Path) -> dict:
    state = inspect(root, name)
    if state["errors"]:
        raise ValueError("; ".join(state["errors"]))
    if state["dirty"]:
        raise ValueError("Commit workspace changes before exporting a source release")
    destination = destination.resolve()
    if destination.exists():
        raise ValueError(f"Refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".source-release-", dir=destination.parent))
    try:
        source = Path(state["path"])
        archive = temp / "source.tar"
        with archive.open("xb") as output:
            subprocess.run(
                ["git", "-C", str(source), "archive", "--format=tar", state["commit"]],
                stdout=output, check=True,
            )
        lock = subprocess.check_output(
            ["git", "-C", str(source), "show", f"{state['commit']}:uv.lock"]
        )
        digest = hashlib.sha256()
        with archive.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        record = {
            "schema_version": 1,
            "project": name,
            "artifact_namespace": state["artifact_namespace"],
            "source_commit": state["commit"],
            "source_branch": state["actual_branch"],
            "baseline": state["baseline"],
            "source_sha256": digest.hexdigest(),
            "uv_lock_sha256": hashlib.sha256(lock).hexdigest(),
            "environment": state["environment"],
            "validation": "source export only; no policy or hardware acceptance implied",
        }
        (temp / "release.json").write_text(json.dumps(record, indent=2) + "\n")
        # Exclusive creation also rejects an empty directory or a concurrent
        # exporter. No existing release is ever overwritten.
        destination.mkdir(exist_ok=False)
        archive.replace(destination / "source.tar")
        (temp / "release.json").replace(destination / "release.json")
        return record
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "check", "export"))
    parser.add_argument("project", nargs="?")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "export":
            if args.project is None or args.out is None:
                parser.error("export requires a project and --out")
            print(json.dumps(export(ROOT, args.project, args.out), indent=2))
            return
        names = [args.project] if args.project else registry(ROOT)
        states = [inspect(ROOT, name) for name in names]
        print(json.dumps(states, indent=2))
        if args.command == "check" and any(s["errors"] or s.get("dirty") for s in states):
            raise SystemExit(1)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
