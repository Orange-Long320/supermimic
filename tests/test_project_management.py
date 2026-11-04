"""Source isolation and immutable exports, without ML dependencies or hardware."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("project_manager", ROOT / "scripts/project.py")
manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manager)


class ProjectIsolationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.keyboard = self.base / "keyboard"
        self.stairs = self.base / "stairs"
        self.keyboard.mkdir()
        self.git(self.keyboard, "init", "-b", "main")
        self.git(self.keyboard, "config", "user.name", "Isolation Test")
        self.git(self.keyboard, "config", "user.email", "test@example.invalid")
        (self.keyboard / "src").mkdir()
        (self.keyboard / "src/policy.py").write_text("VALUE = 'keyboard baseline'\n")
        (self.keyboard / "uv.lock").write_text("version = 1\n")
        self.git(self.keyboard, "add", ".")
        self.git(self.keyboard, "commit", "-m", "baseline")
        self.baseline = self.git(self.keyboard, "rev-parse", "HEAD")
        self.git(self.keyboard, "worktree", "add", "-b", "stairs", str(self.stairs))
        (self.keyboard / "projects").mkdir()
        self.entries = {
            "keyboard": dict(path=".", branch="main", baseline=self.baseline,
                             artifact_namespace="keyboard", environment="/envs/keyboard"),
            "stairs": dict(path="../stairs", branch="stairs", baseline=self.baseline,
                           artifact_namespace="stairs", environment="/envs/stairs"),
        }
        self.save_registry()

    def git(self, path, *args):
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True, stderr=subprocess.PIPE
        ).strip()

    def save_registry(self):
        (self.keyboard / "projects/workspaces.json").write_text(
            json.dumps(dict(schema_version=1, workspaces=self.entries))
        )

    def test_stairs_edits_and_commits_leave_keyboard_source_and_head_unchanged(self):
        original = (self.keyboard / "src/policy.py").read_bytes()
        (self.stairs / "src/policy.py").write_text("VALUE = 'stairs experiment'\n")
        self.git(self.stairs, "add", ".")
        self.git(self.stairs, "commit", "-m", "stairs only")
        self.assertEqual((self.keyboard / "src/policy.py").read_bytes(), original)
        self.assertEqual(self.git(self.keyboard, "rev-parse", "HEAD"), self.baseline)
        self.assertNotEqual(manager.inspect(self.keyboard, "stairs")["commit"], self.baseline)

    def test_export_records_exact_commit_lock_and_archive_hash(self):
        out = self.base / "release"
        record = manager.export(self.keyboard, "stairs", out)
        self.assertEqual(record["source_commit"], self.baseline)
        self.assertEqual(record["project"], "stairs")
        self.assertEqual(record["source_sha256"], hashlib.sha256((out / "source.tar").read_bytes()).hexdigest())
        self.assertEqual(record["uv_lock_sha256"], hashlib.sha256(b"version = 1\n").hexdigest())
        with tarfile.open(out / "source.tar") as archive:
            self.assertEqual(archive.extractfile("src/policy.py").read(), b"VALUE = 'keyboard baseline'\n")
        self.assertEqual(json.loads((out / "release.json").read_text()), record)

    def test_dirty_source_and_wrong_branch_are_rejected(self):
        (self.stairs / "src/policy.py").write_text("uncommitted\n")
        with self.assertRaisesRegex(ValueError, "Commit workspace"):
            manager.export(self.keyboard, "stairs", self.base / "dirty-release")
        self.entries["stairs"]["branch"] = "wrong-branch"
        self.save_registry()
        with self.assertRaisesRegex(ValueError, "Expected branch"):
            manager.export(self.keyboard, "stairs", self.base / "wrong-release")

    def test_existing_even_empty_release_is_not_overwritten(self):
        out = self.base / "release"
        out.mkdir()
        with self.assertRaisesRegex(ValueError, "overwrite"):
            manager.export(self.keyboard, "stairs", out)
        self.assertEqual(list(out.iterdir()), [])

    def test_cross_workspace_source_link_is_rejected(self):
        (self.stairs / "src/policy.py").unlink()
        (self.stairs / "src/policy.py").symlink_to(self.keyboard / "src/policy.py")
        state = manager.inspect(self.keyboard, "stairs")
        self.assertTrue(any("External source link" in error for error in state["errors"]))

    def test_unknown_workspace_and_unrelated_baseline_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown workspace"):
            manager.inspect(self.keyboard, "typo")
        self.entries["stairs"]["baseline"] = "0" * 40
        self.save_registry()
        self.assertTrue(manager.inspect(self.keyboard, "stairs")["errors"])


if __name__ == "__main__":
    unittest.main()
