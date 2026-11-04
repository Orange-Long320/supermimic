"""Keep the stairs task independent of keyboard runtime and patch installers."""

import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = ("supermimic.control", "supermimic.deploy", "supermimic.locomotion",
             "keyboard_supervision", "supermimic_keyboard")


def forbidden_imports(source):
    violations = []
    for node in ast.walk(ast.parse(source)):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            names = [base] + [base + "." + alias.name for alias in node.names]
        elif isinstance(node, ast.Call) and node.args:
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name in ("__import__", "import_module") and isinstance(node.args[0], ast.Constant):
                names = [node.args[0].value]
        violations += [name for name in names if isinstance(name, str) and any(
            name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN)]
    return violations


class StairsBoundaryTests(unittest.TestCase):
    def test_task_and_entrypoints_do_not_import_keyboard(self):
        for directory in (ROOT / "src/supermimic_stairs", ROOT / "scripts/stairs"):
            for path in directory.rglob("*.py"):
                with self.subTest(path=path):
                    self.assertEqual(forbidden_imports(path.read_text()), [])

    def test_guard_catches_direct_from_and_literal_dynamic_imports(self):
        for source in (
            "import supermimic.deploy.core",
            "from supermimic import deploy",
            "from keyboard_supervision import install",
            "importlib.import_module('supermimic.locomotion.trainer')",
        ):
            self.assertTrue(forbidden_imports(source), source)
        self.assertFalse(forbidden_imports("from supermimic.world_model import WorldModel"))


if __name__ == "__main__":
    unittest.main()
