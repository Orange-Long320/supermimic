import importlib.util
from pathlib import Path

import pytest


def test_launcher_compares_canonical_paths_and_rejects_old_fixture(tmp_path):
    file = Path(__file__).parents[1]/'scripts/experiments/run_idle_context_comparison.py'
    spec = importlib.util.spec_from_file_location('idle_comparison', file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = tmp_path/'actual'
    actual.mkdir()
    alias = tmp_path/'HHD'
    alias.symlink_to(actual, target_is_directory=True)
    script = actual/'start-sim.sh'
    script.write_text(f'run={actual}\nexport ROS_DOMAIN_ID=206\n')
    module.validate_fixture_launcher(alias, 206)
    script.write_text(f'run={tmp_path / "old"}\nexport ROS_DOMAIN_ID=206\n')
    with pytest.raises(ValueError, match='working directory'):
        module.validate_fixture_launcher(alias, 206)
    script.write_text(f'run={actual}\nexport ROS_DOMAIN_ID=201\n')
    with pytest.raises(ValueError, match='ROS domain'):
        module.validate_fixture_launcher(alias, 206)
