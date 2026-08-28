"""
Pytest configuration for tests.
"""

import sys
from importlib import resources
from pathlib import Path

import pytest

# Add project root to sys.path so src module can be imported
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@pytest.fixture(autouse=True)
def use_stable_test_config(monkeypatch):
    """Autouse fixture to make package resource lookup return the tests directory.

    This ensures `src.config` will load `tests/config.yaml` as the default
    configuration during tests instead of the package's changing file.
    """
    tests_dir = Path(__file__).parent

    # Monkeypatch importlib.resources.files to return the tests directory Path.
    # `resources.files(__name__).joinpath('config.yaml')` will become
    # `tests_dir.joinpath('config.yaml')`.
    monkeypatch.setattr(resources, "files", lambda pkg: tests_dir)
    yield
