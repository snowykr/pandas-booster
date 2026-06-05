from __future__ import annotations

import os
from pathlib import Path

import conftest
import pytest


def test_prepend_repo_pythonpath_preserves_existing_entries() -> None:
    repo_python = str(Path(__file__).resolve().parents[1] / "python")

    assert conftest.prepend_repo_pythonpath("/tmp/site-packages", repo_python) == (
        f"{repo_python}{os.pathsep}/tmp/site-packages"
    )


@pytest.mark.skipif(
    conftest.wheel_smoke_mode(),
    reason="wheel smoke must not prepend checkout python/ ahead of site-packages",
)
def test_pytest_configure_updates_environment_for_child_processes() -> None:
    entries = os.environ["PYTHONPATH"].split(os.pathsep)
    assert entries[0] == str(Path(__file__).resolve().parents[1] / "python")
