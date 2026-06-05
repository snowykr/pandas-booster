"""Release contract tests."""

from __future__ import annotations

import sys

import pytest
from conftest import _load_release_contract_module, _project_version


def test_main_dispatches_tag_command(monkeypatch: pytest.MonkeyPatch):
    contract = _load_release_contract_module()

    monkeypatch.setattr(
        sys, "argv", ["check_release_contract.py", "tag", "--tag", f"v{_project_version()}"]
    )

    assert contract.main() == 0
