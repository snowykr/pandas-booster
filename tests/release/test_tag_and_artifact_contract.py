"""Release contract tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from conftest import _load_release_contract_module, _project_version


def test_validate_tag_requires_v_prefix():
    contract = _load_release_contract_module()

    with pytest.raises(contract.ContractError, match="tags must start with 'v'"):
        contract.validate_tag(argparse.Namespace(tag="0.1.0"))


def test_validate_tag_requires_version_match():
    contract = _load_release_contract_module()

    with pytest.raises(contract.ContractError, match="Release tag/version mismatch"):
        contract.validate_tag(argparse.Namespace(tag="v9.9.9"))


def test_validate_tag_accepts_current_project_version():
    contract = _load_release_contract_module()

    assert contract.validate_tag(argparse.Namespace(tag=f"v{_project_version()}")) == 0


def test_validate_artifacts_requires_expected_counts(tmp_path: Path):
    contract = _load_release_contract_module()
    dist_path = tmp_path / "dist"
    dist_path.mkdir()
    (dist_path / "pandas_booster-0.1.0-cp311.whl").write_text("wheel", encoding="utf-8")

    with pytest.raises(contract.ContractError, match="Expected exactly 1 sdist file"):
        contract.validate_artifacts(
            argparse.Namespace(dist=str(dist_path), expected_wheel_count=1, require_sdist=True)
        )


def test_validate_artifacts_accepts_expected_counts(tmp_path: Path):
    contract = _load_release_contract_module()
    dist_path = tmp_path / "dist"
    dist_path.mkdir()
    (dist_path / "pandas_booster-0.1.0-cp311.whl").write_text("wheel", encoding="utf-8")
    (dist_path / "pandas_booster-0.1.0.tar.gz").write_text("sdist", encoding="utf-8")

    assert (
        contract.validate_artifacts(
            argparse.Namespace(dist=str(dist_path), expected_wheel_count=1, require_sdist=True)
        )
        == 0
    )
