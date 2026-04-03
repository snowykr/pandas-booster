from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")


def _load_release_contract_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "check_release_contract.py"
    spec = importlib.util.spec_from_file_location("check_release_contract", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load release contract script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _project_version() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    with (repo_root / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def test_validate_metadata_requires_release_readme_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    contract = _load_release_contract_module()

    (tmp_path / "scripts").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        """
[build-system]
requires = [\"maturin>=1.4,<2.0\"]

[project]
name = \"pandas-booster\"
version = \"0.1.0\"
readme = \"README.md\"
requires-python = \">=3.9\"
classifiers = [
    \"Programming Language :: Python :: 3.9\",
    \"Programming Language :: Python :: 3.10\",
    \"Programming Language :: Python :: 3.11\",
    \"Programming Language :: Python :: 3.12\",
]

[project.urls]
Homepage = \"https://github.com/snowykr/pandas-booster\"
Repository = \"https://github.com/snowykr/pandas-booster\"
Issues = \"https://github.com/snowykr/pandas-booster/issues\"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "Cargo.toml").write_text(
        """
[package]
name = \"pandas_booster\"
version = \"0.1.0\"
repository = \"https://github.com/snowykr/pandas-booster\"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# pandas-booster\n", encoding="utf-8")

    monkeypatch.setattr(contract, "project_root", lambda: tmp_path)

    with pytest.raises(contract.ContractError, match="README.md must contain"):
        contract.validate_metadata(argparse.Namespace())


def test_validate_metadata_requires_release_readme_preconditions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    contract = _load_release_contract_module()

    (tmp_path / "scripts").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        """
[build-system]
requires = [\"maturin>=1.4,<2.0\"]

[project]
name = \"pandas-booster\"
version = \"0.1.0\"
readme = \"README.md\"
requires-python = \">=3.9\"
classifiers = [
    \"Programming Language :: Python :: 3.9\",
    \"Programming Language :: Python :: 3.10\",
    \"Programming Language :: Python :: 3.11\",
    \"Programming Language :: Python :: 3.12\",
]

[project.urls]
Homepage = \"https://github.com/snowykr/pandas-booster\"
Repository = \"https://github.com/snowykr/pandas-booster\"
Issues = \"https://github.com/snowykr/pandas-booster/issues\"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "Cargo.toml").write_text(
        """
[package]
name = \"pandas_booster\"
version = \"0.1.0\"
repository = \"https://github.com/snowykr/pandas-booster\"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        """
# pandas-booster

https://github.com/snowykr/pandas-booster/actions/workflows/ci.yml
pip install pandas-booster
Trusted Publisher
publish.yml
""".strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(contract, "project_root", lambda: tmp_path)

    with pytest.raises(
        contract.ContractError, match="PyPI project exists.|GitHub environment `pypi`"
    ):
        contract.validate_metadata(argparse.Namespace())


def test_validate_metadata_requires_supported_python_classifier_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    contract = _load_release_contract_module()

    (tmp_path / "scripts").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        """
[build-system]
requires = [\"maturin>=1.4,<2.0\"]

[project]
name = \"pandas-booster\"
version = \"0.1.0\"
readme = \"README.md\"
requires-python = \">=3.9\"
classifiers = [
    \"Programming Language :: Python :: 3\",
    \"Programming Language :: Python :: 3.11\",
    \"Programming Language :: Python :: 3.12\",
]

[project.urls]
Homepage = \"https://github.com/snowykr/pandas-booster\"
Repository = \"https://github.com/snowykr/pandas-booster\"
Issues = \"https://github.com/snowykr/pandas-booster/issues\"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "Cargo.toml").write_text(
        """
[package]
name = \"pandas_booster\"
version = \"0.1.0\"
repository = \"https://github.com/snowykr/pandas-booster\"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        """
# pandas-booster

https://github.com/snowykr/pandas-booster/actions/workflows/ci.yml
pip install pandas-booster
PyPI project exists.
Trusted Publisher
publish.yml
GitHub environment `pypi` is configured
""".strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(contract, "project_root", lambda: tmp_path)

    with pytest.raises(contract.ContractError, match="must declare Python classifiers"):
        contract.validate_metadata(argparse.Namespace())


def test_validate_metadata_accepts_current_repo_release_readme():
    contract = _load_release_contract_module()
    assert contract.validate_metadata(argparse.Namespace()) == 0


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


def test_validate_workflow_rejects_missing_guardrail_token(tmp_path: Path):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "publish.yml"
    workflow_path.write_text("name: Publish\n", encoding="utf-8")

    with pytest.raises(contract.ContractError, match="missing required token"):
        contract.validate_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_workflow_accepts_current_publish_workflow():
    contract = _load_release_contract_module()

    assert contract.validate_workflow(argparse.Namespace(file=".github/workflows/publish.yml")) == 0


def test_validate_workflow_requires_tag_gated_publish_condition(tmp_path: Path):
    contract = _load_release_contract_module()
    required_publish_gate = (
        "startsWith(github.ref, 'refs/tags/v') && (github.event_name == 'push' || inputs.publish)"
    )
    workflow_path = tmp_path / "publish.yml"
    workflow_path.write_text(
        "\n".join(
            token for token in contract.WORKFLOW_REQUIRED_TOKENS if token != required_publish_gate
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(contract.ContractError, match="startsWith"):
        contract.validate_workflow(argparse.Namespace(file=str(workflow_path)))


def test_ci_keeps_non_tag_release_readiness_paths():
    repo_root = Path(__file__).resolve().parents[1]
    ci_text = (repo_root / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert (
        "python scripts/check_release_contract.py workflow --file .github/workflows/publish.yml"
        in ci_text
    )
    assert "name: Build Wheel Smoke" in ci_text
    assert "github.event_name == 'pull_request'" in ci_text
    assert "name: Release Matrix" in ci_text
    assert "github.ref == 'refs/heads/main'" in ci_text
    assert "github.event_name == 'workflow_dispatch'" in ci_text


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


def test_main_dispatches_tag_command(monkeypatch: pytest.MonkeyPatch):
    contract = _load_release_contract_module()

    monkeypatch.setattr(
        sys, "argv", ["check_release_contract.py", "tag", "--tag", f"v{_project_version()}"]
    )

    assert contract.main() == 0
