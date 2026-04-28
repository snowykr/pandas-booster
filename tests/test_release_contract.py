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


def _write_package_init(root: Path, version: str) -> None:
    package_dir = root / "python" / "pandas_booster"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text(f'__version__ = "{version}"\n', encoding="utf-8")


def _job_block(workflow_text: str, job_name: str) -> str:
    marker = f"  {job_name}:"
    lines = workflow_text.splitlines()

    try:
        start = lines.index(marker)
    except ValueError as exc:
        jobs_start = next((idx for idx, line in enumerate(lines) if line == "jobs:"), None)
        available_jobs = [
            line.strip().removesuffix(":")
            for line in (lines[jobs_start + 1 :] if jobs_start is not None else [])
            if line.startswith("  ") and not line.startswith("    ") and line.endswith(":")
        ]
        context = f" Available jobs: {', '.join(available_jobs)}." if available_jobs else ""
        raise AssertionError(f"Workflow is missing expected job {job_name!r}.{context}") from exc

    block = [lines[start]]

    for line in lines[start + 1 :]:
        if line.startswith("  ") and not line.startswith("    "):
            break
        block.append(line)

    return "\n".join(block)


def _job_if_expression(workflow_text: str, job_name: str) -> str:
    block_lines = _job_block(workflow_text, job_name).splitlines()

    for idx, line in enumerate(block_lines):
        if line == "    if: >":
            expr_lines: list[str] = []
            for cont in block_lines[idx + 1 :]:
                if not cont.startswith("      "):
                    break
                expr_lines.append(cont.strip())
            return " ".join(expr_lines)

        if line.startswith("    if: "):
            return line.removeprefix("    if: ").strip()

    raise AssertionError(f"Job {job_name!r} is missing an if expression")


def test_job_block_reports_missing_job_with_available_jobs() -> None:
    workflow_text = """
jobs:
  build-wheel-smoke:
    runs-on: ubuntu-latest
  test-wheel-smoke:
    runs-on: ubuntu-latest
""".strip()

    with pytest.raises(AssertionError) as exc_info:
        _job_block(workflow_text, "build-and-test-quick")

    message = str(exc_info.value)
    assert "build-and-test-quick" in message
    assert "build-wheel-smoke" in message
    assert "test-wheel-smoke" in message


def test_job_block_available_jobs_only_lists_job_entries() -> None:
    workflow_text = """
name: CI
on:
  push:
  workflow_dispatch:
jobs:
  build-wheel-smoke:
    runs-on: ubuntu-latest
  stress-tests:
    runs-on: ubuntu-latest
permissions:
  contents: read
""".strip()

    with pytest.raises(AssertionError) as exc_info:
        _job_block(workflow_text, "missing-job")

    message = str(exc_info.value)
    assert "build-wheel-smoke" in message
    assert "stress-tests" in message
    assert "workflow_dispatch" not in message
    assert "permissions" not in message


def test_validate_metadata_requires_release_readme_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    contract = _load_release_contract_module()

    (tmp_path / "scripts").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        """
[build-system]
requires = [\"maturin>=1.13,<2.0\"]

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
    _write_package_init(tmp_path, "0.1.0")

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
requires = [\"maturin>=1.13,<2.0\"]

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
    _write_package_init(tmp_path, "0.1.0")

    monkeypatch.setattr(contract, "project_root", lambda: tmp_path)

    with pytest.raises(contract.ContractError) as exc_info:
        contract.validate_metadata(argparse.Namespace())

    message = str(exc_info.value)
    assert "README.md must contain 'PyPI project exists.'" in message
    assert "README.md must contain 'GitHub environment `pypi` is configured'" in message


def test_validate_metadata_requires_supported_python_classifier_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    contract = _load_release_contract_module()

    (tmp_path / "scripts").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        """
[build-system]
requires = [\"maturin>=1.13,<2.0\"]

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
    _write_package_init(tmp_path, "0.1.0")

    monkeypatch.setattr(contract, "project_root", lambda: tmp_path)

    with pytest.raises(contract.ContractError, match="must declare Python classifiers"):
        contract.validate_metadata(argparse.Namespace())


def test_validate_metadata_accepts_current_repo_release_readme():
    contract = _load_release_contract_module()
    assert contract.validate_metadata(argparse.Namespace()) == 0


def test_validate_metadata_requires_package_dunder_version_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    contract = _load_release_contract_module()

    (tmp_path / "scripts").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        """
[build-system]
requires = [\"maturin>=1.13,<2.0\"]

[project]
name = \"pandas-booster\"
version = \"0.1.2\"
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
version = \"0.1.2\"
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
    _write_package_init(tmp_path, "0.1.1")

    monkeypatch.setattr(contract, "project_root", lambda: tmp_path)

    with pytest.raises(contract.ContractError, match="__version__ must exactly match"):
        contract.validate_metadata(argparse.Namespace())


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
    main_pr_smoke_gate = (
        "github.event_name == 'pull_request' && github.event.pull_request.base.ref == 'main'"
    )
    non_main_pr_quick_gate = (
        "(github.event_name == 'pull_request' && github.event.pull_request.base.ref != 'main')"
    )
    stress_gate = (
        "(github.event_name == 'pull_request' && github.event.pull_request.base.ref == 'main' "
        "&& contains(github.event.pull_request.labels.*.name, 'run-stress')) || "
        "(github.event_name == 'push' && github.ref == 'refs/heads/main') || "
        "github.event_name == 'workflow_dispatch'"
    )

    assert (
        "python scripts/check_release_contract.py workflow --file .github/workflows/publish.yml"
        in ci_text
    )
    assert "ruff check python tests scripts benches" in ci_text
    assert "name: Build Wheel Smoke" in ci_text
    assert _job_if_expression(ci_text, "build-wheel-smoke") == main_pr_smoke_gate
    assert _job_if_expression(ci_text, "test-wheel-smoke") == main_pr_smoke_gate
    assert _job_if_expression(ci_text, "build-and-test-quick") == non_main_pr_quick_gate
    assert _job_if_expression(ci_text, "stress-tests") == stress_gate
    assert "name: Release Matrix" in ci_text
    assert "name: Stress Tests (Determinism)" in ci_text
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
