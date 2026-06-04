"""Release contract tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from conftest import _load_release_contract_module

from ._helpers import _job_block, _job_if_expression


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
    repo_root = Path(__file__).resolve().parents[2]
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
    assert "ruff check python tests scripts benchmarks" in ci_text
    assert "name: Build Wheel Smoke" in ci_text
    assert _job_if_expression(ci_text, "build-wheel-smoke") == main_pr_smoke_gate
    assert _job_if_expression(ci_text, "test-wheel-smoke") == main_pr_smoke_gate
    test_wheel_smoke_block = _job_block(ci_text, "test-wheel-smoke")
    assert "PANDAS_BOOSTER_WHEEL_SMOKE" in test_wheel_smoke_block
    assert "uv sync --locked --extra dev --no-install-project" in test_wheel_smoke_block
    assert "uv pip install" in test_wheel_smoke_block
    assert "site-packages" in test_wheel_smoke_block
    assert "uv run --no-sync pytest" in test_wheel_smoke_block
    assert "-o pythonpath=" in test_wheel_smoke_block
    assert _job_if_expression(ci_text, "build-and-test-quick") == non_main_pr_quick_gate
    assert _job_if_expression(ci_text, "stress-tests") == stress_gate
    assert "name: Release Matrix" in ci_text
    assert "name: Stress Tests (Determinism)" in ci_text
    assert "github.ref == 'refs/heads/main'" in ci_text
    assert "github.event_name == 'workflow_dispatch'" in ci_text
