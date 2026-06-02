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


def test_validate_supply_chain_workflow_rejects_paths_filter(tmp_path: Path):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = "\n".join(contract.SUPPLY_CHAIN_WORKFLOW_REQUIRED_TOKENS)
    workflow_path.write_text(f"{workflow_text}\npaths:\n", encoding="utf-8")

    with pytest.raises(contract.ContractError, match="must not use workflow-level paths"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_supply_chain_workflow_requires_comment_lifecycle_tokens(tmp_path: Path):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    lifecycle_token = "pandas-booster:supply-chain-risk-guard"
    workflow_text = "\n".join(
        token
        for token in contract.SUPPLY_CHAIN_WORKFLOW_REQUIRED_TOKENS
        if token != lifecycle_token
    )
    workflow_path.write_text(workflow_text + "\n", encoding="utf-8")

    with pytest.raises(contract.ContractError, match="pandas-booster:supply-chain-risk-guard"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_supply_chain_workflow_rejects_inline_shell_scanner(tmp_path: Path):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = "\n".join(contract.SUPPLY_CHAIN_WORKFLOW_REQUIRED_TOKENS)
    legacy_script = "supply_chain" + "_audit.sh"
    workflow_path.write_text(
        f'{workflow_text}\ncat > "$RUNNER_TEMP/{legacy_script}"\n',
        encoding="utf-8",
    )

    with pytest.raises(contract.ContractError, match=legacy_script):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_supply_chain_workflow_rejects_floating_actions(tmp_path: Path):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = "\n".join(contract.SUPPLY_CHAIN_WORKFLOW_REQUIRED_TOKENS)
    workflow_text = workflow_text.replace(
        "actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5",
        "actions/checkout@v4",
    )
    workflow_path.write_text(workflow_text + "\n", encoding="utf-8")

    with pytest.raises(contract.ContractError, match="actions/checkout@v4"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_supply_chain_workflow_rejects_pr_controlled_script_fallback(
    tmp_path: Path,
):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = "\n".join(contract.SUPPLY_CHAIN_WORKFLOW_REQUIRED_TOKENS)
    workflow_path.write_text(
        f"{workflow_text}\ncp scripts/supply_chain_audit.py\n",
        encoding="utf-8",
    )

    with pytest.raises(contract.ContractError, match="cp scripts/supply_chain_audit.py"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_supply_chain_workflow_rejects_direct_pr_script_execution(
    tmp_path: Path,
):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = "\n".join(
            (
                *contract.SUPPLY_CHAIN_WORKFLOW_REQUIRED_TOKENS,
                "concurrency:",
                "  group: supply-chain-risk-guard-${{ github.event.pull_request.number }}",
                "  cancel-in-progress: true",
                "run: |",
                "  python scripts/supply_chain_audit.py scan --base \"$BASE_SHA\"",
                "  python scripts/supply_chain_comment.py sync --pr-number \"$PR_NUMBER\"",
        )
    )
    workflow_path.write_text(workflow_text + "\n", encoding="utf-8")

    with pytest.raises(contract.ContractError, match="PR-controlled supply-chain scripts"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


@pytest.mark.parametrize(
    "run_line",
    [
        "python ./scripts/supply_chain_audit.py scan --base \"$BASE_SHA\"",
        "python3 scripts/supply_chain_audit.py scan --base \"$BASE_SHA\"",
        "python -u scripts/supply_chain_audit.py scan --base \"$BASE_SHA\"",
        "python -W ignore scripts/supply_chain_audit.py scan --base \"$BASE_SHA\"",
        "python -m scripts.supply_chain_audit scan --base \"$BASE_SHA\"",
        "python -c 'import runpy; runpy.run_module(\"scripts.supply_chain_audit\")'",
        "python -c 'import scripts.supply_chain_audit'",
        "python -c 'from scripts import supply_chain_comment'",
        "python \"${GITHUB_WORKSPACE}/scripts/supply_chain_audit.py\" scan --base \"$BASE_SHA\"",
        "python \"$GITHUB_WORKSPACE/scripts/supply_chain_audit.py\" scan --base \"$BASE_SHA\"",
        "python \"$GITHUB_WORKSPACE\"/scripts/supply_chain_audit.py scan --base \"$BASE_SHA\"",
        "python \"$PWD/scripts/supply_chain_audit.py\" scan --base \"$BASE_SHA\"",
        "python \"$PWD\"/scripts/supply_chain_audit.py scan --base \"$BASE_SHA\"",
        "python \"${{ github.workspace }}\"/scripts/supply_chain_audit.py "
        "scan --base \"$BASE_SHA\"",
        "python ../pandas-booster/scripts/supply_chain_audit.py scan --base \"$BASE_SHA\"",
        "uv run -m scripts.supply_chain_audit scan --base \"$BASE_SHA\"",
        "uv run scripts/supply_chain_audit.py scan --base \"$BASE_SHA\"",
        "uv run python scripts/supply_chain_comment.py sync --pr-number \"$PR_NUMBER\"",
    ],
)
def test_validate_supply_chain_workflow_rejects_direct_pr_script_execution_variants(
    tmp_path: Path,
    run_line: str,
):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = "\n".join(
        (
            *contract.SUPPLY_CHAIN_WORKFLOW_REQUIRED_TOKENS,
            "concurrency:",
            "  group: supply-chain-risk-guard-${{ github.event.pull_request.number }}",
            "  cancel-in-progress: true",
            "run: |",
            f"  {run_line}",
        )
    )
    workflow_path.write_text(workflow_text + "\n", encoding="utf-8")

    with pytest.raises(contract.ContractError, match="PR-controlled supply-chain scripts"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_supply_chain_workflow_rejects_split_line_direct_pr_script_execution(
    tmp_path: Path,
):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = "\n".join(
        (
            *contract.SUPPLY_CHAIN_WORKFLOW_REQUIRED_TOKENS,
            "concurrency:",
            "  group: supply-chain-risk-guard-${{ github.event.pull_request.number }}",
            "  cancel-in-progress: true",
            "run: |",
            "  # python \"$AUDIT_SCRIPT\" scan",
            "  python \\",
            "    scripts/supply_chain_audit.py scan --base \"$BASE_SHA\"",
        )
    )
    workflow_path.write_text(workflow_text + "\n", encoding="utf-8")

    with pytest.raises(contract.ContractError, match="PR-controlled supply-chain scripts"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


@pytest.mark.parametrize(
    "run_line",
    [
        "python scripts//supply_chain_audit.py scan --base \"$BASE_SHA\"",
        "python scripts/./supply_chain_comment.py sync --pr-number \"$PR_NUMBER\"",
    ],
)
def test_validate_supply_chain_workflow_rejects_normalized_pr_script_paths(
    tmp_path: Path,
    run_line: str,
):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = _current_supply_chain_workflow_text().replace(
        '          python "$AUDIT_SCRIPT" scan',
        f"          {run_line}\n          python \"$AUDIT_SCRIPT\" scan",
        1,
    )
    workflow_path.write_text(workflow_text, encoding="utf-8")

    with pytest.raises(contract.ContractError, match="PR-controlled supply-chain scripts"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_supply_chain_workflow_rejects_minimally_indented_run_block_pr_script(
    tmp_path: Path,
):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = _current_supply_chain_workflow_text().replace(
        "run: |\n          set -euo pipefail",
        "run: |\n         python scripts/supply_chain_audit.py scan --base a --head b --findings c",
        1,
    )
    workflow_path.write_text(workflow_text, encoding="utf-8")

    with pytest.raises(contract.ContractError, match="PR-controlled|supply-chain script"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_supply_chain_workflow_requires_concurrency(tmp_path: Path):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = "\n".join(contract.SUPPLY_CHAIN_WORKFLOW_REQUIRED_TOKENS)
    workflow_path.write_text(workflow_text + "\n", encoding="utf-8")

    with pytest.raises(contract.ContractError, match="concurrency"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_supply_chain_workflow_rejects_github_event_in_run_block(
    tmp_path: Path,
):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = "\n".join(
            (
                *contract.SUPPLY_CHAIN_WORKFLOW_REQUIRED_TOKENS,
                "concurrency:",
                "  group: supply-chain-risk-guard-${{ github.event.pull_request.number }}",
                "  cancel-in-progress: true",
                "run: |",
                "  echo '${{ github.event.pull_request.head.sha }}'",
            )
    )
    workflow_path.write_text(workflow_text + "\n", encoding="utf-8")

    with pytest.raises(contract.ContractError, match="github.event"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_supply_chain_workflow_rejects_minimally_indented_run_block_github_event(
    tmp_path: Path,
):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = _current_supply_chain_workflow_text().replace(
        "run: |\n          set -euo pipefail",
        "run: |\n         echo '${{ github.event.pull_request.head.sha }}'",
        1,
    )
    workflow_path.write_text(workflow_text, encoding="utf-8")

    with pytest.raises(contract.ContractError, match="github.event"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_supply_chain_workflow_rejects_github_event_in_inline_run(
    tmp_path: Path,
):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = "\n".join(
            (
                *contract.SUPPLY_CHAIN_WORKFLOW_REQUIRED_TOKENS,
                "concurrency:",
                "  group: supply-chain-risk-guard-${{ github.event.pull_request.number }}",
                "  cancel-in-progress: true",
                "run: echo '${{ github.event.pull_request.head.sha }}'",
            )
    )
    workflow_path.write_text(workflow_text + "\n", encoding="utf-8")

    with pytest.raises(contract.ContractError, match="github.event"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_supply_chain_workflow_rejects_extra_triggers_and_write_permissions(
    tmp_path: Path,
):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = "\n".join(
            (
                *contract.SUPPLY_CHAIN_WORKFLOW_REQUIRED_TOKENS,
                "concurrency:",
                "  group: supply-chain-risk-guard-${{ github.event.pull_request.number }}",
                "  cancel-in-progress: true",
            "push:",
            "issues: write",
            "contents: write",
            "actions: write",
            "checks: write",
            "statuses: write",
            "id-token: write",
        )
    )
    workflow_path.write_text(workflow_text + "\n", encoding="utf-8")

    with pytest.raises(contract.ContractError) as exc_info:
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))

    message = str(exc_info.value)
    assert "must not use push triggers" in message
    assert "must not request issues: write" in message
    assert "must not request contents: write" in message
    assert "must not request actions: write" in message
    assert "must not request checks: write" in message
    assert "must not request statuses: write" in message
    assert "must not request id-token: write" in message


def test_validate_supply_chain_workflow_rejects_extra_on_block_trigger(tmp_path: Path):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = _current_supply_chain_workflow_text().replace(
        "on:\n  pull_request:",
        "on:\n  pull_request:\n  issue_comment:",
        1,
    )
    workflow_path.write_text(workflow_text, encoding="utf-8")

    with pytest.raises(contract.ContractError, match="issue_comment triggers"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_supply_chain_workflow_rejects_extra_on_block_trigger_with_deep_indent(
    tmp_path: Path,
):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = _current_supply_chain_workflow_text().replace(
        "on:\n  pull_request:\n    types: [opened, synchronize, reopened]",
        "on:\n    pull_request:\n      types: [opened, synchronize, reopened]\n"
        "    pull_request_review:",
        1,
    )
    workflow_path.write_text(workflow_text, encoding="utf-8")

    with pytest.raises(contract.ContractError, match="pull_request_review triggers"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_supply_chain_workflow_rejects_inline_on_event_list(tmp_path: Path):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = _current_supply_chain_workflow_text().replace(
        "on:\n  pull_request:\n    types: [opened, synchronize, reopened]",
        "on: [pull_request, push]\n"
        "# pull_request:\n"
        "#   types: [opened, synchronize, reopened]",
        1,
    )
    workflow_path.write_text(workflow_text, encoding="utf-8")

    with pytest.raises(contract.ContractError, match="pull_request-only on block"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_supply_chain_workflow_ignores_nested_fake_on_block(tmp_path: Path):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = _current_supply_chain_workflow_text().replace(
        "on:\n  pull_request:\n    types: [opened, synchronize, reopened]",
        "on: [pull_request, push]\n"
        "# pull_request:\n"
        "#   types: [opened, synchronize, reopened]",
        1,
    ).replace(
        "          set -euo pipefail",
        "          on:\n"
        "            pull_request:\n"
        "              types: [opened, synchronize, reopened]\n"
        "          set -euo pipefail",
        1,
    )
    workflow_path.write_text(workflow_text, encoding="utf-8")

    with pytest.raises(contract.ContractError, match="pull_request-only on block"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_supply_chain_workflow_requires_structural_pull_request_event(
    tmp_path: Path,
):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = _current_supply_chain_workflow_text().replace(
        "on:\n  pull_request:\n    types: [opened, synchronize, reopened]",
        "on:\n# pull_request:\n# types: [opened, synchronize, reopened]",
        1,
    )
    workflow_path.write_text(workflow_text, encoding="utf-8")

    with pytest.raises(contract.ContractError, match="only pull_request"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_supply_chain_workflow_requires_structural_pull_request_types(
    tmp_path: Path,
):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = _current_supply_chain_workflow_text().replace(
        "    types: [opened, synchronize, reopened]",
        "    types: [closed]\n# types: [opened, synchronize, reopened]",
        1,
    )
    workflow_path.write_text(workflow_text, encoding="utf-8")

    with pytest.raises(contract.ContractError, match="opened, synchronize, reopened"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_supply_chain_workflow_rejects_extra_read_permission(tmp_path: Path):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = _current_supply_chain_workflow_text().replace(
        "permissions:\n  pull-requests: write\n  contents: read",
        "permissions:\n  pull-requests: write\n  contents: read\n  checks: read",
        1,
    )
    workflow_path.write_text(workflow_text, encoding="utf-8")

    with pytest.raises(contract.ContractError, match="must not request checks: read"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


@pytest.mark.parametrize(
    "extra_permission",
    [
        "  checks: read # extra permission",
        "  actions: write # extra permission",
        "  checks : read",
        "  actions : write",
    ],
)
def test_validate_supply_chain_workflow_rejects_extra_permission_with_trailing_comment(
    tmp_path: Path,
    extra_permission: str,
):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = _current_supply_chain_workflow_text().replace(
        "permissions:\n  pull-requests: write\n  contents: read",
        "permissions:\n  pull-requests: write\n  contents: read\n" + extra_permission,
        1,
    )
    workflow_path.write_text(workflow_text, encoding="utf-8")

    with pytest.raises(contract.ContractError, match="must not request"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_supply_chain_workflow_rejects_duplicate_permissions_blocks(
    tmp_path: Path,
):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = _current_supply_chain_workflow_text() + "\n".join(
        [
            "",
            "permissions:",
            "  contents: read",
            "  pull-requests: write",
            "  checks: read",
        ]
    )
    workflow_path.write_text(workflow_text, encoding="utf-8")

    with pytest.raises(contract.ContractError) as exc_info:
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))

    message = str(exc_info.value)
    assert "exactly one top-level permissions block" in message
    assert "must not request checks: read" in message


def test_validate_supply_chain_workflow_requires_structural_pull_requests_permission(
    tmp_path: Path,
):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = _current_supply_chain_workflow_text().replace(
        "  pull-requests: write",
        "  # pull-requests: write",
        1,
    )
    workflow_path.write_text(workflow_text, encoding="utf-8")

    with pytest.raises(contract.ContractError, match="must request pull-requests: write"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_supply_chain_workflow_rejects_extra_read_permission_with_deep_indent(
    tmp_path: Path,
):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = _current_supply_chain_workflow_text().replace(
        "permissions:\n  pull-requests: write\n  contents: read",
        "permissions:\n    pull-requests: write\n    contents: read\n    checks: read",
        1,
    )
    workflow_path.write_text(workflow_text, encoding="utf-8")

    with pytest.raises(contract.ContractError, match="must not request checks: read"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_supply_chain_workflow_rejects_job_write_all_permission(tmp_path: Path):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = _current_supply_chain_workflow_text().replace(
        "    runs-on: ubuntu-latest",
        "    runs-on: ubuntu-latest\n    permissions: write-all",
        1,
    )
    workflow_path.write_text(workflow_text, encoding="utf-8")

    with pytest.raises(contract.ContractError, match="inline or job-level permissions"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_supply_chain_workflow_rejects_inline_permission_map(tmp_path: Path):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = _current_supply_chain_workflow_text().replace(
        "permissions:\n  pull-requests: write\n  contents: read",
        "permissions: {contents: read, pull-requests: write, actions: write}",
        1,
    )
    workflow_path.write_text(workflow_text, encoding="utf-8")

    with pytest.raises(contract.ContractError, match="inline or job-level permissions"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_supply_chain_workflow_rejects_pull_request_paths_ignore(
    tmp_path: Path,
):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = _current_supply_chain_workflow_text().replace(
        "    types: [opened, synchronize, reopened]",
        "    types: [opened, synchronize, reopened]\n    paths-ignore:\n      - README.md",
        1,
    )
    workflow_path.write_text(workflow_text, encoding="utf-8")

    with pytest.raises(contract.ContractError, match="workflow-level paths filters"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_supply_chain_workflow_rejects_audit_script_env_override(
    tmp_path: Path,
):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = _current_supply_chain_workflow_text().replace(
        "        env:\n          BASE_SHA: ${{ github.event.pull_request.base.sha }}\n"
        "          HEAD_SHA: ${{ github.event.pull_request.head.sha }}",
        "        env:\n          BASE_SHA: ${{ github.event.pull_request.base.sha }}\n"
        "          HEAD_SHA: ${{ github.event.pull_request.head.sha }}\n"
        "          AUDIT_SCRIPT: scripts/supply_chain_audit.py",
        1,
    )
    workflow_path.write_text(workflow_text, encoding="utf-8")

    with pytest.raises(contract.ContractError, match="AUDIT_SCRIPT"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_supply_chain_workflow_rejects_any_audit_script_env_override(
    tmp_path: Path,
):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = _current_supply_chain_workflow_text().replace(
        "        env:\n          BASE_SHA: ${{ github.event.pull_request.base.sha }}\n"
        "          HEAD_SHA: ${{ github.event.pull_request.head.sha }}",
        "        env:\n          BASE_SHA: ${{ github.event.pull_request.base.sha }}\n"
        "          HEAD_SHA: ${{ github.event.pull_request.head.sha }}\n"
        "          AUDIT_SCRIPT: $GITHUB_WORKSPACE/audit.py",
        1,
    )
    workflow_path.write_text(workflow_text, encoding="utf-8")

    with pytest.raises(contract.ContractError, match="AUDIT_SCRIPT"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_supply_chain_workflow_rejects_comment_script_env_override(
    tmp_path: Path,
):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = _current_supply_chain_workflow_text().replace(
        "          COMMENT_MARKER: \"<!-- pandas-booster:supply-chain-risk-guard -->\"",
        "          COMMENT_MARKER: \"<!-- pandas-booster:supply-chain-risk-guard -->\"\n"
        "          COMMENT_SCRIPT: scripts/supply_chain_comment.py",
        1,
    )
    workflow_path.write_text(workflow_text, encoding="utf-8")

    with pytest.raises(contract.ContractError, match="COMMENT_SCRIPT"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_supply_chain_workflow_rejects_any_comment_script_env_override(
    tmp_path: Path,
):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = _current_supply_chain_workflow_text().replace(
        "          COMMENT_MARKER: \"<!-- pandas-booster:supply-chain-risk-guard -->\"",
        "          COMMENT_MARKER: \"<!-- pandas-booster:supply-chain-risk-guard -->\"\n"
        "          COMMENT_SCRIPT: $GITHUB_WORKSPACE/comment.py",
        1,
    )
    workflow_path.write_text(workflow_text, encoding="utf-8")

    with pytest.raises(contract.ContractError, match="COMMENT_SCRIPT"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_supply_chain_workflow_rejects_inline_audit_script_env_override(
    tmp_path: Path,
):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = _current_supply_chain_workflow_text().replace(
        "        env:\n          BASE_SHA: ${{ github.event.pull_request.base.sha }}\n"
        "          HEAD_SHA: ${{ github.event.pull_request.head.sha }}",
        "        env: {BASE_SHA: x, HEAD_SHA: y, "
        "AUDIT_SCRIPT: scripts/supply_chain_audit.py}",
        1,
    )
    workflow_path.write_text(workflow_text, encoding="utf-8")

    with pytest.raises(contract.ContractError, match="AUDIT_SCRIPT"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_supply_chain_workflow_rejects_inline_comment_script_env_override(
    tmp_path: Path,
):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = _current_supply_chain_workflow_text().replace(
        "        env:\n"
        "          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}\n"
        "          PR_NUMBER: ${{ github.event.pull_request.number }}\n"
        "          HEAD_SHA: ${{ github.event.pull_request.head.sha }}\n"
        "          SCAN_FOUND: ${{ steps.scan.outputs.found }}\n"
        "          COMMENT_MARKER: \"<!-- pandas-booster:supply-chain-risk-guard -->\"",
        "        env: {GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}, "
        "PR_NUMBER: ${{ github.event.pull_request.number }}, "
        "HEAD_SHA: ${{ github.event.pull_request.head.sha }}, "
        "SCAN_FOUND: ${{ steps.scan.outputs.found }}, "
        "COMMENT_MARKER: \"<!-- pandas-booster:supply-chain-risk-guard -->\", "
        "COMMENT_SCRIPT: $GITHUB_WORKSPACE/comment.py}",
        1,
    )
    workflow_path.write_text(workflow_text, encoding="utf-8")

    with pytest.raises(contract.ContractError, match="COMMENT_SCRIPT"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_supply_chain_workflow_rejects_github_env_audit_script_override(
    tmp_path: Path,
):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = _current_supply_chain_workflow_text().replace(
        '          echo "AUDIT_SCRIPT=$RUNNER_TEMP/supply-chain/audit.py" >> "$GITHUB_ENV"',
        '          echo "AUDIT_SCRIPT=scripts/supply_chain_audit.py" >> "$GITHUB_ENV"',
        1,
    )
    workflow_path.write_text(workflow_text, encoding="utf-8")

    with pytest.raises(contract.ContractError, match="AUDIT_SCRIPT"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_supply_chain_workflow_rejects_github_env_comment_script_override(
    tmp_path: Path,
):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = _current_supply_chain_workflow_text().replace(
        '          echo "COMMENT_SCRIPT=$RUNNER_TEMP/supply-chain/comment.py" >> "$GITHUB_ENV"',
        '          echo "COMMENT_SCRIPT=scripts/supply_chain_comment.py" >> "$GITHUB_ENV"',
        1,
    )
    workflow_path.write_text(workflow_text, encoding="utf-8")

    with pytest.raises(contract.ContractError, match="COMMENT_SCRIPT"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_supply_chain_workflow_accepts_current_workflow():
    contract = _load_release_contract_module()

    assert (
        contract.validate_supply_chain_workflow(
            argparse.Namespace(file=".github/workflows/supply-chain-audit.yml")
        )
        == 0
    )


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
    assert "ruff check python tests scripts benchmarks" in ci_text
    assert "name: Build Wheel Smoke" in ci_text
    assert _job_if_expression(ci_text, "build-wheel-smoke") == main_pr_smoke_gate
    assert _job_if_expression(ci_text, "test-wheel-smoke") == main_pr_smoke_gate
    assert _job_if_expression(ci_text, "build-and-test-quick") == non_main_pr_quick_gate
    assert _job_if_expression(ci_text, "stress-tests") == stress_gate
    assert "name: Release Matrix" in ci_text
    assert "name: Stress Tests (Determinism)" in ci_text
    assert "github.ref == 'refs/heads/main'" in ci_text
    assert "github.event_name == 'workflow_dispatch'" in ci_text


def _current_supply_chain_workflow_text() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / ".github" / "workflows" / "supply-chain-audit.yml").read_text(
        encoding="utf-8"
    )


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
