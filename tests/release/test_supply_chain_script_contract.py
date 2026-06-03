"""Release contract tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from conftest import _load_release_contract_module

from ._helpers import _current_supply_chain_workflow_text


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
            '  python scripts/supply_chain_audit.py scan --base "$BASE_SHA"',
            '  python scripts/supply_chain_comment.py sync --pr-number "$PR_NUMBER"',
        )
    )
    workflow_path.write_text(workflow_text + "\n", encoding="utf-8")

    with pytest.raises(contract.ContractError, match="PR-controlled supply-chain scripts"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


@pytest.mark.parametrize(
    "run_line",
    [
        'python ./scripts/supply_chain_audit.py scan --base "$BASE_SHA"',
        'python3 scripts/supply_chain_audit.py scan --base "$BASE_SHA"',
        'python -u scripts/supply_chain_audit.py scan --base "$BASE_SHA"',
        'python -W ignore scripts/supply_chain_audit.py scan --base "$BASE_SHA"',
        'python -m scripts.supply_chain_audit scan --base "$BASE_SHA"',
        "python -c 'import runpy; runpy.run_module(\"scripts.supply_chain_audit\")'",
        "python -c 'import scripts.supply_chain_audit'",
        "python -c 'from scripts import supply_chain_comment'",
        'python "${GITHUB_WORKSPACE}/scripts/supply_chain_audit.py" scan --base "$BASE_SHA"',
        'python "$GITHUB_WORKSPACE/scripts/supply_chain_audit.py" scan --base "$BASE_SHA"',
        'python "$GITHUB_WORKSPACE"/scripts/supply_chain_audit.py scan --base "$BASE_SHA"',
        'python "$PWD/scripts/supply_chain_audit.py" scan --base "$BASE_SHA"',
        'python "$PWD"/scripts/supply_chain_audit.py scan --base "$BASE_SHA"',
        'python "${{ github.workspace }}"/scripts/supply_chain_audit.py scan --base "$BASE_SHA"',
        'python ../pandas-booster/scripts/supply_chain_audit.py scan --base "$BASE_SHA"',
        'uv run -m scripts.supply_chain_audit scan --base "$BASE_SHA"',
        'uv run scripts/supply_chain_audit.py scan --base "$BASE_SHA"',
        'uv run python scripts/supply_chain_comment.py sync --pr-number "$PR_NUMBER"',
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
            '  # python "$AUDIT_SCRIPT" scan',
            "  python \\",
            '    scripts/supply_chain_audit.py scan --base "$BASE_SHA"',
        )
    )
    workflow_path.write_text(workflow_text + "\n", encoding="utf-8")

    with pytest.raises(contract.ContractError, match="PR-controlled supply-chain scripts"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


@pytest.mark.parametrize(
    "run_line",
    [
        'python scripts//supply_chain_audit.py scan --base "$BASE_SHA"',
        'python scripts/./supply_chain_comment.py sync --pr-number "$PR_NUMBER"',
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
        f'          {run_line}\n          python "$AUDIT_SCRIPT" scan',
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
