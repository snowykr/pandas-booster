"""Release contract tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from conftest import _load_release_contract_module

from ._helpers import _current_supply_chain_workflow_text


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
        "on: [pull_request, push]\n# pull_request:\n#   types: [opened, synchronize, reopened]",
        1,
    )
    workflow_path.write_text(workflow_text, encoding="utf-8")

    with pytest.raises(contract.ContractError, match="pull_request-only on block"):
        contract.validate_supply_chain_workflow(argparse.Namespace(file=str(workflow_path)))


def test_validate_supply_chain_workflow_ignores_nested_fake_on_block(tmp_path: Path):
    contract = _load_release_contract_module()
    workflow_path = tmp_path / "supply-chain-audit.yml"
    workflow_text = (
        _current_supply_chain_workflow_text()
        .replace(
            "on:\n  pull_request:\n    types: [opened, synchronize, reopened]",
            "on: [pull_request, push]\n# pull_request:\n#   types: [opened, synchronize, reopened]",
            1,
        )
        .replace(
            "          set -euo pipefail",
            "          on:\n"
            "            pull_request:\n"
            "              types: [opened, synchronize, reopened]\n"
            "          set -euo pipefail",
            1,
        )
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
