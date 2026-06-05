"""Release contract tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from conftest import _load_release_contract_module

from ._helpers import _current_supply_chain_workflow_text


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
