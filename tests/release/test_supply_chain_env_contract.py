"""Release contract tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from conftest import _load_release_contract_module

from ._helpers import _current_supply_chain_workflow_text


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
        '          COMMENT_MARKER: "<!-- pandas-booster:supply-chain-risk-guard -->"',
        '          COMMENT_MARKER: "<!-- pandas-booster:supply-chain-risk-guard -->"\n'
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
        '          COMMENT_MARKER: "<!-- pandas-booster:supply-chain-risk-guard -->"',
        '          COMMENT_MARKER: "<!-- pandas-booster:supply-chain-risk-guard -->"\n'
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
        "        env: {BASE_SHA: x, HEAD_SHA: y, AUDIT_SCRIPT: scripts/supply_chain_audit.py}",
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
        '          COMMENT_MARKER: "<!-- pandas-booster:supply-chain-risk-guard -->"',
        "        env: {GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}, "
        "PR_NUMBER: ${{ github.event.pull_request.number }}, "
        "HEAD_SHA: ${{ github.event.pull_request.head.sha }}, "
        "SCAN_FOUND: ${{ steps.scan.outputs.found }}, "
        'COMMENT_MARKER: "<!-- pandas-booster:supply-chain-risk-guard -->", '
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
