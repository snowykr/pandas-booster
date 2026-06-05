"""Release contract tests."""

from __future__ import annotations

import argparse

from conftest import _load_release_contract_module


def test_validate_supply_chain_workflow_accepts_current_workflow():
    contract = _load_release_contract_module()

    assert (
        contract.validate_supply_chain_workflow(
            argparse.Namespace(file=".github/workflows/supply-chain-audit.yml")
        )
        == 0
    )
