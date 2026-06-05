from __future__ import annotations

from typing import Final

import pytest

from ._helpers import (
    _expected_exports,
    _expected_support_exports,
    _missing_exports,
    _registered_exports,
    _registered_exports_from_source,
    _stubbed_exports,
)

_SUPPORT_EXPORT_REGISTRATION: Final = (
    "add_pyfunctions!(m, get_fallback_threshold, get_thread_count, "
    "has_ordered_single_key_float_prod_abi);"
)


def _assert_export_contract(expected: set[str], registered: set[str], stubbed: set[str]) -> None:
    missing_registered, missing_stubbed = _missing_exports(expected, registered, stubbed)

    assert not missing_registered
    assert not missing_stubbed
    assert registered == expected
    assert stubbed == expected


def test_support_exports_remain_registered_and_stubbed() -> None:
    support_exports = _expected_support_exports()

    assert support_exports <= _registered_exports()
    assert support_exports <= _stubbed_exports()


def test_support_export_registration_snippet_matches_expected_symbols() -> None:
    support_exports = _expected_support_exports()

    assert _registered_exports_from_source(_SUPPORT_EXPORT_REGISTRATION) == support_exports


def test_export_guardrail_rejects_missing_support_registration() -> None:
    support_exports = _expected_support_exports()
    registered_exports = _registered_exports_from_source(
        "add_pyfunctions!(m, get_fallback_threshold, get_thread_count);"
    )

    with pytest.raises(AssertionError, match="has_ordered_single_key_float_prod_abi"):
        _assert_export_contract(support_exports, registered_exports, support_exports)


def test_export_guardrail_rejects_missing_support_stub() -> None:
    support_exports = _expected_support_exports()
    stubbed_exports = support_exports - {"has_ordered_single_key_float_prod_abi"}

    with pytest.raises(AssertionError, match="has_ordered_single_key_float_prod_abi"):
        _assert_export_contract(support_exports, support_exports, stubbed_exports)


def test_export_guardrail_rejects_unexpected_registered_symbol() -> None:
    expected_exports = _expected_exports()
    unexpected_symbol = "groupby_sum_f32"

    with pytest.raises(AssertionError):
        _assert_export_contract(
            expected_exports,
            expected_exports | {unexpected_symbol},
            expected_exports,
        )
