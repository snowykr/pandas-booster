from ._helpers import (
    _assert_expected_groupby_exports_are_registered_and_stubbed,
    _expected_exports,
    _expected_profile_exports,
    _missing_exports,
    _registered_exports,
    _registered_exports_from_source,
    _stubbed_exports,
)


def test_rust_pyi_exports_match_expected_pyo3_surface():
    expected_exports = _expected_exports()
    registered_exports = _registered_exports()
    stubbed_exports = _stubbed_exports()
    missing_registered, missing_stubbed = _missing_exports(
        expected_exports,
        registered_exports,
        stubbed_exports,
    )

    assert registered_exports, (
        "src/python_wrappers/register.rs must register exports through add_pyfunctions!(m, ...)"
    )
    assert not missing_registered
    assert not missing_stubbed
    assert registered_exports == expected_exports
    assert stubbed_exports == expected_exports


def test_export_matrix_detection_flags_missing_macro_registration():
    expected_exports = _expected_exports()
    registered_exports = _registered_exports_from_source(
        "add_pyfunctions!(m, groupby_sum_f64, groupby_sum_i64);"
    )
    stubbed_exports = expected_exports

    missing_registered, missing_stubbed = _missing_exports(
        expected_exports,
        registered_exports,
        stubbed_exports,
    )

    assert "groupby_prod_f64" in missing_registered
    assert not missing_stubbed


def test_export_matrix_detection_flags_missing_stub_entry():
    expected_exports = _expected_exports()
    registered_exports = expected_exports
    stubbed_exports = expected_exports - {"groupby_prod_f64"}

    missing_registered, missing_stubbed = _missing_exports(
        expected_exports,
        registered_exports,
        stubbed_exports,
    )

    assert not missing_registered
    assert missing_stubbed == {"groupby_prod_f64"}


def test_multi_key_sorted_profile_symbols_are_registered_and_stubbed():
    expected_profile_exports = _expected_profile_exports()
    expected_multi_key_sorted_exports = {
        "profile_groupby_multi_max_f64_sorted",
        "profile_groupby_multi_max_i64_sorted",
    }
    registered_exports = _registered_exports()
    stubbed_exports = _stubbed_exports()
    missing_registered, missing_stubbed = _missing_exports(
        expected_multi_key_sorted_exports,
        registered_exports,
        stubbed_exports,
    )

    assert expected_multi_key_sorted_exports <= expected_profile_exports
    assert not missing_registered, (
        "missing multi-key sorted profile export registration contract: "
        f"{sorted(missing_registered)}"
    )
    assert not missing_stubbed, (
        "missing multi-key sorted profile stub contract: "
        f"{sorted(missing_stubbed)}"
    )


def test_prod_symbols_are_registered_and_stubbed():
    _assert_expected_groupby_exports_are_registered_and_stubbed(("prod",))


def test_median_symbols_are_registered_and_stubbed():
    _assert_expected_groupby_exports_are_registered_and_stubbed(("median",))
