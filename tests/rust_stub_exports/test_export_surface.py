import numpy as np
import pandas as pd
import pytest

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
        "profile_groupby_multi_count_f64_sorted",
        "profile_groupby_multi_count_i64_sorted",
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
        f"missing multi-key sorted profile stub contract: {sorted(missing_stubbed)}"
    )


@pytest.mark.parametrize("value_kind", ["i64", "f64", "all_nan"])
def test_live_multi_key_sorted_profile_matches_normal_export_and_pandas(value_kind):
    import pandas_booster._rust as rust

    limits = np.iinfo(np.int64)
    key_seeds = [
        np.array([limits.max, limits.max, limits.min, limits.min, -1, -1, 0, 0], dtype=np.int64),
        np.array([-7, -7, 8, 8, -9, -9, 0, 0], dtype=np.int64),
        np.array([2, 2, -3, -3, 4, 4, -5, -5], dtype=np.int64),
    ]
    value_seed = (
        np.array([limits.min, limits.max, -7, 9, -1, 0, 3, 4], dtype=np.int64)
        if value_kind == "i64"
        else np.array([np.nan, 2.0, -0.0, 0.0, 0.0, -0.0, np.nan, np.nan])
    )
    if value_kind == "all_nan":
        value_seed[:] = np.nan
    n_rows = max(100_001, int(rust.get_fallback_threshold()) + 1)
    repeats = (n_rows + len(value_seed) - 1) // len(value_seed)
    keys = [np.tile(seed, repeats) for seed in key_seeds]
    values = np.tile(value_seed, repeats)
    kernel = "i64" if value_kind == "i64" else "f64"
    symbol = f"groupby_multi_max_{kernel}_sorted"
    normal_keys, normal_values = getattr(rust, symbol)(keys, values)
    profile_keys, profile_values, profile = getattr(rust, f"profile_{symbol}")(keys, values)

    names = ["signed_extreme", "secondary", "tertiary"]
    df = pd.DataFrame(dict(zip(names, keys)))
    df["val"] = values
    expected = df.groupby(names, sort=True)["val"].max()
    for result_keys, result_values in [
        (normal_keys, normal_values),
        (profile_keys, profile_values),
    ]:
        assert len(result_keys) == len(names)
        assert all(column.dtype == np.dtype("int64") for column in result_keys)
        result = pd.Series(
            result_values,
            index=pd.MultiIndex.from_arrays(result_keys, names=names),
            name="val",
        )
        pd.testing.assert_series_equal(result, expected, check_exact=True)
        pd.testing.assert_index_equal(result.index, expected.index, exact=True)
        assert result.dtype == expected.dtype
        if kernel == "f64":
            zeros = expected.to_numpy() == 0
            np.testing.assert_array_equal(
                np.signbit(result_values[zeros]), np.signbit(expected.to_numpy()[zeros])
            )
    for normal_column, profile_column in zip(normal_keys, profile_keys):
        np.testing.assert_array_equal(normal_column, profile_column)
    np.testing.assert_array_equal(normal_values, profile_values)
    assert normal_values.dtype == profile_values.dtype
    assert profile["profile_kind"] == "multi_key_sorted"
    assert profile["final_group_count"] == len(expected)


@pytest.mark.parametrize("kernel", ["f64", "i64"])
@pytest.mark.parametrize(
    "invalid", ["short_key", "short_values", "no_keys", "too_many_keys", "below_threshold"]
)
def test_live_multi_key_sorted_profile_and_normal_reject_invalid_inputs(kernel, invalid):
    import pandas_booster._rust as rust

    n_rows = max(100_001, int(rust.get_fallback_threshold()) + 1)
    key = np.zeros(n_rows, dtype=np.int64)
    keys = [key, key]
    values = np.zeros(n_rows, dtype=np.float64 if kernel == "f64" else np.int64)
    if invalid == "short_key":
        keys = [key, key[:-1]]
        message = "Key column 1 length .* != values length"
    elif invalid == "short_values":
        values = values[:-1]
        message = "Key column 0 length .* != values length"
    elif invalid == "no_keys":
        keys = []
        message = "At least one key column is required"
    elif invalid == "too_many_keys":
        keys = [key] * 11
        message = "Too many key columns"
    else:
        n_small = int(rust.get_fallback_threshold()) - 1
        keys = [key[:n_small], key[:n_small]]
        values = values[:n_small]
        message = "Dataset too small for acceleration"
    symbol = f"groupby_multi_max_{kernel}_sorted"
    for export in [symbol, f"profile_{symbol}"]:
        with pytest.raises(ValueError, match=message):
            getattr(rust, export)(keys, values)


def test_prod_symbols_are_registered_and_stubbed():
    _assert_expected_groupby_exports_are_registered_and_stubbed(("prod",))


def test_median_symbols_are_registered_and_stubbed():
    _assert_expected_groupby_exports_are_registered_and_stubbed(("median",))


def test_profile_exports_include_only_selected_median_surface():
    expected_exports = _expected_exports()

    assert "profile_groupby_median_f64_sorted" in expected_exports
    assert "profile_groupby_median_f64_firstseen_u32" not in expected_exports
    assert "profile_groupby_median_f64_firstseen_u64" not in expected_exports
    assert "profile_groupby_median_i64_sorted" not in expected_exports
