"""Median strict-dispatch and unsupported-domain fallback contract tests."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from ._helpers import (
    _accessor_groupby_result,
    _delete_groupby_kernel_symbols,
    _patch_all_numeric_kernels_for_agg_to_raise,
    _proxy_groupby_result,
)


class TestMedianFallbackContracts:
    @pytest.mark.parametrize(
        ("by", "target", "missing_symbol"),
        [
            pytest.param("key", "val", "groupby_median_f64", id="single-key"),
            pytest.param(["k1", "k2"], "val", "groupby_multi_median_f64", id="multi-key"),
        ],
    )
    def test_missing_median_symbols_still_fall_back_in_strict_abi_before_dispatch(
        self,
        monkeypatch: pytest.MonkeyPatch,
        by: str | list[str],
        target: str,
        missing_symbol: str,
    ):
        import pandas_booster
        import pandas_booster._abi_compat as abi
        import pandas_booster._rust as rust

        monkeypatch.setenv("PANDAS_BOOSTER_STRICT_ABI", "1")
        monkeypatch.delenv("PANDAS_BOOSTER_ABI_SKEW_NOTICE", raising=False)
        monkeypatch.delenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", raising=False)

        if isinstance(by, str):
            df = pd.DataFrame(
                {
                    "key": np.repeat([1, 2, 3], 4),
                    "val": np.array([1.0, 2.0, 4.0, 8.0, 2.5, 3.5, 6.5, 7.5, 5.0, 6.0, 9.0, 10.0]),
                }
            )
            _delete_groupby_kernel_symbols(monkeypatch, rust, "median", kernel="f64", multi=False)
            monkeypatch.setattr(rust, "groupby_median_i64", lambda *_args: None, raising=False)
        else:
            df = pd.DataFrame(
                {
                    "k1": [1, 1, 1, 2, 2, 2],
                    "k2": [10, 10, 20, 10, 10, 20],
                    "val": [1.0, 5.0, 9.0, 2.0, 6.0, 10.0],
                }
            )
            _delete_groupby_kernel_symbols(monkeypatch, rust, "median", kernel="f64", multi=True)
            monkeypatch.setattr(
                rust, "groupby_multi_median_i64", lambda *_args: None, raising=False
            )

        monkeypatch.setattr(abi, "_WARNED_ABI_SKEW", False)

        expected = df.groupby(by, sort=True)[target].median()

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            accessor_result = _accessor_groupby_result(df, by, target, "median")

        pd.testing.assert_series_equal(accessor_result, expected, check_exact=False, rtol=1e-12)
        abi_warnings = [w for w in rec if issubclass(w.category, abi.PandasBoosterAbiSkewWarning)]
        abi_warnings = [w for w in abi_warnings if abi.ABI_SKEW_PREFIX in str(w.message)]
        assert abi_warnings == []

        abi._WARNED_ABI_SKEW = False
        pandas_booster.activate()
        try:
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                proxy_result = df.groupby(by, sort=True)[target].median()
        finally:
            pandas_booster.deactivate()

        pd.testing.assert_series_equal(proxy_result, expected, check_exact=False, rtol=1e-12)
        abi_warnings = [w for w in rec if issubclass(w.category, abi.PandasBoosterAbiSkewWarning)]
        abi_warnings = [w for w in abi_warnings if abi.ABI_SKEW_PREFIX in str(w.message)]
        assert abi_warnings == []

    @pytest.mark.parametrize(
        ("case_name", "df"),
        [
            (
                "extension_dtype_value",
                pd.DataFrame(
                    {
                        "key": np.repeat([1, 2, 3], 4),
                        "val": pd.array([1.0, 2.0, 4.0, 8.0] * 3, dtype="Float64"),
                    }
                ),
            ),
            (
                "nullable_pd_na_value",
                pd.DataFrame(
                    {
                        "key": np.repeat([1, 2, 3], 4),
                        "val": pd.array([1.0, pd.NA, 4.0, 8.0] * 3, dtype="Float64"),
                    }
                ),
            ),
            (
                "extension_dtype_key",
                pd.DataFrame(
                    {
                        "key": pd.array([1, 1, 2, 2, 3, 3], dtype="Int64"),
                        "val": [1.0, 2.0, 10.0, 14.0, 5.0, 9.0],
                    }
                ),
            ),
            (
                "non_integer_key",
                pd.DataFrame(
                    {
                        "key": ["a", "a", "b", "b", "c", "c"],
                        "val": [1.0, 2.0, 10.0, 14.0, 5.0, 9.0],
                    }
                ),
            ),
            (
                "bool_value",
                pd.DataFrame(
                    {
                        "key": [1, 1, 2, 2, 3, 3],
                        "val": [True, False, True, True, False, False],
                    }
                ),
            ),
            (
                "object_value",
                pd.DataFrame(
                    {
                        "key": [1, 1, 2, 2, 3, 3],
                        "val": np.array([1.0, 2.0, 10.0, 14.0, 5.0, 9.0], dtype=object),
                    }
                ),
            ),
            (
                "string_value",
                pd.DataFrame(
                    {
                        "key": [1, 1, 2, 2],
                        "val": ["a", "b", "c", "d"],
                    }
                ),
            ),
            (
                "category_value",
                pd.DataFrame(
                    {
                        "key": [1, 1, 2, 2],
                        "val": pd.Categorical(["a", "b", "c", "d"]),
                    }
                ),
            ),
            (
                "datetime_value",
                pd.DataFrame(
                    {
                        "key": [1, 1, 2, 2],
                        "val": pd.date_range("2026-01-01", periods=4),
                    }
                ),
            ),
            (
                "uint64_value",
                pd.DataFrame(
                    {
                        "key": [1, 1, 2, 2, 3, 3],
                        "val": np.array([0, 2**63, 4, 8, 16, 32], dtype=np.uint64),
                    }
                ),
            ),
            (
                "uint64_key",
                pd.DataFrame(
                    {
                        "key": np.array([0, 0, 2**63, 2**63], dtype=np.uint64),
                        "val": [1.0, 2.0, 10.0, 14.0],
                    }
                ),
            ),
        ],
    )
    def test_median_unsupported_domains_fall_back_for_accessor_and_proxy(
        self,
        monkeypatch: pytest.MonkeyPatch,
        case_name: str,
        df: pd.DataFrame,
    ):
        import pandas_booster._rust as rust

        _patch_all_numeric_kernels_for_agg_to_raise(
            monkeypatch,
            rust,
            "median",
            f"{case_name} should stay on pandas fallback for median",
        )

        try:
            expected = df.groupby("key", sort=True)["val"].median()
        except Exception as pandas_exc:
            with pytest.raises(type(pandas_exc)):
                _ = _accessor_groupby_result(df, "key", "val", "median")
            with pytest.raises(type(pandas_exc)):
                _ = _proxy_groupby_result(df, "key", "val", "median")
            return

        accessor_result = _accessor_groupby_result(df, "key", "val", "median")
        proxy_result = _proxy_groupby_result(df, "key", "val", "median")

        pd.testing.assert_series_equal(accessor_result, expected, check_exact=False, rtol=1e-12)
        pd.testing.assert_series_equal(proxy_result, expected, check_exact=False, rtol=1e-12)
