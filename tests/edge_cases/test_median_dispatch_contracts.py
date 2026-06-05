"""Median dispatch and threshold fallback contract tests."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from ._helpers import (
    _accessor_groupby_result,
    _patch_all_numeric_kernels_for_agg_to_raise,
    _patch_pandas_series_groupby_agg_to_raise,
    _patch_single_std_var_kernel,
    _proxy_groupby_result,
)


class TestMedianDispatchContracts:
    def test_supported_domain_median_fallback_is_warning_free_before_rust_kernels_exist(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        import pandas_booster
        import pandas_booster._abi_compat as abi
        import pandas_booster._rust as rust

        monkeypatch.delenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", raising=False)
        monkeypatch.setattr(abi, "_WARNED_ABI_SKEW", False)
        for kernel in ("f64", "i64"):
            for prefix in ("groupby", "groupby_multi"):
                for suffix in ("", "_sorted", "_firstseen_u32", "_firstseen_u64"):
                    monkeypatch.delattr(rust, f"{prefix}_median_{kernel}{suffix}", raising=False)

        df = pd.DataFrame({"key": [1, 1, 2, 2], "val": [1.5, 4.5, 10.0, 14.0]})
        expected = df.groupby("key", sort=True)["val"].median()

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            accessor_result = _accessor_groupby_result(df, "key", "val", "median")

            pandas_booster.activate()
            try:
                proxy_result = df.groupby("key", sort=True)["val"].median()
            finally:
                pandas_booster.deactivate()

        assert rec == []
        pd.testing.assert_series_equal(accessor_result, expected, check_exact=False, rtol=1e-12)
        pd.testing.assert_series_equal(proxy_result, expected, check_exact=False, rtol=1e-12)

    def test_small_single_key_float_median_uses_rust_without_threshold_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        import pandas_booster._rust as rust

        monkeypatch.delenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", raising=False)

        df = pd.DataFrame(
            {
                "key": [1, 1, 1, 2, 2, 2],
                "val": [1.5, 4.5, 8.5, 10.0, 14.0, 20.0],
            }
        )
        expected = df.groupby("key", sort=True)["val"].median()

        _patch_single_std_var_kernel(
            monkeypatch, rust, expected, "median", kernel="f64", result_dtype=np.dtype(np.float64)
        )
        _patch_pandas_series_groupby_agg_to_raise(
            monkeypatch,
            "median",
            "supported small single-key float median should not use threshold fallback",
        )

        accessor_result = _accessor_groupby_result(df, "key", "val", "median")
        proxy_result = _proxy_groupby_result(df, "key", "val", "median")

        assert accessor_result.dtype == np.dtype(np.float64)
        assert proxy_result.dtype == np.dtype(np.float64)
        pd.testing.assert_series_equal(accessor_result, expected, check_exact=False, rtol=1e-12)
        pd.testing.assert_series_equal(proxy_result, expected, check_exact=False, rtol=1e-12)

    def test_integer_backed_median_without_fallback_uses_float64_materialization_path(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        import pandas_booster._rust as rust

        monkeypatch.delenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", raising=False)

        df = pd.DataFrame(
            {
                "key": [1, 1, 1, 2, 2, 2],
                "val": np.array([1, 5, 9, 10, 14, 20], dtype=np.int64),
            }
        )
        expected = df.groupby("key", sort=True)["val"].median()

        _patch_single_std_var_kernel(
            monkeypatch, rust, expected, "median", kernel="i64", result_dtype=np.dtype(np.int64)
        )
        _patch_pandas_series_groupby_agg_to_raise(
            monkeypatch,
            "median",
            "integer-backed median should dispatch but materialize as float64",
        )

        accessor_result = _accessor_groupby_result(df, "key", "val", "median")
        proxy_result = _proxy_groupby_result(df, "key", "val", "median")

        assert expected.dtype == np.dtype(np.float64)
        assert accessor_result.dtype == np.dtype(np.float64)
        assert proxy_result.dtype == np.dtype(np.float64)
        pd.testing.assert_series_equal(accessor_result, expected, check_exact=False, rtol=1e-12)
        pd.testing.assert_series_equal(proxy_result, expected, check_exact=False, rtol=1e-12)

    def test_single_key_float_median_rollback_env_forces_pandas_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        import pandas_booster._rust as rust

        monkeypatch.setenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", "1")

        df = pd.DataFrame(
            {
                "key": [1, 1, 1, 2, 2, 2],
                "val": [1.5, 4.5, 8.5, 10.0, 14.0, 20.0],
            }
        )
        expected = df.groupby("key", sort=True)["val"].median()

        _patch_all_numeric_kernels_for_agg_to_raise(
            monkeypatch,
            rust,
            "median",
            "single-key float median rollback should force pandas fallback",
        )

        accessor_result = _accessor_groupby_result(df, "key", "val", "median")
        proxy_result = _proxy_groupby_result(df, "key", "val", "median")

        pd.testing.assert_series_equal(accessor_result, expected, check_exact=False, rtol=1e-12)
        pd.testing.assert_series_equal(proxy_result, expected, check_exact=False, rtol=1e-12)
