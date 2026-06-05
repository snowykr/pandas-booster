"""Median rollback-scope and missing-kernel fallback contract tests."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from ._helpers import (
    _accessor_groupby_result,
    _patch_multi_std_var_kernel,
    _patch_pandas_series_groupby_agg_to_raise,
    _patch_single_std_var_kernel,
    _proxy_groupby_result,
)


class TestMedianRollbackContracts:
    def test_single_key_median_exact_kernel_missing_falls_back_without_abi_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        import pandas_booster
        import pandas_booster._abi_compat as abi
        import pandas_booster._rust as rust

        monkeypatch.delenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", raising=False)
        monkeypatch.setattr(abi, "_WARNED_ABI_SKEW", False)

        for kernel in ("f64", "i64"):
            for suffix in ("", "_sorted", "_firstseen_u32", "_firstseen_u64"):
                monkeypatch.delattr(rust, f"groupby_median_{kernel}{suffix}", raising=False)

        def fake_multi_groupby(_key_arrays, _values_arr):
            return [np.asarray([1, 2], dtype=np.int64)], np.asarray([2.0, 5.0], dtype=np.float64)

        monkeypatch.setattr(
            rust, "groupby_multi_median_f64_sorted", fake_multi_groupby, raising=False
        )

        df = pd.DataFrame(
            {
                "key": np.tile([1, 2], 50_000),
                "val": np.linspace(0.0, 1.0, 100_000, dtype=np.float64),
            }
        )
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

    def test_float_rollback_scope_does_not_broaden_to_int_backed_median(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        import pandas_booster._rust as rust

        monkeypatch.setenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", "1")

        df = pd.DataFrame(
            {
                "key": [1, 1, 1, 2, 2, 2],
                "val": np.array([1, 5, 9, 10, 14, 20], dtype=np.int64),
            }
        )
        expected = df.groupby("key", sort=True)["val"].median()

        _patch_single_std_var_kernel(
            monkeypatch, rust, expected, "median", kernel="i64", result_dtype=np.dtype(np.float64)
        )
        _patch_pandas_series_groupby_agg_to_raise(
            monkeypatch,
            "median",
            "float rollback env must not broaden to int-backed median fallback",
        )

        accessor_result = _accessor_groupby_result(df, "key", "val", "median")
        proxy_result = _proxy_groupby_result(df, "key", "val", "median")

        pd.testing.assert_series_equal(accessor_result, expected, check_exact=False, rtol=1e-12)
        pd.testing.assert_series_equal(proxy_result, expected, check_exact=False, rtol=1e-12)

    def test_float_rollback_scope_does_not_broaden_to_multi_key_median(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        import pandas_booster._rust as rust

        monkeypatch.setenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", "1")

        df = pd.DataFrame(
            {
                "k1": [1, 1, 1, 2, 2, 2],
                "k2": [10, 10, 20, 10, 10, 20],
                "val": [1.0, 5.0, 9.0, 2.0, 6.0, 10.0],
            }
        )
        expected = df.groupby(["k1", "k2"], sort=True)["val"].median()

        _patch_multi_std_var_kernel(
            monkeypatch, rust, expected, "median", kernel="f64", result_dtype=np.dtype(np.float64)
        )
        _patch_pandas_series_groupby_agg_to_raise(
            monkeypatch,
            "median",
            "float rollback env must not broaden to multi-key median fallback",
        )

        accessor_result = _accessor_groupby_result(df, ["k1", "k2"], "val", "median")
        proxy_result = _proxy_groupby_result(df, ["k1", "k2"], "val", "median")

        pd.testing.assert_series_equal(accessor_result, expected, check_exact=False, rtol=1e-12)
        pd.testing.assert_series_equal(proxy_result, expected, check_exact=False, rtol=1e-12)
