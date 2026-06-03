"""Std/var ordering and dtype normalization contract tests."""

from __future__ import annotations

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


class TestStdVarOrderingContracts:
    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_single_key_unsorted_float_std_var_preserves_first_seen_order_and_nan_semantics(
        self, agg: str
    ):
        df = pd.DataFrame(
            {
                "key": [5, 2, 5, 4, 2, 7, 8, 8, 1],
                "val": [1.0, np.nan, 3.0, np.nan, 5.0, 11.0, 2.0, 2.0, np.nan],
            }
        )

        result = _accessor_groupby_result(df, "key", "val", agg, sort=False)
        expected = getattr(df.groupby("key", sort=False)["val"], agg)()

        assert result.index.tolist() == [5, 2, 4, 7, 8, 1]
        assert result.dtype == np.dtype(np.float64)
        assert result.loc[8] == 0.0
        assert np.isnan(result.loc[2])
        assert np.isnan(result.loc[4])
        assert np.isnan(result.loc[7])
        assert np.isnan(result.loc[1])
        pd.testing.assert_series_equal(result, expected, check_exact=False, rtol=1e-12)

    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_single_key_unsorted_int_backed_std_var_preserves_first_seen_order_and_float64_dtype(
        self, agg: str
    ):
        df = pd.DataFrame(
            {
                "key": [8, 3, 8, 5, 3, 4],
                "val": np.array([2, 6, 4, 10, 14, 12], dtype=np.int64),
            }
        )

        result = _accessor_groupby_result(df, "key", "val", agg, sort=False)
        expected = getattr(df.groupby("key", sort=False)["val"], agg)()

        assert result.index.tolist() == [8, 3, 5, 4]
        assert result.dtype == np.dtype(np.float64)
        assert np.isnan(result.loc[5])
        assert np.isnan(result.loc[4])
        pd.testing.assert_series_equal(result, expected, check_exact=False, rtol=1e-12)

    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_single_key_high_cardinality_partitioned_std_var_preserves_first_seen_order(
        self, agg: str
    ):
        group_count = 10_000
        expected_order = np.concatenate(
            (
                np.arange(5_000, group_count, dtype=np.int64),
                np.arange(0, 5_000, dtype=np.int64),
            )
        )
        keys = np.repeat(expected_order, 2)
        base = np.arange(group_count, dtype=np.float64)
        values = np.empty(group_count * 2, dtype=np.float64)
        values[0::2] = base
        values[1::2] = base + 0.5

        df = pd.DataFrame({"key": keys, "val": values})

        accessor_result = _accessor_groupby_result(df, "key", "val", agg, sort=False)
        proxy_result = _proxy_groupby_result(df, "key", "val", agg, sort=False)
        expected = getattr(df.groupby("key", sort=False)["val"], agg)()

        assert accessor_result.index.tolist() == expected_order.tolist()
        assert proxy_result.index.tolist() == expected_order.tolist()
        pd.testing.assert_series_equal(accessor_result, expected, check_exact=False, rtol=1e-12)
        pd.testing.assert_series_equal(proxy_result, expected, check_exact=False, rtol=1e-12)

    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_multi_key_unsorted_std_var_preserves_existing_first_seen_order(self, agg: str):
        df = pd.DataFrame(
            {
                "k1": [2, 1, 2, 1, 2, 3, 1],
                "k2": [9, 8, 9, 7, 8, 7, 8],
                "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            }
        )

        result = _accessor_groupby_result(df, ["k1", "k2"], "val", agg, sort=False)
        expected = getattr(df.groupby(["k1", "k2"], sort=False)["val"], agg)()

        assert result.index.tolist() == [(2, 9), (1, 8), (1, 7), (2, 8), (3, 7)]
        pd.testing.assert_series_equal(result, expected, check_exact=False, rtol=1e-12)

    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_float_env_rollback_scope_does_not_broaden_to_multi_key_std_var(
        self, monkeypatch: pytest.MonkeyPatch, agg: str
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
        expected = getattr(df.groupby(["k1", "k2"], sort=True)["val"], agg)()

        _patch_multi_std_var_kernel(
            monkeypatch, rust, expected, agg, kernel="f64", result_dtype=np.float64
        )
        _patch_pandas_series_groupby_agg_to_raise(
            monkeypatch,
            agg,
            "float rollback env must not broaden to multi-key std/var pandas fallback",
        )

        accessor_result = _accessor_groupby_result(df, ["k1", "k2"], "val", agg)
        proxy_result = _proxy_groupby_result(df, ["k1", "k2"], "val", agg)

        assert accessor_result.dtype == np.dtype(np.float64)
        assert proxy_result.dtype == np.dtype(np.float64)
        pd.testing.assert_series_equal(accessor_result, expected, check_exact=False, rtol=1e-12)
        pd.testing.assert_series_equal(proxy_result, expected, check_exact=False, rtol=1e-12)

    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_float_env_rollback_scope_does_not_broaden_to_int_backed_std_var(
        self, monkeypatch: pytest.MonkeyPatch, agg: str
    ):
        import pandas_booster._rust as rust

        monkeypatch.setenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", "1")

        df = pd.DataFrame(
            {
                "key": [1, 1, 2, 2, 3, 3],
                "val": np.array([1, 3, 10, 14, 5, 9], dtype=np.int64),
            }
        )
        expected = getattr(df.groupby("key", sort=True)["val"], agg)()

        _patch_single_std_var_kernel(
            monkeypatch, rust, expected, agg, kernel="i64", result_dtype=np.float64
        )
        _patch_pandas_series_groupby_agg_to_raise(
            monkeypatch,
            agg,
            "float rollback env must not broaden to int-backed std/var pandas fallback",
        )

        accessor_result = _accessor_groupby_result(df, "key", "val", agg)
        proxy_result = _proxy_groupby_result(df, "key", "val", agg)

        assert accessor_result.dtype == np.dtype(np.float64)
        assert proxy_result.dtype == np.dtype(np.float64)
        pd.testing.assert_series_equal(accessor_result, expected, check_exact=False, rtol=1e-12)
        pd.testing.assert_series_equal(proxy_result, expected, check_exact=False, rtol=1e-12)

    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_accessor_and_proxy_normalize_float_result_abi_to_float64(
        self, monkeypatch: pytest.MonkeyPatch, agg: str
    ):
        import pandas_booster._rust as rust

        monkeypatch.delenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", raising=False)

        df = pd.DataFrame(
            {
                "key": [1, 1, 2, 2, 3, 3],
                "val": np.array([4, 8, 10, 18, 20, 28], dtype=np.int64),
            }
        )
        expected = getattr(df.groupby("key", sort=True)["val"], agg)().astype(np.float64)

        _patch_single_std_var_kernel(
            monkeypatch, rust, expected, agg, kernel="i64", result_dtype=np.float32
        )
        _patch_pandas_series_groupby_agg_to_raise(
            monkeypatch,
            agg,
            "supported std/var float-result ABI should not fall back to pandas",
        )

        accessor_result = _accessor_groupby_result(df, "key", "val", agg)
        proxy_result = _proxy_groupby_result(df, "key", "val", agg)

        assert accessor_result.dtype == np.dtype(np.float64)
        assert proxy_result.dtype == np.dtype(np.float64)
        pd.testing.assert_series_equal(accessor_result, expected, check_exact=False, rtol=1e-6)
        pd.testing.assert_series_equal(proxy_result, expected, check_exact=False, rtol=1e-6)
