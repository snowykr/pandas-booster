"""Std/var dispatch and rollback contract tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ._helpers import (
    _accessor_groupby_result,
    _patch_all_std_var_kernels_to_raise,
    _patch_pandas_series_groupby_agg_to_raise,
    _patch_single_std_var_firstseen_only_kernel,
    _patch_single_std_var_kernel,
    _proxy_groupby_result,
)


class TestStdVarDispatchContracts:
    @pytest.mark.parametrize(
        ("kernel", "values", "expected_order"),
        [
            (
                "f64",
                np.array([1.0, np.nan, 3.0, np.nan, 5.0, 11.0, 2.0, 2.0, np.nan]),
                [5, 2, 4, 7, 8, 1],
            ),
            (
                "i64",
                np.array([2, 6, 4, 10, 14, 12], dtype=np.int64),
                [8, 3, 5, 4],
            ),
        ],
    )
    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_single_key_unsorted_std_var_supported_calls_route_to_firstseen_kernels(
        self,
        monkeypatch: pytest.MonkeyPatch,
        agg: str,
        kernel: str,
        values: np.ndarray,
        expected_order: list[int],
    ):
        import pandas_booster._rust as rust

        monkeypatch.delenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", raising=False)

        key_values = [8, 3, 8, 5, 3, 4] if kernel == "i64" else [5, 2, 5, 4, 2, 7, 8, 8, 1]
        df = pd.DataFrame({"key": key_values, "val": values})
        expected = getattr(df.groupby("key", sort=False)["val"], agg)()
        calls: list[str] = []

        _patch_single_std_var_firstseen_only_kernel(
            monkeypatch,
            rust,
            expected,
            agg,
            kernel=kernel,
            result_dtype=np.dtype(np.float64),
            calls=calls,
        )
        _patch_pandas_series_groupby_agg_to_raise(
            monkeypatch,
            agg,
            "supported single-key unsorted std/var should not fall back to pandas",
        )

        accessor_result = _accessor_groupby_result(df, "key", "val", agg, sort=False)
        proxy_result = _proxy_groupby_result(df, "key", "val", agg, sort=False)

        assert calls == ["firstseen", "firstseen"]
        assert accessor_result.index.tolist() == expected_order
        assert proxy_result.index.tolist() == expected_order
        pd.testing.assert_series_equal(accessor_result, expected, check_exact=False, rtol=1e-12)
        pd.testing.assert_series_equal(proxy_result, expected, check_exact=False, rtol=1e-12)

    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_small_single_key_float_std_var_uses_rust_without_env_rollback(
        self, monkeypatch: pytest.MonkeyPatch, agg: str
    ):
        import pandas_booster._rust as rust

        monkeypatch.delenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", raising=False)

        df = pd.DataFrame(
            {
                "key": [1, 1, 2, 2, 3, 3],
                "val": [1.5, 2.5, 10.0, 14.0, 5.5, 8.5],
            }
        )
        expected = getattr(df.groupby("key", sort=True)["val"], agg)()

        _patch_single_std_var_kernel(
            monkeypatch, rust, expected, agg, kernel="f64", result_dtype=np.float64
        )
        _patch_pandas_series_groupby_agg_to_raise(
            monkeypatch,
            agg,
            (
                "supported single-key float std/var should not fall back to pandas "
                "when rollback is off"
            ),
        )

        accessor_result = _accessor_groupby_result(df, "key", "val", agg)
        proxy_result = _proxy_groupby_result(df, "key", "val", agg)

        assert accessor_result.dtype == np.dtype(np.float64)
        assert proxy_result.dtype == np.dtype(np.float64)
        pd.testing.assert_series_equal(accessor_result, expected, check_exact=False, rtol=1e-12)
        pd.testing.assert_series_equal(proxy_result, expected, check_exact=False, rtol=1e-12)

    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_single_key_float_std_var_env_rollback_forces_fallback(
        self, monkeypatch: pytest.MonkeyPatch, agg: str
    ):
        import pandas_booster._rust as rust

        monkeypatch.setenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", "1")

        df = pd.DataFrame(
            {
                "key": np.repeat([1, 2, 3], 4),
                "val": np.array(
                    [
                        1.0,
                        2.0,
                        4.0,
                        8.0,
                        2.5,
                        3.5,
                        6.5,
                        7.5,
                        5.0,
                        6.0,
                        9.0,
                        10.0,
                    ]
                ),
            }
        )
        expected = getattr(df.groupby("key", sort=True)["val"], agg)()

        _patch_all_std_var_kernels_to_raise(
            monkeypatch,
            rust,
            "single-key float std/var should use pandas fallback when rollback env is enabled",
        )

        accessor_result = _accessor_groupby_result(df, "key", "val", agg)
        proxy_result = _proxy_groupby_result(df, "key", "val", agg)

        pd.testing.assert_series_equal(accessor_result, expected, check_exact=False, rtol=1e-12)
        pd.testing.assert_series_equal(proxy_result, expected, check_exact=False, rtol=1e-12)

    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_single_key_unsorted_float_std_var_env_rollback_forces_fallback(
        self, monkeypatch: pytest.MonkeyPatch, agg: str
    ):
        import pandas_booster._rust as rust

        monkeypatch.setenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", "1")

        df = pd.DataFrame(
            {
                "key": [5, 2, 5, 4, 2, 7, 8, 8, 1],
                "val": [1.0, np.nan, 3.0, np.nan, 5.0, 11.0, 2.0, 2.0, np.nan],
            }
        )
        expected = getattr(df.groupby("key", sort=False)["val"], agg)()

        _patch_all_std_var_kernels_to_raise(
            monkeypatch,
            rust,
            (
                "single-key unsorted float std/var should use pandas fallback "
                "when rollback env is enabled"
            ),
        )

        accessor_result = _accessor_groupby_result(df, "key", "val", agg, sort=False)
        proxy_result = _proxy_groupby_result(df, "key", "val", agg, sort=False)

        assert accessor_result.index.tolist() == [5, 2, 4, 7, 8, 1]
        assert proxy_result.index.tolist() == [5, 2, 4, 7, 8, 1]
        pd.testing.assert_series_equal(accessor_result, expected, check_exact=False, rtol=1e-12)
        pd.testing.assert_series_equal(proxy_result, expected, check_exact=False, rtol=1e-12)
