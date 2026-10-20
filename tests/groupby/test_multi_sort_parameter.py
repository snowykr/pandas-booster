"""Multi-key sort parameter behavior tests."""

from typing import Literal, cast
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas_booster.accessor import BoosterAccessor

AggFunc = Literal["sum", "mean", "min", "max", "count", "prod"]
StdVarAgg = Literal["std", "var"]


def _assert_accelerated_sorted_result(df, agg, monkeypatch):
    monkeypatch.delenv("PANDAS_BOOSTER_FORCE_PANDAS_SORT", raising=False)
    monkeypatch.setenv("PANDAS_BOOSTER_STRICT_ABI", "1")
    accessor = cast(BoosterAccessor, df.booster)
    rust = accessor._get_rust_module()
    n_rows = max(100_001, int(rust.get_fallback_threshold()) + 1)
    repeats = (n_rows + len(df) - 1) // len(df)
    df = df.iloc[np.tile(np.arange(len(df)), repeats)].reset_index(drop=True)
    assert len(df) > rust.get_fallback_threshold()
    keys = [name for name in df.columns if name != "val"]
    kernel = "i64" if pd.api.types.is_integer_dtype(df["val"]) else "f64"
    symbol = f"groupby_multi_{agg}_{kernel}_sorted"
    with patch.object(rust, symbol, wraps=getattr(rust, symbol)) as spy:
        result = cast(BoosterAccessor, df.booster).groupby(keys, "val", agg, sort=True)
    spy.assert_called_once()
    assert len(spy.call_args.args[1]) == len(df)
    expected = getattr(df.groupby(keys, sort=True)["val"], agg)()
    pd.testing.assert_series_equal(result, expected, check_exact=True)
    pd.testing.assert_index_equal(result.index, expected.index, exact=True)
    assert result.index.names == keys
    assert [level.dtype for level in result.index.levels] == [
        level.dtype for level in expected.index.levels
    ]
    assert result.dtype == expected.dtype
    if kernel == "f64" and agg == "max":
        zeros = expected.to_numpy() == 0
        np.testing.assert_array_equal(
            np.signbit(result.to_numpy()[zeros]), np.signbit(expected.to_numpy()[zeros])
        )


class TestSortParameter:
    """Test sort=True/False parameter behavior."""

    def test_sort_true_matches_pandas_order(self, large_multi_df):
        booster_result = cast(BoosterAccessor, large_multi_df.booster).groupby(
            ["k1", "k2"], "val_float", "sum", sort=True
        )
        pandas_result = large_multi_df.groupby(["k1", "k2"], sort=True)["val_float"].sum()

        pd.testing.assert_series_equal(
            booster_result,
            pandas_result,
            check_exact=False,
            rtol=1e-10,
        )

    def test_sort_false_values_match_pandas(self, large_multi_df):
        booster_result = cast(BoosterAccessor, large_multi_df.booster).groupby(
            ["k1", "k2"], "val_float", "sum", sort=False
        )
        pandas_result = large_multi_df.groupby(["k1", "k2"], sort=False)["val_float"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    def test_sort_false_single_key(self):
        np.random.seed(42)
        n = 200_000
        df = pd.DataFrame({"key": np.random.randint(0, 1000, n), "val": np.random.random(n)})

        booster_result = cast(BoosterAccessor, df.booster).groupby("key", "val", "sum", sort=False)
        pandas_result = df.groupby("key", sort=False)["val"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    def test_sort_true_is_default(self, large_multi_df):
        result_default = cast(BoosterAccessor, large_multi_df.booster).groupby(
            ["k1", "k2"], "val_float", "sum"
        )
        result_explicit = cast(BoosterAccessor, large_multi_df.booster).groupby(
            ["k1", "k2"], "val_float", "sum", sort=True
        )

        pd.testing.assert_series_equal(result_default, result_explicit)

    def test_sort_parameter_three_keys(self, large_multi_df):
        booster_sorted = cast(BoosterAccessor, large_multi_df.booster).groupby(
            ["k1", "k2", "k3"], "val_float", "sum", sort=True
        )
        pandas_sorted = large_multi_df.groupby(["k1", "k2", "k3"], sort=True)["val_float"].sum()

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=False,
            rtol=1e-10,
        )

    def test_sort_true_multi_key_max_preserves_pandas_order_and_nan_semantics(self, monkeypatch):
        df = pd.DataFrame(
            {
                "k1": [2, -1, 2, -1, 0, 3, 3, 4, 4, 5],
                "k2": [10, 9, 10, 8, -5, 7, 7, 1, 2, 0],
                "k3": [0, 1, 0, 2, 3, 4, 4, 5, 6, 7],
                "val": [1.0, np.nan, 5.0, -2.0, 10.0, np.nan, np.nan, 8.0, -1.0, 3.0],
            }
        )

        _assert_accelerated_sorted_result(df, "max", monkeypatch)

    def test_sort_true_multi_key_count_f64_excludes_nan_in_pandas_order(self, monkeypatch):
        df = pd.DataFrame(
            {
                "k1": [3, 1, 3, 2, 1, 2, -1, -1, 4, 4, 5, 5],
                "k2": [0, 2, 0, 1, 2, 1, 9, 8, 7, 7, 6, 6],
                "k3": [5, 4, 5, 3, 4, 3, 2, 1, 0, 0, -1, -1],
                "val": [1.0, np.nan, np.nan, 2.0, 4.0, np.nan, 5.0, np.nan, 7.0, 8.0, np.nan, 9.0],
            }
        )

        _assert_accelerated_sorted_result(df, "count", monkeypatch)

    def test_sort_true_multi_key_count_i64_counts_all_rows_in_pandas_order(self, monkeypatch):
        df = pd.DataFrame(
            {
                "k1": [10, 9, 10, -2, -2, 0, 1, 1, 1, 3, 4, 4],
                "k2": [1, 1, 1, 5, 4, 0, 3, 3, 2, 8, 7, 7],
                "k3": [2, 3, 2, 1, 1, 0, -1, -1, -2, -3, -4, -4],
                "val": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            }
        )

        _assert_accelerated_sorted_result(df, "count", monkeypatch)

    @pytest.mark.parametrize("agg", ["max", "count"])
    @pytest.mark.parametrize("value_kind", ["i64", "f64", "all_nan"])
    def test_sorted_extreme_keys_repeated_tuples_and_exact_dtypes(
        self, agg, value_kind, monkeypatch
    ):
        limits = np.iinfo(np.int64)
        values = (
            np.array([limits.min, limits.max, -1, 0, 1, -7, 9, 9, 3, 4], dtype=np.int64)
            if value_kind == "i64"
            else np.array([np.nan, 2.0, -0.0, 0.0, 0.0, -0.0, np.nan, np.nan, -3.0, 4.0])
        )
        if value_kind == "all_nan":
            values[:] = np.nan
        df = pd.DataFrame(
            {
                "k1": np.array(
                    [limits.max, limits.max, limits.min, limits.min, -1, -1, 0, 0, 1, 1],
                    dtype=np.int64,
                ),
                "k2": np.array([7, 7, -8, -8, 0, 0, -1, -1, 2, 2], dtype=np.int32),
                "k3": np.array([-2, -2, 3, 3, -4, -4, 5, 5, 0, 0], dtype=np.int16),
                "val": values,
            }
        )
        _assert_accelerated_sorted_result(df, agg, monkeypatch)

    def test_non_integer_key_uses_fallback(self):
        np.random.seed(42)
        n = 200_000
        df = pd.DataFrame(
            {
                "k1": np.random.randint(0, 100, size=n),
                "k2": np.random.choice(["a", "b", "c"], size=n),  # String key
                "val": np.random.random(n),
            }
        )

        # Should fall back to Pandas and still work
        booster_result = cast(BoosterAccessor, df.booster).groupby(["k1", "k2"], "val", "sum")
        pandas_result = df.groupby(["k1", "k2"])["val"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    def test_nullable_key_uses_fallback(self):
        np.random.seed(42)
        n = 200_000
        df = pd.DataFrame(
            {
                "k1": pd.array(np.random.randint(0, 100, n).tolist(), dtype="Int64"),
                "k2": np.random.randint(0, 50, n),
                "val": np.random.random(n),
            }
        )
        df.loc[0, "k1"] = pd.NA  # Make it nullable

        booster_result = cast(BoosterAccessor, df.booster).groupby(["k1", "k2"], "val", "sum")
        pandas_result = df.groupby(["k1", "k2"])["val"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )
