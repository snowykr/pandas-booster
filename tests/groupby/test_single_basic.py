"""Single-key basic groupby behavior tests."""

from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import pytest
from pandas_booster.accessor import BoosterAccessor

AggFunc = Literal["sum", "mean", "min", "max", "count", "prod"]


class TestBoosterGroupBy:
    def test_sum_f64_matches_pandas(self, large_df):
        booster_result = cast(BoosterAccessor, large_df.booster).groupby("key", "val_float", "sum")
        pandas_result = large_df.groupby("key")["val_float"].sum()

        booster_sorted = booster_result.sort_index()
        pandas_sorted = pandas_result.sort_index()

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=False,
            rtol=1e-10,
        )

    def test_mean_f64_matches_pandas(self, large_df):
        booster_result = cast(BoosterAccessor, large_df.booster).groupby("key", "val_float", "mean")
        pandas_result = large_df.groupby("key")["val_float"].mean()

        booster_sorted = booster_result.sort_index()
        pandas_sorted = pandas_result.sort_index()

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=False,
            rtol=1e-10,
        )

    def test_min_f64_matches_pandas(self, large_df):
        booster_result = cast(BoosterAccessor, large_df.booster).groupby("key", "val_float", "min")
        pandas_result = large_df.groupby("key")["val_float"].min()

        booster_sorted = booster_result.sort_index()
        pandas_sorted = pandas_result.sort_index()

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=False,
            rtol=1e-10,
        )

    def test_max_f64_matches_pandas(self, large_df):
        booster_result = cast(BoosterAccessor, large_df.booster).groupby("key", "val_float", "max")
        pandas_result = large_df.groupby("key")["val_float"].max()

        booster_sorted = booster_result.sort_index()
        pandas_sorted = pandas_result.sort_index()

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=False,
            rtol=1e-10,
        )

    def test_sum_i64_matches_pandas(self, large_df):
        booster_result = cast(BoosterAccessor, large_df.booster).groupby("key", "val_int", "sum")
        pandas_result = large_df.groupby("key")["val_int"].sum()

        booster_sorted = booster_result.sort_index()
        pandas_sorted = pandas_result.sort_index()

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=True,
        )

    def test_small_df_uses_fallback(self, small_df):
        booster_result = cast(BoosterAccessor, small_df.booster).groupby("key", "val_float", "sum")
        pandas_result = small_df.groupby("key")["val_float"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    def test_median_matches_pandas_on_large_supported_input(self, large_df):
        booster_result = cast(BoosterAccessor, large_df.booster).groupby(
            "key", "val_float", "median"
        )
        pandas_result = large_df.groupby("key")["val_float"].median()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )


class TestGroupbyCount:
    def test_count_f64_matches_pandas(self, large_df):
        booster_result = cast(BoosterAccessor, large_df.booster).groupby(
            "key", "val_float", "count"
        )
        pandas_result = large_df.groupby("key")["val_float"].count()

        booster_sorted = booster_result.sort_index()
        pandas_sorted = pandas_result.sort_index()

        # Check values match
        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=True,
        )

        # Check dtype is int64
        assert pd.api.types.is_integer_dtype(booster_result.dtype)

    def test_count_i64_matches_pandas(self, large_df):
        booster_result = cast(BoosterAccessor, large_df.booster).groupby("key", "val_int", "count")
        pandas_result = large_df.groupby("key")["val_int"].count()

        booster_sorted = booster_result.sort_index()
        pandas_sorted = pandas_result.sort_index()

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=True,
        )
        assert pd.api.types.is_integer_dtype(booster_result.dtype)

    def test_count_with_nan_matches_pandas(self):
        np.random.seed(42)
        n = 200_000
        values = np.random.random(n)
        mask = np.random.random(n) < 0.1
        df = pd.DataFrame(
            {
                "key": np.random.randint(0, 100, size=n),
                "val": np.where(mask, np.nan, values),
            }
        )

        booster_result = cast(BoosterAccessor, df.booster).groupby("key", "val", "count")
        pandas_result = df.groupby("key")["val"].count()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=True,
        )
        assert pd.api.types.is_integer_dtype(booster_result.dtype)

    def test_invalid_agg_raises(self, large_df):
        with pytest.raises(ValueError, match="Unsupported aggregation"):
            cast(BoosterAccessor, large_df.booster).groupby(
                "key", "val_float", cast(Any, "invalid")
            )

    def test_thread_count(self, large_df):
        thread_count = large_df.booster.thread_count()
        assert thread_count > 0
