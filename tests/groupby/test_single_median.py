"""Single-key median behavior tests."""

from typing import Literal, cast

import numpy as np
import pandas as pd
from pandas_booster.accessor import BoosterAccessor

AggFunc = Literal["sum", "mean", "min", "max", "count", "prod"]


class TestMedianSingleKeyAccessor:
    def _make_float_df(self) -> pd.DataFrame:
        n = 200_000
        keys = np.full(n, 4, dtype=np.int64)
        vals = np.full(n, 10.0, dtype=np.float64)

        keys[[0, 50_000, 100_000]] = 3
        vals[[0, 50_000, 100_000]] = [1.0, 5.0, 9.0]

        keys[[1, 60_000]] = 2
        vals[[1, 60_000]] = np.nan

        keys[[2, 30_000, 90_000]] = 1
        vals[[2, 30_000, 90_000]] = [np.nan, 4.0, 6.0]

        keys[3] = 4
        vals[3] = 11.0

        keys[[4, 150_000]] = 5
        vals[[4, 150_000]] = [2.0, 8.0]

        return pd.DataFrame({"key": keys, "val": vals})

    def _make_int_df(self) -> pd.DataFrame:
        n = 200_000
        keys = np.array([2, 99, 1, 5], dtype=np.int64)
        keys = np.tile(keys, (n + len(keys) - 1) // len(keys))[:n]
        vals = (np.arange(n, dtype=np.int64) % 12) - 6
        return pd.DataFrame({"key": keys, "val": vals})

    def test_float_special_cases_match_pandas_sort_true(self):
        df = self._make_float_df()

        booster_result = cast(BoosterAccessor, df.booster).groupby("key", "val", "median")
        pandas_result = df.groupby("key")["val"].median()

        booster_sorted = booster_result.sort_index()
        pandas_sorted = pandas_result.sort_index()

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=False,
            rtol=1e-10,
        )
        assert booster_result.dtype == np.float64
        assert booster_sorted.loc[3] == 5.0
        assert booster_sorted.loc[5] == 5.0
        assert np.isnan(booster_sorted.loc[2])

    def test_float_special_cases_match_pandas_sort_false(self):
        df = self._make_float_df()

        booster_result = cast(BoosterAccessor, df.booster).groupby(
            "key", "val", "median", sort=False
        )
        pandas_result = df.groupby("key", sort=False)["val"].median()

        assert booster_result.index.tolist() == pandas_result.index.tolist() == [3, 2, 1, 4, 5]
        pd.testing.assert_series_equal(
            booster_result,
            pandas_result,
            check_exact=False,
            rtol=1e-10,
        )
        assert booster_result.dtype == np.float64

    def test_int_values_match_pandas_sort_true_with_float64_dtype(self):
        df = self._make_int_df()

        booster_result = cast(BoosterAccessor, df.booster).groupby("key", "val", "median")
        pandas_result = df.groupby("key")["val"].median()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )
        assert booster_result.dtype == np.float64

    def test_int_values_match_pandas_sort_false_with_float64_dtype(self):
        df = self._make_int_df()

        booster_result = cast(BoosterAccessor, df.booster).groupby(
            "key", "val", "median", sort=False
        )
        pandas_result = df.groupby("key", sort=False)["val"].median()

        assert booster_result.index.tolist() == pandas_result.index.tolist() == [2, 99, 1, 5]
        pd.testing.assert_series_equal(
            booster_result,
            pandas_result,
            check_exact=False,
            rtol=1e-10,
        )
        assert booster_result.dtype == np.float64
