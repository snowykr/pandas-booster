"""Single-key std and var behavior tests."""

from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import pytest
from pandas_booster.accessor import BoosterAccessor

AggFunc = Literal["sum", "mean", "min", "max", "count", "prod"]


class TestStdVarSingleKeyRed:
    def _make_float_df(self) -> pd.DataFrame:
        n = 200_000
        keys = np.full(n, 4, dtype=np.int64)
        vals = np.full(n, 10.0, dtype=np.float64)

        # First-seen order for sort=False coverage.
        keys[0] = 3
        vals[0] = 1.0

        keys[1] = 2
        vals[1] = np.nan

        keys[2] = 1
        vals[2] = 7.0

        keys[3] = 4
        vals[3] = 5.0

        # Group 3 has mixed NaN/non-NaN values and at least two observations.
        mixed_rows = np.arange(4, 80_004, dtype=np.int64)
        keys[mixed_rows] = 3
        vals[mixed_rows] = np.where((mixed_rows % 2) == 0, np.nan, 3.0)

        # Group 2 is all NaN.
        all_nan_rows = np.arange(80_004, 120_004, dtype=np.int64)
        keys[all_nan_rows] = 2
        vals[all_nan_rows] = np.nan

        return pd.DataFrame({"key": keys, "val": vals})

    def _make_int_df(self) -> pd.DataFrame:
        n = 200_000
        keys = np.array([2, 99, 1, 5], dtype=np.int64)
        keys = np.tile(keys, (n + len(keys) - 1) // len(keys))[:n]
        vals = (np.arange(n, dtype=np.int64) % 11) - 5
        return pd.DataFrame({"key": keys, "val": vals})

    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_float_special_cases_match_pandas_sort_true(self, agg: str):
        df = self._make_float_df()

        booster_result = cast(BoosterAccessor, df.booster).groupby("key", "val", cast(Any, agg))
        pandas_result = getattr(df.groupby("key")["val"], agg)()

        booster_sorted = booster_result.sort_index()
        pandas_sorted = pandas_result.sort_index()

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=False,
            rtol=1e-10,
        )
        assert booster_result.dtype == np.float64
        assert np.isnan(booster_sorted.loc[1])
        assert np.isnan(booster_sorted.loc[2])
        assert np.isfinite(booster_sorted.loc[3])

    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_float_special_cases_match_pandas_sort_false(self, agg: str):
        df = self._make_float_df()

        booster_result = cast(BoosterAccessor, df.booster).groupby(
            "key", "val", cast(Any, agg), sort=False
        )
        pandas_result = getattr(df.groupby("key", sort=False)["val"], agg)()

        assert booster_result.index.tolist() == pandas_result.index.tolist() == [3, 2, 1, 4]
        pd.testing.assert_series_equal(
            booster_result,
            pandas_result,
            check_exact=False,
            rtol=1e-10,
        )
        assert booster_result.dtype == np.float64

    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_int_values_match_pandas_sort_true_with_float64_dtype(self, agg: str):
        df = self._make_int_df()

        booster_result = cast(BoosterAccessor, df.booster).groupby("key", "val", cast(Any, agg))
        pandas_result = getattr(df.groupby("key")["val"], agg)()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )
        assert booster_result.dtype == np.float64

    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_int_values_match_pandas_sort_false_with_float64_dtype(self, agg: str):
        df = self._make_int_df()

        booster_result = cast(BoosterAccessor, df.booster).groupby(
            "key", "val", cast(Any, agg), sort=False
        )
        pandas_result = getattr(df.groupby("key", sort=False)["val"], agg)()

        assert booster_result.index.tolist() == pandas_result.index.tolist() == [2, 99, 1, 5]
        pd.testing.assert_series_equal(
            booster_result,
            pandas_result,
            check_exact=False,
            rtol=1e-10,
        )
        assert booster_result.dtype == np.float64
