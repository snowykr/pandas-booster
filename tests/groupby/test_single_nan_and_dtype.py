"""Single-key NaN, dtype, and array-layout behavior tests."""

from typing import Literal, cast

import numpy as np
import pandas as pd
from pandas_booster.accessor import BoosterAccessor

AggFunc = Literal["sum", "mean", "min", "max", "count", "prod"]


class TestNaNHandling:
    def test_sum_with_nan(self):
        np.random.seed(123)
        n = 200_000
        values = np.random.random(n)
        mask = np.random.random(n) < 0.1
        df = pd.DataFrame(
            {
                "key": np.repeat([1, 2], n // 2),
                "val": np.where(mask, np.nan, values),
            }
        )

        booster_result = cast(BoosterAccessor, df.booster).groupby("key", "val", "sum")
        pandas_result = df.groupby("key")["val"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    def test_mean_with_nan(self):
        np.random.seed(456)
        n = 200_000
        values = np.random.random(n)
        mask = np.random.random(n) < 0.1
        df = pd.DataFrame(
            {
                "key": np.repeat([1, 2], n // 2),
                "val": np.where(mask, np.nan, values),
            }
        )

        booster_result = cast(BoosterAccessor, df.booster).groupby("key", "val", "mean")
        pandas_result = df.groupby("key")["val"].mean()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )


class TestAllNaNGroup:
    def test_sum_all_nan_group(self):
        n = 200_000
        df = pd.DataFrame(
            {
                "key": np.concatenate([np.ones(n // 2, dtype=int), np.full(n // 2, 2, dtype=int)]),
                "val": np.concatenate([np.full(n // 2, np.nan), np.random.random(n // 2)]),
            }
        )

        booster_result = cast(BoosterAccessor, df.booster).groupby("key", "val", "sum")
        pandas_result = df.groupby("key")["val"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )


class TestNullableInteger:
    def test_nullable_int_fallback(self):
        n = 200_000
        df = pd.DataFrame(
            {
                "key": pd.array(np.random.randint(0, 100, n).tolist(), dtype="Int64"),
                "val": np.random.random(n),
            }
        )
        df.loc[0, "key"] = pd.NA

        booster_result = cast(BoosterAccessor, df.booster).groupby("key", "val", "sum")
        pandas_result = df.groupby("key")["val"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )


class TestIntegerSum:
    def test_i64_sum_matches_pandas(self):
        np.random.seed(42)
        n = 500_000
        df = pd.DataFrame(
            {
                "key": np.random.randint(0, 1000, size=n),
                "val_int": np.random.randint(0, 100, size=n),
            }
        )

        booster_result = cast(BoosterAccessor, df.booster).groupby("key", "val_int", "sum")
        pandas_result = df.groupby("key")["val_int"].sum()

        booster_sorted = booster_result.sort_index()
        pandas_sorted = pandas_result.sort_index()

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=True,
        )

    def test_i64_min_matches_pandas(self):
        np.random.seed(42)
        n = 500_000
        df = pd.DataFrame(
            {
                "key": np.random.randint(0, 1000, size=n),
                "val_int": np.random.randint(-1000, 1000, size=n),
            }
        )

        booster_result = cast(BoosterAccessor, df.booster).groupby("key", "val_int", "min")
        pandas_result = df.groupby("key")["val_int"].min()

        booster_sorted = booster_result.sort_index()
        pandas_sorted = pandas_result.sort_index()

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=True,
        )

    def test_i64_max_matches_pandas(self):
        np.random.seed(42)
        n = 500_000
        df = pd.DataFrame(
            {
                "key": np.random.randint(0, 1000, size=n),
                "val_int": np.random.randint(-1000, 1000, size=n),
            }
        )

        booster_result = cast(BoosterAccessor, df.booster).groupby("key", "val_int", "max")
        pandas_result = df.groupby("key")["val_int"].max()

        booster_sorted = booster_result.sort_index()
        pandas_sorted = pandas_result.sort_index()

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=True,
        )


class TestNonContiguousArray:
    def test_non_contiguous_slice_still_works(self):
        np.random.seed(42)
        n = 300_000
        df = pd.DataFrame(
            {
                "key": np.random.randint(0, 100, size=n),
                "val": np.random.random(size=n),
            }
        )
        sliced_df = df.iloc[::2].copy()
        sliced_df = sliced_df.reset_index(drop=True)

        booster_result = cast(BoosterAccessor, sliced_df.booster).groupby("key", "val", "sum")
        pandas_result = sliced_df.groupby("key")["val"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    def test_fortran_order_array_works(self):
        n = 200_000
        keys = np.asfortranarray(np.random.randint(0, 100, size=n))
        vals = np.asfortranarray(np.random.random(size=n))
        df = pd.DataFrame({"key": keys, "val": vals})

        booster_result = cast(BoosterAccessor, df.booster).groupby("key", "val", "sum")
        pandas_result = df.groupby("key")["val"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )
