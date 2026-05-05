from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import pytest
from pandas_booster.accessor import BoosterAccessor

AggFunc = Literal["sum", "mean", "min", "max", "count", "prod"]


@pytest.fixture
def large_df():
    np.random.seed(42)
    n = 500_000
    return pd.DataFrame(
        {
            "key": np.random.randint(0, 1000, size=n),
            "val_int": np.random.randint(0, 100, size=n),
            "val_float": np.random.random(size=n) * 100,
        }
    )


@pytest.fixture
def small_df():
    return pd.DataFrame(
        {
            "key": [1, 2, 1, 2, 1],
            "val_int": [10, 20, 30, 40, 50],
            "val_float": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


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


class TestFirstSeenOrderSingleKeySortFalse:
    def _make_df(self) -> pd.DataFrame:
        n = 200_000
        keys = np.empty(n, dtype=np.int64)
        vals = np.empty(n, dtype=np.float64)

        # Deterministic first-seen group order across the dataset.
        # Include a group whose values are all NaN to ensure the group
        # is not dropped in the first-seen path.
        first_order = [2, 99, 1, 5]
        first_rows = [0, 60_000, 120_000, 180_000]

        # Build keys in segments so each group's first appearance is far apart.
        # - [0, 60_000): only group 2
        # - [60_000, 120_000): groups 2, 99
        # - [120_000, 180_000): groups 2, 99, 1
        # - [180_000, n): groups 2, 99, 1, 5
        keys[:] = 2
        if first_rows[1] + 1 < first_rows[2]:
            seg = np.array([2, 99], dtype=np.int64)
            keys[first_rows[1] + 1 : first_rows[2]] = seg[
                np.arange(first_rows[2] - (first_rows[1] + 1)) % seg.size
            ]
        if first_rows[2] + 1 < first_rows[3]:
            seg = np.array([2, 99, 1], dtype=np.int64)
            keys[first_rows[2] + 1 : first_rows[3]] = seg[
                np.arange(first_rows[3] - (first_rows[2] + 1)) % seg.size
            ]
        if first_rows[3] + 1 < n:
            seg = np.array(first_order, dtype=np.int64)
            keys[first_rows[3] + 1 :] = seg[np.arange(n - (first_rows[3] + 1)) % seg.size]

        # Stamp first-seen rows (must not be overwritten).
        for k, row in zip(first_order, first_rows):
            keys[row] = k

        # Sanity: verify first-seen rows are exactly as intended.
        for k, row in zip(first_order, first_rows):
            first_idx = int(np.flatnonzero(keys == k)[0])
            assert first_idx == row

        vals[:] = np.random.random(n).astype(np.float64)

        # Force group 99 to be all-NaN.
        vals[keys == 99] = np.nan

        return pd.DataFrame({"key": keys, "val": vals})

    @pytest.mark.parametrize("agg", ["sum", "mean", "min", "max", "count", "prod"])
    def test_sort_false_preserves_first_seen_float(self, agg: AggFunc):
        df = self._make_df()
        booster_result = cast(BoosterAccessor, df.booster).groupby("key", "val", agg, sort=False)
        pandas_result = getattr(df.groupby("key", sort=False)["val"], agg)()

        # Order must match Pandas appearance order.
        assert booster_result.index.tolist() == pandas_result.index.tolist() == [2, 99, 1, 5]

        pd.testing.assert_series_equal(
            booster_result,
            pandas_result,
            check_exact=(agg == "count"),
            rtol=(0.0 if agg == "count" else 1e-10),
        )

    @pytest.mark.parametrize("agg", ["sum", "mean", "min", "max", "count", "prod"])
    def test_sort_false_preserves_first_seen_int(self, agg: AggFunc):
        n = 200_000
        keys = np.array([2, 99, 1, 5], dtype=np.int64)
        keys = np.tile(keys, (n + len(keys) - 1) // len(keys))[:n]
        vals = (np.arange(n, dtype=np.int64) % 7) - 3
        df = pd.DataFrame({"key": keys, "val": vals})

        booster_result = cast(BoosterAccessor, df.booster).groupby("key", "val", agg, sort=False)
        pandas_result = getattr(df.groupby("key", sort=False)["val"], agg)()

        assert booster_result.index.tolist() == pandas_result.index.tolist() == [2, 99, 1, 5]
        pd.testing.assert_series_equal(
            booster_result,
            pandas_result,
            check_exact=(agg != "mean"),
            check_dtype=True,
            rtol=(1e-10 if agg == "mean" else 0.0),
        )

    def test_mean_all_nan_group(self):
        n = 200_000
        df = pd.DataFrame(
            {
                "key": np.concatenate([np.ones(n // 2, dtype=int), np.full(n // 2, 2, dtype=int)]),
                "val": np.concatenate([np.full(n // 2, np.nan), np.random.random(n // 2)]),
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

    def test_min_all_nan_group(self):
        n = 200_000
        df = pd.DataFrame(
            {
                "key": np.concatenate([np.ones(n // 2, dtype=int), np.full(n // 2, 2, dtype=int)]),
                "val": np.concatenate([np.full(n // 2, np.nan), np.random.random(n // 2)]),
            }
        )

        booster_result = cast(BoosterAccessor, df.booster).groupby("key", "val", "min")
        pandas_result = df.groupby("key")["val"].min()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    def test_max_all_nan_group(self):
        n = 200_000
        df = pd.DataFrame(
            {
                "key": np.concatenate([np.ones(n // 2, dtype=int), np.full(n // 2, 2, dtype=int)]),
                "val": np.concatenate([np.full(n // 2, np.nan), np.random.random(n // 2)]),
            }
        )

        booster_result = cast(BoosterAccessor, df.booster).groupby("key", "val", "max")
        pandas_result = df.groupby("key")["val"].max()

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


class TestHighCardinality:
    def test_high_cardinality_unique_keys(self):
        n = 150_000
        df = pd.DataFrame(
            {
                "key": np.arange(n),
                "val": np.ones(n),
            }
        )

        booster_result = cast(BoosterAccessor, df.booster).groupby("key", "val", "sum")
        pandas_result = df.groupby("key")["val"].sum()

        assert len(booster_result) == n
        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    def test_high_cardinality_mean(self):
        np.random.seed(123)
        n = 200_000
        n_groups = 50_000
        df = pd.DataFrame(
            {
                "key": np.random.randint(0, n_groups, size=n),
                "val": np.random.random(size=n),
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


class TestFirstSeenOrderSortFalse:
    def _make_ordered_df(self) -> pd.DataFrame:
        # Must be above fallback threshold to force acceleration.
        np.random.seed(123)
        n = 200_000
        keys = np.empty(n, dtype=np.int64)
        pattern = np.array([5, 1, 3, 0], dtype=np.int64)
        keys[: pattern.size] = pattern
        # Fill remaining with the same key set (no new groups introduced).
        keys[pattern.size :] = np.tile(
            pattern, (n - pattern.size + pattern.size - 1) // pattern.size
        )[: n - pattern.size]
        vals = np.random.random(n).astype(np.float64)
        mask = np.random.random(n) < 0.05
        vals[mask] = np.nan
        return pd.DataFrame({"key": keys, "val": vals})

    @pytest.mark.parametrize("agg", ["sum", "mean", "min", "max", "count", "prod"])
    def test_booster_sort_false_preserves_first_seen_single_key(self, agg: AggFunc):
        df = self._make_ordered_df()
        booster_result = cast(BoosterAccessor, df.booster).groupby("key", "val", agg, sort=False)
        pandas_grouped = df.groupby("key", sort=False)["val"]
        pandas_result = getattr(pandas_grouped, agg)()

        pd.testing.assert_series_equal(
            booster_result,
            pandas_result,
            check_exact=(agg == "count"),
            rtol=(0.0 if agg == "count" else 1e-10),
        )

    @pytest.mark.parametrize("agg", ["sum", "mean", "min", "max", "count", "prod"])
    def test_proxy_sort_false_preserves_first_seen_single_key(self, agg: AggFunc):
        import pandas_booster

        df = self._make_ordered_df()
        pandas_grouped = df.groupby("key", sort=False)["val"]
        pandas_result = getattr(pandas_grouped, agg)()

        pandas_booster.activate()
        try:
            proxy_grouped = df.groupby("key", sort=False)["val"]
            proxy_result = getattr(proxy_grouped, agg)()
        finally:
            pandas_booster.deactivate()

        pd.testing.assert_series_equal(
            proxy_result,
            pandas_result,
            check_exact=(agg == "count"),
            rtol=(0.0 if agg == "count" else 1e-10),
        )


def _assert_prod_series_matches_pandas(
    booster_result: pd.Series,
    pandas_result: pd.Series,
    *,
    check_dtype: bool = True,
) -> None:
    pd.testing.assert_series_equal(
        booster_result,
        pandas_result,
        check_exact=False,
        check_dtype=check_dtype,
        rtol=1e-10,
        atol=0.0,
    )
    zero_mask = (pandas_result.to_numpy() == 0.0) & ~pd.isna(pandas_result.to_numpy())
    if zero_mask.any():
        np.testing.assert_array_equal(
            np.signbit(booster_result.to_numpy()[zero_mask]),
            np.signbit(pandas_result.to_numpy()[zero_mask]),
        )


class TestProdSingleKey:
    def test_prod_f64_matches_pandas_sort_modes(self):
        n = 200_000
        idx = np.arange(n, dtype=np.int64)
        df = pd.DataFrame(
            {
                "key": np.tile(np.array([3, 1, 2, 5], dtype=np.int64), n // 4),
                "val": (1.0 + (idx % 7).astype(np.float64) / 1000.0),
            }
        )
        df.loc[df.index % 997 == 0, "val"] = np.nan

        for sort in (True, False):
            booster_result = cast(BoosterAccessor, df.booster).groupby(
                "key", "val", "prod", sort=sort
            )
            pandas_result = df.groupby("key", sort=sort)["val"].prod()
            _assert_prod_series_matches_pandas(booster_result, pandas_result)

    def test_prod_i64_matches_pandas_sort_modes(self):
        n = 200_000
        vals = np.tile(np.array([-2, -1, 1, 2, 3], dtype=np.int64), n // 5)
        df = pd.DataFrame(
            {
                "key": np.tile(np.array([9, 4, 9, 1, 4], dtype=np.int64), n // 5),
                "val": vals,
            }
        )

        for sort in (True, False):
            booster_result = cast(BoosterAccessor, df.booster).groupby(
                "key", "val", "prod", sort=sort
            )
            pandas_result = df.groupby("key", sort=sort)["val"].prod()
            pd.testing.assert_series_equal(booster_result, pandas_result, check_exact=True)

    def test_prod_float_special_values_match_pandas_observables(self):
        pairs = [
            (10, np.nan),
            (10, np.nan),
            (20, np.nan),
            (20, 2.0),
            (30, np.inf),
            (30, 0.0),
            (40, -np.inf),
            (40, 0.0),
            (50, -0.0),
            (50, 2.0),
            (60, 0.0),
            (60, -2.0),
            (70, 1e308),
            (70, 1e308),
            (80, 1e-308),
            (80, 1e-308),
        ]
        repeats = 12_500
        df = pd.DataFrame(
            {
                "key": np.array([k for k, _ in pairs] * repeats, dtype=np.int64),
                "val": np.array([v for _, v in pairs] * repeats, dtype=np.float64),
            }
        )

        booster_result = cast(BoosterAccessor, df.booster).groupby("key", "val", "prod", sort=True)
        pandas_result = df.groupby("key", sort=True)["val"].prod()

        _assert_prod_series_matches_pandas(booster_result, pandas_result)
        assert booster_result.loc[10] == pandas_result.loc[10] == 1.0
        assert np.isnan(booster_result.loc[30]) and np.isnan(pandas_result.loc[30])
        assert np.isnan(booster_result.loc[40]) and np.isnan(pandas_result.loc[40])
        assert np.signbit(booster_result.loc[50]) == np.signbit(pandas_result.loc[50])
        assert np.signbit(booster_result.loc[60]) == np.signbit(pandas_result.loc[60])

    def test_single_key_float_prod_respects_force_pandas_escape_hatch(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        import pandas_booster._rust as rust

        n = 120_000
        df = pd.DataFrame(
            {
                "key": np.repeat(np.array([1, 2], dtype=np.int64), n // 2),
                "val": np.linspace(1.001, 1.01, n, dtype=np.float64),
            }
        )
        expected = df.groupby("key")["val"].prod()

        def _boom(*_args, **_kwargs):
            raise AssertionError(
                "float prod should use pandas when force-pandas escape hatch is set"
            )

        monkeypatch.setenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", "1")
        for suffix in ("", "_sorted", "_firstseen_u32", "_firstseen_u64"):
            monkeypatch.setattr(rust, f"groupby_prod_f64{suffix}", _boom, raising=False)

        result = cast(BoosterAccessor, df.booster).groupby("key", "val", "prod", sort=True)
        pd.testing.assert_series_equal(result, expected)
