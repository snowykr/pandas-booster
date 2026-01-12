import numpy as np
import pandas as pd
import pytest


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
        import pandas_booster

        booster_result = large_df.booster.groupby("key", "val_float", "sum")
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
        import pandas_booster

        booster_result = large_df.booster.groupby("key", "val_float", "mean")
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
        import pandas_booster

        booster_result = large_df.booster.groupby("key", "val_float", "min")
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
        import pandas_booster

        booster_result = large_df.booster.groupby("key", "val_float", "max")
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
        import pandas_booster

        booster_result = large_df.booster.groupby("key", "val_int", "sum")
        pandas_result = large_df.groupby("key")["val_int"].sum()

        booster_sorted = booster_result.sort_index().astype(float)
        pandas_sorted = pandas_result.sort_index().astype(float)

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=False,
            rtol=1e-10,
        )

    def test_small_df_uses_fallback(self, small_df):
        import pandas_booster

        booster_result = small_df.booster.groupby("key", "val_float", "sum")
        pandas_result = small_df.groupby("key")["val_float"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
        )

    def test_invalid_agg_raises(self, large_df):
        import pandas_booster

        with pytest.raises(ValueError, match="Unsupported aggregation"):
            large_df.booster.groupby("key", "val_float", "invalid")

    def test_thread_count(self, large_df):
        import pandas_booster

        thread_count = large_df.booster.thread_count()
        assert thread_count > 0


class TestNaNHandling:
    def test_sum_with_nan(self):
        import pandas_booster

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

        booster_result = df.booster.groupby("key", "val", "sum")
        pandas_result = df.groupby("key")["val"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    def test_mean_with_nan(self):
        import pandas_booster

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

        booster_result = df.booster.groupby("key", "val", "mean")
        pandas_result = df.groupby("key")["val"].mean()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )


class TestAllNaNGroup:
    def test_sum_all_nan_group(self):
        import pandas_booster

        n = 200_000
        df = pd.DataFrame(
            {
                "key": np.concatenate([np.ones(n // 2, dtype=int), np.full(n // 2, 2, dtype=int)]),
                "val": np.concatenate([np.full(n // 2, np.nan), np.random.random(n // 2)]),
            }
        )

        booster_result = df.booster.groupby("key", "val", "sum")
        pandas_result = df.groupby("key")["val"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    def test_mean_all_nan_group(self):
        import pandas_booster

        n = 200_000
        df = pd.DataFrame(
            {
                "key": np.concatenate([np.ones(n // 2, dtype=int), np.full(n // 2, 2, dtype=int)]),
                "val": np.concatenate([np.full(n // 2, np.nan), np.random.random(n // 2)]),
            }
        )

        booster_result = df.booster.groupby("key", "val", "mean")
        pandas_result = df.groupby("key")["val"].mean()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    def test_min_all_nan_group(self):
        import pandas_booster

        n = 200_000
        df = pd.DataFrame(
            {
                "key": np.concatenate([np.ones(n // 2, dtype=int), np.full(n // 2, 2, dtype=int)]),
                "val": np.concatenate([np.full(n // 2, np.nan), np.random.random(n // 2)]),
            }
        )

        booster_result = df.booster.groupby("key", "val", "min")
        pandas_result = df.groupby("key")["val"].min()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    def test_max_all_nan_group(self):
        import pandas_booster

        n = 200_000
        df = pd.DataFrame(
            {
                "key": np.concatenate([np.ones(n // 2, dtype=int), np.full(n // 2, 2, dtype=int)]),
                "val": np.concatenate([np.full(n // 2, np.nan), np.random.random(n // 2)]),
            }
        )

        booster_result = df.booster.groupby("key", "val", "max")
        pandas_result = df.groupby("key")["val"].max()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )


class TestNullableInteger:
    def test_nullable_int_fallback(self):
        import pandas_booster

        n = 200_000
        df = pd.DataFrame(
            {
                "key": pd.array(np.random.randint(0, 100, n), dtype="Int64"),
                "val": np.random.random(n),
            }
        )
        df.loc[0, "key"] = pd.NA

        booster_result = df.booster.groupby("key", "val", "sum")
        pandas_result = df.groupby("key")["val"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )


class TestIntegerSum:
    def test_i64_sum_matches_pandas(self):
        import pandas_booster

        np.random.seed(42)
        n = 500_000
        df = pd.DataFrame(
            {
                "key": np.random.randint(0, 1000, size=n),
                "val_int": np.random.randint(0, 100, size=n),
            }
        )

        booster_result = df.booster.groupby("key", "val_int", "sum")
        pandas_result = df.groupby("key")["val_int"].sum()

        booster_sorted = booster_result.sort_index()
        pandas_sorted = pandas_result.sort_index().astype(float)

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=False,
            rtol=1e-10,
        )

    def test_i64_min_matches_pandas(self):
        import pandas_booster

        np.random.seed(42)
        n = 500_000
        df = pd.DataFrame(
            {
                "key": np.random.randint(0, 1000, size=n),
                "val_int": np.random.randint(-1000, 1000, size=n),
            }
        )

        booster_result = df.booster.groupby("key", "val_int", "min")
        pandas_result = df.groupby("key")["val_int"].min()

        booster_sorted = booster_result.sort_index()
        pandas_sorted = pandas_result.sort_index().astype(float)

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=False,
            rtol=1e-10,
        )

    def test_i64_max_matches_pandas(self):
        import pandas_booster

        np.random.seed(42)
        n = 500_000
        df = pd.DataFrame(
            {
                "key": np.random.randint(0, 1000, size=n),
                "val_int": np.random.randint(-1000, 1000, size=n),
            }
        )

        booster_result = df.booster.groupby("key", "val_int", "max")
        pandas_result = df.groupby("key")["val_int"].max()

        booster_sorted = booster_result.sort_index()
        pandas_sorted = pandas_result.sort_index().astype(float)

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=False,
            rtol=1e-10,
        )


class TestNonContiguousArray:
    def test_non_contiguous_slice_still_works(self):
        import pandas_booster

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

        booster_result = sliced_df.booster.groupby("key", "val", "sum")
        pandas_result = sliced_df.groupby("key")["val"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    def test_fortran_order_array_works(self):
        import pandas_booster

        n = 200_000
        keys = np.asfortranarray(np.random.randint(0, 100, size=n))
        vals = np.asfortranarray(np.random.random(size=n))
        df = pd.DataFrame({"key": keys, "val": vals})

        booster_result = df.booster.groupby("key", "val", "sum")
        pandas_result = df.groupby("key")["val"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )


class TestHighCardinality:
    def test_high_cardinality_unique_keys(self):
        import pandas_booster

        n = 150_000
        df = pd.DataFrame(
            {
                "key": np.arange(n),
                "val": np.ones(n),
            }
        )

        booster_result = df.booster.groupby("key", "val", "sum")
        pandas_result = df.groupby("key")["val"].sum()

        assert len(booster_result) == n
        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    def test_high_cardinality_mean(self):
        import pandas_booster

        np.random.seed(123)
        n = 200_000
        n_groups = 50_000
        df = pd.DataFrame(
            {
                "key": np.random.randint(0, n_groups, size=n),
                "val": np.random.random(size=n),
            }
        )

        booster_result = df.booster.groupby("key", "val", "mean")
        pandas_result = df.groupby("key")["val"].mean()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )
