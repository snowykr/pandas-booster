"""Multi-key basic groupby behavior tests."""

from typing import Literal, cast

import pandas as pd
from pandas_booster.accessor import BoosterAccessor

AggFunc = Literal["sum", "mean", "min", "max", "count", "prod"]
StdVarAgg = Literal["std", "var"]


class TestMultiKeyGroupBySum:
    """Test multi-key groupby sum operations."""

    def test_two_keys_sum_f64_matches_pandas(self, large_multi_df):
        booster_result = cast(BoosterAccessor, large_multi_df.booster).groupby(
            ["k1", "k2"], "val_float", "sum"
        )
        pandas_result = large_multi_df.groupby(["k1", "k2"])["val_float"].sum()

        booster_sorted = booster_result.sort_index()
        pandas_sorted = pandas_result.sort_index()

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=False,
            rtol=1e-10,
        )

    def test_three_keys_sum_f64_matches_pandas(self, large_multi_df):
        booster_result = cast(BoosterAccessor, large_multi_df.booster).groupby(
            ["k1", "k2", "k3"], "val_float", "sum"
        )
        pandas_result = large_multi_df.groupby(["k1", "k2", "k3"])["val_float"].sum()

        booster_sorted = booster_result.sort_index()
        pandas_sorted = pandas_result.sort_index()

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=False,
            rtol=1e-10,
        )

    def test_two_keys_sum_i64_matches_pandas(self, large_multi_df):
        booster_result = cast(BoosterAccessor, large_multi_df.booster).groupby(
            ["k1", "k2"], "val_int", "sum"
        )
        pandas_result = large_multi_df.groupby(["k1", "k2"])["val_int"].sum()

        booster_sorted = booster_result.sort_index()
        pandas_sorted = pandas_result.sort_index()

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=True,
        )


class TestMultiKeyGroupByMean:
    """Test multi-key groupby mean operations."""

    def test_two_keys_mean_f64_matches_pandas(self, large_multi_df):
        booster_result = cast(BoosterAccessor, large_multi_df.booster).groupby(
            ["k1", "k2"], "val_float", "mean"
        )
        pandas_result = large_multi_df.groupby(["k1", "k2"])["val_float"].mean()

        booster_sorted = booster_result.sort_index()
        pandas_sorted = pandas_result.sort_index()

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=False,
            rtol=1e-10,
        )

    def test_three_keys_mean_f64_matches_pandas(self, large_multi_df):
        booster_result = cast(BoosterAccessor, large_multi_df.booster).groupby(
            ["k1", "k2", "k3"], "val_float", "mean"
        )
        pandas_result = large_multi_df.groupby(["k1", "k2", "k3"])["val_float"].mean()

        booster_sorted = booster_result.sort_index()
        pandas_sorted = pandas_result.sort_index()

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=False,
            rtol=1e-10,
        )


class TestMultiKeyGroupByMinMax:
    """Test multi-key groupby min/max operations."""

    def test_two_keys_min_f64_matches_pandas(self, large_multi_df):
        booster_result = cast(BoosterAccessor, large_multi_df.booster).groupby(
            ["k1", "k2"], "val_float", "min"
        )
        pandas_result = large_multi_df.groupby(["k1", "k2"])["val_float"].min()

        booster_sorted = booster_result.sort_index()
        pandas_sorted = pandas_result.sort_index()

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=False,
            rtol=1e-10,
        )

    def test_two_keys_max_f64_matches_pandas(self, large_multi_df):
        booster_result = cast(BoosterAccessor, large_multi_df.booster).groupby(
            ["k1", "k2"], "val_float", "max"
        )
        pandas_result = large_multi_df.groupby(["k1", "k2"])["val_float"].max()

        booster_sorted = booster_result.sort_index()
        pandas_sorted = pandas_result.sort_index()

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=False,
            rtol=1e-10,
        )

    def test_three_keys_min_i64_matches_pandas(self, large_multi_df):
        booster_result = cast(BoosterAccessor, large_multi_df.booster).groupby(
            ["k1", "k2", "k3"], "val_int", "min"
        )
        pandas_result = large_multi_df.groupby(["k1", "k2", "k3"])["val_int"].min()

        booster_sorted = booster_result.sort_index()
        pandas_sorted = pandas_result.sort_index()

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=True,
        )


class TestMultiKeyFallback:
    """Test fallback to Pandas for multi-key groupby."""

    def test_small_df_uses_fallback(self, small_multi_df):
        booster_result = cast(BoosterAccessor, small_multi_df.booster).groupby(
            ["k1", "k2"], "val_float", "sum"
        )
        pandas_result = small_multi_df.groupby(["k1", "k2"])["val_float"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )


class TestMultiKeyMultiIndex:
    """Test that multi-key results have proper MultiIndex structure."""

    def test_multiindex_names(self, large_multi_df):
        """MultiIndex should have correct level names."""
        result = cast(BoosterAccessor, large_multi_df.booster).groupby(
            ["k1", "k2"], "val_float", "sum"
        )

        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == ["k1", "k2"]

    def test_multiindex_three_levels(self, large_multi_df):
        """Three-key groupby should have 3-level MultiIndex."""
        result = cast(BoosterAccessor, large_multi_df.booster).groupby(
            ["k1", "k2", "k3"], "val_float", "sum"
        )

        assert isinstance(result.index, pd.MultiIndex)
        assert len(result.index.names) == 3
        assert result.index.names == ["k1", "k2", "k3"]

    def test_series_name_preserved(self, large_multi_df):
        """Result Series should have target column name."""
        result = cast(BoosterAccessor, large_multi_df.booster).groupby(
            ["k1", "k2"], "val_float", "sum"
        )

        assert result.name == "val_float"
