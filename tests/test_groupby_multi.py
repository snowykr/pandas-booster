"""Tests for multi-column groupby functionality.

These tests verify that the multi-key groupby operations produce results
that match native Pandas groupby behavior.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def large_multi_df():
    """Large DataFrame for multi-key groupby tests (above fallback threshold)."""
    np.random.seed(42)
    n = 500_000
    return pd.DataFrame(
        {
            "k1": np.random.randint(0, 100, size=n),
            "k2": np.random.randint(0, 50, size=n),
            "k3": np.random.randint(0, 20, size=n),
            "val_int": np.random.randint(0, 100, size=n),
            "val_float": np.random.random(size=n) * 100,
        }
    )


@pytest.fixture
def small_multi_df():
    """Small DataFrame that should trigger fallback."""
    return pd.DataFrame(
        {
            "k1": [1, 1, 2, 2, 1],
            "k2": [10, 20, 10, 20, 10],
            "val_int": [10, 20, 30, 40, 50],
            "val_float": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


class TestMultiKeyGroupBySum:
    """Test multi-key groupby sum operations."""

    def test_two_keys_sum_f64_matches_pandas(self, large_multi_df):
        import pandas_booster

        booster_result = large_multi_df.booster.groupby(["k1", "k2"], "val_float", "sum")
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
        import pandas_booster

        booster_result = large_multi_df.booster.groupby(["k1", "k2", "k3"], "val_float", "sum")
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
        import pandas_booster

        booster_result = large_multi_df.booster.groupby(["k1", "k2"], "val_int", "sum")
        pandas_result = large_multi_df.groupby(["k1", "k2"])["val_int"].sum()

        booster_sorted = booster_result.sort_index().astype(float)
        pandas_sorted = pandas_result.sort_index().astype(float)

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=False,
            rtol=1e-10,
        )


class TestMultiKeyGroupByMean:
    """Test multi-key groupby mean operations."""

    def test_two_keys_mean_f64_matches_pandas(self, large_multi_df):
        import pandas_booster

        booster_result = large_multi_df.booster.groupby(["k1", "k2"], "val_float", "mean")
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
        import pandas_booster

        booster_result = large_multi_df.booster.groupby(["k1", "k2", "k3"], "val_float", "mean")
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
        import pandas_booster

        booster_result = large_multi_df.booster.groupby(["k1", "k2"], "val_float", "min")
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
        import pandas_booster

        booster_result = large_multi_df.booster.groupby(["k1", "k2"], "val_float", "max")
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
        import pandas_booster

        booster_result = large_multi_df.booster.groupby(["k1", "k2", "k3"], "val_int", "min")
        pandas_result = large_multi_df.groupby(["k1", "k2", "k3"])["val_int"].min()

        booster_sorted = booster_result.sort_index().astype(float)
        pandas_sorted = pandas_result.sort_index().astype(float)

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=False,
            rtol=1e-10,
        )


class TestMultiKeyFallback:
    """Test fallback to Pandas for multi-key groupby."""

    def test_small_df_uses_fallback(self, small_multi_df):
        import pandas_booster

        booster_result = small_multi_df.booster.groupby(["k1", "k2"], "val_float", "sum")
        pandas_result = small_multi_df.groupby(["k1", "k2"])["val_float"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
        )

    def test_non_integer_key_uses_fallback(self):
        import pandas_booster

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
        booster_result = df.booster.groupby(["k1", "k2"], "val", "sum")
        pandas_result = df.groupby(["k1", "k2"])["val"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    def test_nullable_key_uses_fallback(self):
        import pandas_booster

        np.random.seed(42)
        n = 200_000
        df = pd.DataFrame(
            {
                "k1": pd.array(np.random.randint(0, 100, n), dtype="Int64"),
                "k2": np.random.randint(0, 50, n),
                "val": np.random.random(n),
            }
        )
        df.loc[0, "k1"] = pd.NA  # Make it nullable

        booster_result = df.booster.groupby(["k1", "k2"], "val", "sum")
        pandas_result = df.groupby(["k1", "k2"])["val"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )


class TestMultiKeyNaNHandling:
    """Test NaN handling in multi-key groupby."""

    def test_sum_with_nan_values(self):
        import pandas_booster

        np.random.seed(123)
        n = 200_000
        values = np.random.random(n)
        mask = np.random.random(n) < 0.1
        df = pd.DataFrame(
            {
                "k1": np.random.randint(0, 50, n),
                "k2": np.random.randint(0, 50, n),
                "val": np.where(mask, np.nan, values),
            }
        )

        booster_result = df.booster.groupby(["k1", "k2"], "val", "sum")
        pandas_result = df.groupby(["k1", "k2"])["val"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    def test_mean_with_nan_values(self):
        import pandas_booster

        np.random.seed(456)
        n = 200_000
        values = np.random.random(n)
        mask = np.random.random(n) < 0.1
        df = pd.DataFrame(
            {
                "k1": np.random.randint(0, 50, n),
                "k2": np.random.randint(0, 50, n),
                "val": np.where(mask, np.nan, values),
            }
        )

        booster_result = df.booster.groupby(["k1", "k2"], "val", "mean")
        pandas_result = df.groupby(["k1", "k2"])["val"].mean()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )


class TestMultiKeyEdgeCases:
    """Test edge cases for multi-key groupby."""

    def test_single_key_in_list(self, large_multi_df):
        """Single key passed as list should work like string."""
        import pandas_booster

        booster_list = large_multi_df.booster.groupby(["k1"], "val_float", "sum")
        booster_str = large_multi_df.booster.groupby("k1", "val_float", "sum")
        pandas_result = large_multi_df.groupby("k1")["val_float"].sum()

        # Both should match pandas
        pd.testing.assert_series_equal(
            booster_list.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )
        pd.testing.assert_series_equal(
            booster_str.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    def test_too_many_keys_raises(self, large_multi_df):
        """Requesting more than 10 keys should raise ValueError."""
        import pandas_booster

        n = 200_000
        df = pd.DataFrame({f"k{i}": np.random.randint(0, 10, n) for i in range(11)})
        df["val"] = np.random.random(n)

        key_cols = [f"k{i}" for i in range(11)]
        with pytest.raises(ValueError, match="Too many key columns"):
            df.booster.groupby(key_cols, "val", "sum")

    def test_four_keys_inline_allocation(self):
        """Test with exactly 4 keys (SmallVec inline limit)."""
        import pandas_booster

        np.random.seed(42)
        n = 200_000
        df = pd.DataFrame(
            {
                "k1": np.random.randint(0, 20, n),
                "k2": np.random.randint(0, 20, n),
                "k3": np.random.randint(0, 20, n),
                "k4": np.random.randint(0, 20, n),
                "val": np.random.random(n),
            }
        )

        booster_result = df.booster.groupby(["k1", "k2", "k3", "k4"], "val", "sum")
        pandas_result = df.groupby(["k1", "k2", "k3", "k4"])["val"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    def test_five_keys_heap_allocation(self):
        """Test with 5 keys (beyond SmallVec inline limit)."""
        import pandas_booster

        np.random.seed(42)
        n = 200_000
        df = pd.DataFrame(
            {
                "k1": np.random.randint(0, 10, n),
                "k2": np.random.randint(0, 10, n),
                "k3": np.random.randint(0, 10, n),
                "k4": np.random.randint(0, 10, n),
                "k5": np.random.randint(0, 10, n),
                "val": np.random.random(n),
            }
        )

        booster_result = df.booster.groupby(["k1", "k2", "k3", "k4", "k5"], "val", "sum")
        pandas_result = df.groupby(["k1", "k2", "k3", "k4", "k5"])["val"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    def test_high_cardinality_multi_key(self):
        """Test with high cardinality (many unique groups)."""
        import pandas_booster

        np.random.seed(42)
        n = 200_000
        # Create sparse combinations (many unique groups)
        df = pd.DataFrame(
            {
                "k1": np.random.randint(0, 1000, n),
                "k2": np.random.randint(0, 1000, n),
                "val": np.random.random(n),
            }
        )

        booster_result = df.booster.groupby(["k1", "k2"], "val", "sum")
        pandas_result = df.groupby(["k1", "k2"])["val"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    def test_negative_keys(self):
        """Test with negative key values."""
        import pandas_booster

        np.random.seed(42)
        n = 200_000
        df = pd.DataFrame(
            {
                "k1": np.random.randint(-100, 100, n),
                "k2": np.random.randint(-50, 50, n),
                "val": np.random.random(n),
            }
        )

        booster_result = df.booster.groupby(["k1", "k2"], "val", "sum")
        pandas_result = df.groupby(["k1", "k2"])["val"].sum()

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
        import pandas_booster

        result = large_multi_df.booster.groupby(["k1", "k2"], "val_float", "sum")

        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == ["k1", "k2"]

    def test_multiindex_three_levels(self, large_multi_df):
        """Three-key groupby should have 3-level MultiIndex."""
        import pandas_booster

        result = large_multi_df.booster.groupby(["k1", "k2", "k3"], "val_float", "sum")

        assert isinstance(result.index, pd.MultiIndex)
        assert len(result.index.names) == 3
        assert result.index.names == ["k1", "k2", "k3"]

    def test_series_name_preserved(self, large_multi_df):
        """Result Series should have target column name."""
        import pandas_booster

        result = large_multi_df.booster.groupby(["k1", "k2"], "val_float", "sum")

        assert result.name == "val_float"


class TestMultiKeyAllNaNGroup:
    """Test multi-key groupby with all-NaN groups."""

    def test_sum_all_nan_in_group(self):
        import pandas_booster

        n = 200_000
        # First half of k1=0 groups have all NaN
        k1 = np.concatenate([np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)])
        k2 = np.random.randint(0, 10, n)
        val = np.concatenate([np.full(n // 4, np.nan), np.random.random(3 * n // 4)])

        df = pd.DataFrame({"k1": k1, "k2": k2, "val": val})

        booster_result = df.booster.groupby(["k1", "k2"], "val", "sum")
        pandas_result = df.groupby(["k1", "k2"])["val"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    def test_mean_all_nan_in_group(self):
        import pandas_booster

        n = 200_000
        k1 = np.concatenate([np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)])
        k2 = np.random.randint(0, 10, n)
        val = np.concatenate([np.full(n // 4, np.nan), np.random.random(3 * n // 4)])

        df = pd.DataFrame({"k1": k1, "k2": k2, "val": val})

        booster_result = df.booster.groupby(["k1", "k2"], "val", "mean")
        pandas_result = df.groupby(["k1", "k2"])["val"].mean()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )
