"""Edge case tests for pandas-booster.

Tests covering numerical extremes, empty data, and boundary conditions
that are critical for ensuring library robustness.
"""

import numpy as np
import pandas as pd
import pytest


class TestEmptyData:
    """Tests for empty DataFrame and edge-size scenarios."""

    def test_empty_dataframe_uses_fallback(self):
        """Empty DataFrame should use fallback and return empty Series."""
        import pandas_booster  # noqa: F401

        df = pd.DataFrame(
            {"key": pd.array([], dtype="int64"), "val": pd.array([], dtype="float64")}
        )

        result = df.booster.groupby("key", "val", "sum")
        expected = df.groupby("key")["val"].sum()

        pd.testing.assert_series_equal(result, expected)
        assert len(result) == 0

    def test_single_row_uses_fallback(self):
        """Single row should use fallback (below threshold)."""
        import pandas_booster  # noqa: F401

        df = pd.DataFrame({"key": [1], "val": [42.0]})

        result = df.booster.groupby("key", "val", "sum")
        expected = df.groupby("key")["val"].sum()

        pd.testing.assert_series_equal(result, expected)

    def test_exactly_at_threshold_boundary(self):
        """Test behavior exactly at the fallback threshold boundary."""
        import pandas_booster  # noqa: F401

        # Just below threshold (should use fallback)
        n = 99_999
        df_below = pd.DataFrame(
            {"key": np.random.randint(0, 100, size=n), "val": np.random.random(size=n)}
        )
        result_below = df_below.booster.groupby("key", "val", "sum")
        expected_below = df_below.groupby("key")["val"].sum()
        pd.testing.assert_series_equal(
            result_below.sort_index(), expected_below.sort_index(), check_exact=False, rtol=1e-10
        )

        # At threshold (should use Rust)
        n = 100_000
        df_at = pd.DataFrame(
            {"key": np.random.randint(0, 100, size=n), "val": np.random.random(size=n)}
        )
        result_at = df_at.booster.groupby("key", "val", "sum")
        expected_at = df_at.groupby("key")["val"].sum()
        pd.testing.assert_series_equal(
            result_at.sort_index(), expected_at.sort_index(), check_exact=False, rtol=1e-10
        )


class TestInfiniteValues:
    """Tests for inf and -inf handling in value columns."""

    @pytest.fixture
    def df_with_inf(self):
        """Create DataFrame with infinite values."""
        np.random.seed(42)
        n = 200_000
        values = np.random.random(n) * 100
        # Insert some infinities
        values[100] = np.inf
        values[200] = -np.inf
        values[300] = np.inf
        values[400] = -np.inf
        return pd.DataFrame({"key": np.random.randint(0, 100, size=n), "val": values})

    def test_sum_with_inf(self, df_with_inf):
        """Sum with inf should match Pandas behavior."""
        import pandas_booster  # noqa: F401

        result = df_with_inf.booster.groupby("key", "val", "sum")
        expected = df_with_inf.groupby("key")["val"].sum()

        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_exact=False, rtol=1e-10
        )

    def test_mean_with_inf(self, df_with_inf):
        """Mean with inf should match Pandas behavior."""
        import pandas_booster  # noqa: F401

        result = df_with_inf.booster.groupby("key", "val", "mean")
        expected = df_with_inf.groupby("key")["val"].mean()

        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_exact=False, rtol=1e-10
        )

    def test_min_with_inf(self, df_with_inf):
        """Min with inf should match Pandas behavior."""
        import pandas_booster  # noqa: F401

        result = df_with_inf.booster.groupby("key", "val", "min")
        expected = df_with_inf.groupby("key")["val"].min()

        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_exact=False, rtol=1e-10
        )

    def test_max_with_inf(self, df_with_inf):
        """Max with inf should match Pandas behavior."""
        import pandas_booster  # noqa: F401

        result = df_with_inf.booster.groupby("key", "val", "max")
        expected = df_with_inf.groupby("key")["val"].max()

        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_exact=False, rtol=1e-10
        )

    def test_all_inf_group(self):
        """Group with only inf values."""
        import pandas_booster  # noqa: F401

        n = 200_000
        # Group 1 has only +inf, Group 2 has only -inf, Group 3 has normal values
        keys = np.concatenate(
            [
                np.ones(n // 3, dtype=int),
                np.full(n // 3, 2, dtype=int),
                np.full(n - 2 * (n // 3), 3, dtype=int),
            ]
        )
        values = np.concatenate(
            [
                np.full(n // 3, np.inf),
                np.full(n // 3, -np.inf),
                np.random.random(n - 2 * (n // 3)),
            ]
        )
        df = pd.DataFrame({"key": keys, "val": values})

        for agg in ["sum", "mean", "min", "max"]:
            result = df.booster.groupby("key", "val", agg)
            expected = getattr(df.groupby("key")["val"], agg)()
            pd.testing.assert_series_equal(
                result.sort_index(), expected.sort_index(), check_exact=False, rtol=1e-10
            )


class TestIntegerOverflow:
    """Tests for integer overflow behavior parity with pandas."""

    def test_large_i64_sum_no_overflow(self):
        """Large i64 sums should match pandas overflow/wrap semantics."""
        import pandas_booster  # noqa: F401

        n = 200_000
        # Large values that would overflow i64 when summed
        large_val = 2**62  # Close to i64::MAX / 2
        df = pd.DataFrame(
            {
                "key": np.ones(n, dtype=np.int64),
                "val": np.full(n, large_val, dtype=np.int64),
            }
        )

        result = df.booster.groupby("key", "val", "sum")
        expected = df.groupby("key")["val"].sum()

        pd.testing.assert_series_equal(
            result.sort_index(),
            expected.sort_index(),
            check_exact=True,
        )

    def test_mixed_sign_large_values(self):
        """Large positive and negative values should cancel correctly."""
        import pandas_booster  # noqa: F401

        n = 200_000
        large_val = 2**62
        # Alternating large positive and negative values
        values = np.where(np.arange(n) % 2 == 0, large_val, -large_val).astype(np.int64)
        df = pd.DataFrame(
            {
                "key": np.ones(n, dtype=np.int64),
                "val": values,
            }
        )

        result = df.booster.groupby("key", "val", "sum")
        expected = df.groupby("key")["val"].sum()

        pd.testing.assert_series_equal(
            result.sort_index(),
            expected.sort_index(),
            check_exact=True,
        )


class TestExtremeCardinality:
    """Tests for extreme group cardinality scenarios."""

    def test_single_group_large_dataset(self):
        """All rows in a single group."""
        import pandas_booster  # noqa: F401

        np.random.seed(42)
        n = 200_000
        df = pd.DataFrame(
            {
                "key": np.ones(n, dtype=np.int64),
                "val": np.random.random(n),
            }
        )

        result = df.booster.groupby("key", "val", "sum")
        expected = df.groupby("key")["val"].sum()

        assert len(result) == 1
        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_exact=False, rtol=1e-10
        )

    def test_every_row_unique_key(self):
        """Every row is its own group (maximum cardinality)."""
        import pandas_booster  # noqa: F401

        n = 150_000  # Just above threshold
        df = pd.DataFrame(
            {
                "key": np.arange(n, dtype=np.int64),
                "val": np.random.random(n),
            }
        )

        result = df.booster.groupby("key", "val", "sum")
        expected = df.groupby("key")["val"].sum()

        assert len(result) == n
        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_exact=False, rtol=1e-10
        )

    def test_two_groups_only(self):
        """Only two groups in entire dataset."""
        import pandas_booster  # noqa: F401

        np.random.seed(42)
        n = 200_000
        df = pd.DataFrame(
            {
                "key": np.where(np.arange(n) < n // 2, 0, 1).astype(np.int64),
                "val": np.random.random(n),
            }
        )

        result = df.booster.groupby("key", "val", "mean")
        expected = df.groupby("key")["val"].mean()

        assert len(result) == 2
        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_exact=False, rtol=1e-10
        )


class TestNegativeKeys:
    """Tests for negative integer keys."""

    def test_negative_keys(self):
        """Negative integer keys should work correctly."""
        import pandas_booster  # noqa: F401

        np.random.seed(42)
        n = 200_000
        df = pd.DataFrame(
            {
                "key": np.random.randint(-1000, 1000, size=n),
                "val": np.random.random(n),
            }
        )

        for agg in ["sum", "mean", "min", "max"]:
            result = df.booster.groupby("key", "val", agg)
            expected = getattr(df.groupby("key")["val"], agg)()
            pd.testing.assert_series_equal(
                result.sort_index(), expected.sort_index(), check_exact=False, rtol=1e-10
            )

    def test_extreme_key_values(self):
        """Keys near i64 min/max bounds."""
        import pandas_booster  # noqa: F401

        n = 200_000
        # Use keys near the boundaries of i64
        extreme_keys = np.array([-(2**62), -(2**62) + 1, 0, 2**62 - 1, 2**62], dtype=np.int64)
        keys = np.tile(extreme_keys, n // 5)
        df = pd.DataFrame(
            {
                "key": keys[:n],
                "val": np.random.random(n),
            }
        )

        result = df.booster.groupby("key", "val", "sum")
        expected = df.groupby("key")["val"].sum()

        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_exact=False, rtol=1e-10
        )


class TestMixedNaNAndInf:
    """Tests combining NaN and inf values."""

    def test_nan_and_inf_mixed(self):
        """Mix of NaN and inf values in same column."""
        import pandas_booster  # noqa: F401

        np.random.seed(42)
        n = 200_000
        values = np.random.random(n) * 100
        # Scatter NaN, +inf, -inf
        nan_mask = np.random.random(n) < 0.05
        inf_mask = np.random.random(n) < 0.02
        ninf_mask = np.random.random(n) < 0.02

        values = np.where(nan_mask, np.nan, values)
        values = np.where(inf_mask, np.inf, values)
        values = np.where(ninf_mask, -np.inf, values)

        df = pd.DataFrame(
            {
                "key": np.random.randint(0, 100, size=n),
                "val": values,
            }
        )

        for agg in ["sum", "mean", "min", "max"]:
            result = df.booster.groupby("key", "val", agg)
            expected = getattr(df.groupby("key")["val"], agg)()
            pd.testing.assert_series_equal(
                result.sort_index(), expected.sort_index(), check_exact=False, rtol=1e-10
            )


class TestDtypeFallback:
    """Tests for dtype-based fallback scenarios."""

    def test_string_key_uses_fallback(self):
        """String key column should use Pandas fallback."""
        import pandas_booster  # noqa: F401

        n = 200_000
        df = pd.DataFrame(
            {
                "key": np.array([f"key_{i % 100}" for i in range(n)]),
                "val": np.random.random(n),
            }
        )

        result = df.booster.groupby("key", "val", "sum")
        expected = df.groupby("key")["val"].sum()

        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_exact=False, rtol=1e-10
        )

    def test_float_key_uses_fallback(self):
        """Float key column should use Pandas fallback."""
        import pandas_booster  # noqa: F401

        n = 200_000
        df = pd.DataFrame(
            {
                "key": np.random.random(n) * 100,  # Float keys
                "val": np.random.random(n),
            }
        )

        result = df.booster.groupby("key", "val", "sum")
        expected = df.groupby("key")["val"].sum()

        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_exact=False, rtol=1e-10
        )

    def test_string_value_uses_fallback(self):
        """String value column should use Pandas fallback."""
        import pandas_booster  # noqa: F401

        n = 200_000
        df = pd.DataFrame(
            {
                "key": np.random.randint(0, 100, size=n),
                "val": np.array([f"val_{i}" for i in range(n)]),
            }
        )

        # String columns can't use sum/mean/min/max meaningfully, but we test fallback behavior
        # The accessor should fall back to Pandas which will handle it appropriately
        result = df.booster.groupby("key", "val", "min")
        expected = df.groupby("key")["val"].min()

        pd.testing.assert_series_equal(result.sort_index(), expected.sort_index())


class TestAllOperationsConsistency:
    """Ensure all operations work consistently across edge cases."""

    @pytest.fixture
    def stress_df(self):
        """Create a stress-test DataFrame with various edge values."""
        np.random.seed(42)
        n = 200_000
        values = np.random.random(n) * 1000 - 500  # Range: -500 to 500

        # Inject edge cases
        values[::100] = np.nan  # 1% NaN
        values[::500] = np.inf  # 0.2% +inf
        values[::700] = -np.inf  # ~0.14% -inf
        values[::1000] = 0.0  # 0.1% zeros

        return pd.DataFrame(
            {
                "key": np.random.randint(-50, 50, size=n),
                "val": values,
            }
        )

    @pytest.mark.parametrize("agg", ["sum", "mean", "min", "max"])
    def test_operation_consistency(self, stress_df, agg):
        """All operations should match Pandas on stress DataFrame."""
        import pandas_booster  # noqa: F401

        result = stress_df.booster.groupby("key", "val", agg)
        expected = getattr(stress_df.groupby("key")["val"], agg)()

        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_exact=False, rtol=1e-10
        )
