"""Infinite value, overflow, and mixed NaN/Inf behavior tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


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

    def test_large_i64_sum_wraps_like_pandas(self):
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
