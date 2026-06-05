"""Extreme cardinality and negative-key behavior tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


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
