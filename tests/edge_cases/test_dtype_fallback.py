"""Unsupported dtype fallback behavior tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


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
