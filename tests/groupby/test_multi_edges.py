"""Multi-key edge case behavior tests."""

from typing import Literal, cast

import numpy as np
import pandas as pd
import pytest
from pandas_booster.accessor import BoosterAccessor

AggFunc = Literal["sum", "mean", "min", "max", "count", "prod"]
StdVarAgg = Literal["std", "var"]


class TestMultiKeyEdgeCases:
    """Test edge cases for multi-key groupby."""

    def test_single_key_in_list(self, large_multi_df):
        """Single key passed as list should work like string."""
        booster_list = cast(BoosterAccessor, large_multi_df.booster).groupby(
            ["k1"], "val_float", "sum"
        )
        booster_str = cast(BoosterAccessor, large_multi_df.booster).groupby(
            "k1", "val_float", "sum"
        )
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

    def test_too_many_keys_raises(self):
        """Requesting more than 10 keys should raise ValueError."""
        n = 200_000
        df = pd.DataFrame({f"k{i}": np.random.randint(0, 10, n) for i in range(11)})
        df["val"] = np.random.random(n)

        key_cols = [f"k{i}" for i in range(11)]
        with pytest.raises(ValueError, match="Too many key columns"):
            cast(BoosterAccessor, df.booster).groupby(key_cols, "val", "sum")

    def test_four_keys_inline_allocation(self):
        """Test with 4 keys."""
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

        booster_result = cast(BoosterAccessor, df.booster).groupby(
            ["k1", "k2", "k3", "k4"], "val", "sum"
        )
        pandas_result = df.groupby(["k1", "k2", "k3", "k4"])["val"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    def test_five_keys_heap_allocation(self):
        """Test with 5 keys."""
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

        booster_result = cast(BoosterAccessor, df.booster).groupby(
            ["k1", "k2", "k3", "k4", "k5"], "val", "sum"
        )
        pandas_result = df.groupby(["k1", "k2", "k3", "k4", "k5"])["val"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    def test_high_cardinality_multi_key(self):
        """Test with high cardinality (many unique groups)."""
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

        booster_result = cast(BoosterAccessor, df.booster).groupby(["k1", "k2"], "val", "sum")
        pandas_result = df.groupby(["k1", "k2"])["val"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    def test_negative_keys(self):
        """Test with negative key values."""
        np.random.seed(42)
        n = 200_000
        df = pd.DataFrame(
            {
                "k1": np.random.randint(-100, 100, n),
                "k2": np.random.randint(-50, 50, n),
                "val": np.random.random(n),
            }
        )

        booster_result = cast(BoosterAccessor, df.booster).groupby(["k1", "k2"], "val", "sum")
        pandas_result = df.groupby(["k1", "k2"])["val"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )
