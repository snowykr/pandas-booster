"""Multi-key sort parameter behavior tests."""

from typing import Literal, cast

import numpy as np
import pandas as pd
from pandas_booster.accessor import BoosterAccessor

AggFunc = Literal["sum", "mean", "min", "max", "count", "prod"]
StdVarAgg = Literal["std", "var"]


class TestSortParameter:
    """Test sort=True/False parameter behavior."""

    def test_sort_true_matches_pandas_order(self, large_multi_df):
        booster_result = cast(BoosterAccessor, large_multi_df.booster).groupby(
            ["k1", "k2"], "val_float", "sum", sort=True
        )
        pandas_result = large_multi_df.groupby(["k1", "k2"], sort=True)["val_float"].sum()

        pd.testing.assert_series_equal(
            booster_result,
            pandas_result,
            check_exact=False,
            rtol=1e-10,
        )

    def test_sort_false_values_match_pandas(self, large_multi_df):
        booster_result = cast(BoosterAccessor, large_multi_df.booster).groupby(
            ["k1", "k2"], "val_float", "sum", sort=False
        )
        pandas_result = large_multi_df.groupby(["k1", "k2"], sort=False)["val_float"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    def test_sort_false_single_key(self):
        np.random.seed(42)
        n = 200_000
        df = pd.DataFrame({"key": np.random.randint(0, 1000, n), "val": np.random.random(n)})

        booster_result = cast(BoosterAccessor, df.booster).groupby("key", "val", "sum", sort=False)
        pandas_result = df.groupby("key", sort=False)["val"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    def test_sort_true_is_default(self, large_multi_df):
        result_default = cast(BoosterAccessor, large_multi_df.booster).groupby(
            ["k1", "k2"], "val_float", "sum"
        )
        result_explicit = cast(BoosterAccessor, large_multi_df.booster).groupby(
            ["k1", "k2"], "val_float", "sum", sort=True
        )

        pd.testing.assert_series_equal(result_default, result_explicit)

    def test_sort_parameter_three_keys(self, large_multi_df):
        booster_sorted = cast(BoosterAccessor, large_multi_df.booster).groupby(
            ["k1", "k2", "k3"], "val_float", "sum", sort=True
        )
        pandas_sorted = large_multi_df.groupby(["k1", "k2", "k3"], sort=True)["val_float"].sum()

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=False,
            rtol=1e-10,
        )

    def test_non_integer_key_uses_fallback(self):
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
        booster_result = cast(BoosterAccessor, df.booster).groupby(["k1", "k2"], "val", "sum")
        pandas_result = df.groupby(["k1", "k2"])["val"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    def test_nullable_key_uses_fallback(self):
        np.random.seed(42)
        n = 200_000
        df = pd.DataFrame(
            {
                "k1": pd.array(np.random.randint(0, 100, n).tolist(), dtype="Int64"),
                "k2": np.random.randint(0, 50, n),
                "val": np.random.random(n),
            }
        )
        df.loc[0, "k1"] = pd.NA  # Make it nullable

        booster_result = cast(BoosterAccessor, df.booster).groupby(["k1", "k2"], "val", "sum")
        pandas_result = df.groupby(["k1", "k2"])["val"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )
