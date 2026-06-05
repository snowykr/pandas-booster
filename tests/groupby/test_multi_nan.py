"""Multi-key NaN behavior tests."""

from typing import Literal, cast

import numpy as np
import pandas as pd
from pandas_booster.accessor import BoosterAccessor

AggFunc = Literal["sum", "mean", "min", "max", "count", "prod"]
StdVarAgg = Literal["std", "var"]


class TestMultiKeyNaNHandling:
    """Test NaN handling in multi-key groupby."""

    def test_sum_with_nan_values(self):
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

        booster_result = cast(BoosterAccessor, df.booster).groupby(["k1", "k2"], "val", "sum")
        pandas_result = df.groupby(["k1", "k2"])["val"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    def test_mean_with_nan_values(self):
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

        booster_result = cast(BoosterAccessor, df.booster).groupby(["k1", "k2"], "val", "mean")
        pandas_result = df.groupby(["k1", "k2"])["val"].mean()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )


class TestMultiKeyAllNaNGroup:
    """Test multi-key groupby with all-NaN groups."""

    def test_sum_all_nan_in_group(self):
        n = 200_000
        # First half of k1=0 groups have all NaN
        k1 = np.concatenate([np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)])
        k2 = np.random.randint(0, 10, n)
        val = np.concatenate([np.full(n // 4, np.nan), np.random.random(3 * n // 4)])

        df = pd.DataFrame({"k1": k1, "k2": k2, "val": val})

        booster_result = cast(BoosterAccessor, df.booster).groupby(["k1", "k2"], "val", "sum")
        pandas_result = df.groupby(["k1", "k2"])["val"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    def test_mean_all_nan_in_group(self):
        n = 200_000
        k1 = np.concatenate([np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)])
        k2 = np.random.randint(0, 10, n)
        val = np.concatenate([np.full(n // 4, np.nan), np.random.random(3 * n // 4)])

        df = pd.DataFrame({"k1": k1, "k2": k2, "val": val})

        booster_result = cast(BoosterAccessor, df.booster).groupby(["k1", "k2"], "val", "mean")
        pandas_result = df.groupby(["k1", "k2"])["val"].mean()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )
