"""Single-key high-cardinality behavior tests."""

from typing import Literal, cast

import numpy as np
import pandas as pd
from pandas_booster.accessor import BoosterAccessor

AggFunc = Literal["sum", "mean", "min", "max", "count", "prod"]


class TestHighCardinality:
    def test_high_cardinality_unique_keys(self):
        n = 150_000
        df = pd.DataFrame(
            {
                "key": np.arange(n),
                "val": np.ones(n),
            }
        )

        booster_result = cast(BoosterAccessor, df.booster).groupby("key", "val", "sum")
        pandas_result = df.groupby("key")["val"].sum()

        assert len(booster_result) == n
        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    def test_high_cardinality_mean(self):
        np.random.seed(123)
        n = 200_000
        n_groups = 50_000
        df = pd.DataFrame(
            {
                "key": np.random.randint(0, n_groups, size=n),
                "val": np.random.random(size=n),
            }
        )

        booster_result = cast(BoosterAccessor, df.booster).groupby("key", "val", "mean")
        pandas_result = df.groupby("key")["val"].mean()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )
