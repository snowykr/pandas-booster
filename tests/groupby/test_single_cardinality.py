"""Single-key high-cardinality behavior tests."""

from typing import Literal, cast

import numpy as np
import pandas as pd
import pytest
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

    @pytest.mark.parametrize("agg", ["sum", "mean", "min", "max", "count"])
    def test_high_uniqueness_sort_false_float_targets_match_pandas(self, agg: AggFunc):
        import pandas_booster._rust as rust

        n_rows = max(rust.get_fallback_threshold() + 1, 16_384)
        unique_count = 4_096
        first_seen = np.concatenate(
            (
                np.arange(unique_count // 2, unique_count, dtype=np.int64),
                np.arange(0, unique_count // 2, dtype=np.int64),
            )
        )
        keys = np.resize(first_seen, n_rows)
        values = ((np.arange(n_rows, dtype=np.float64) % 97.0) - 48.0) / 8.0
        values[np.arange(n_rows) % 251 == 0] = np.nan
        df = pd.DataFrame({"key": keys, "val": values})

        booster_result = cast(BoosterAccessor, df.booster).groupby("key", "val", agg, sort=False)
        pandas_result = getattr(df.groupby("key", sort=False)["val"], agg)()

        assert booster_result.index.tolist() == pandas_result.index.tolist()
        assert booster_result.index.tolist()[:unique_count] == first_seen.tolist()
        pd.testing.assert_series_equal(
            booster_result,
            pandas_result,
            check_exact=(agg in {"min", "max", "count"}),
            rtol=(0.0 if agg in {"min", "max", "count"} else 1e-10),
        )
