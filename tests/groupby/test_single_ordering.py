"""Single-key first-seen ordering behavior tests."""

from typing import Literal, cast

import numpy as np
import pandas as pd
import pytest
from pandas_booster.accessor import BoosterAccessor

AggFunc = Literal["sum", "mean", "min", "max", "count", "prod"]


class TestFirstSeenOrderSingleKeySortFalse:
    def _make_df(self) -> pd.DataFrame:
        n = 200_000
        keys = np.empty(n, dtype=np.int64)
        vals = np.empty(n, dtype=np.float64)

        # Deterministic first-seen group order across the dataset.
        # Include a group whose values are all NaN to ensure the group
        # is not dropped in the first-seen path.
        first_order = [2, 99, 1, 5]
        first_rows = [0, 60_000, 120_000, 180_000]

        # Build keys in segments so each group's first appearance is far apart.
        # - [0, 60_000): only group 2
        # - [60_000, 120_000): groups 2, 99
        # - [120_000, 180_000): groups 2, 99, 1
        # - [180_000, n): groups 2, 99, 1, 5
        keys[:] = 2
        if first_rows[1] + 1 < first_rows[2]:
            seg = np.array([2, 99], dtype=np.int64)
            keys[first_rows[1] + 1 : first_rows[2]] = seg[
                np.arange(first_rows[2] - (first_rows[1] + 1)) % seg.size
            ]
        if first_rows[2] + 1 < first_rows[3]:
            seg = np.array([2, 99, 1], dtype=np.int64)
            keys[first_rows[2] + 1 : first_rows[3]] = seg[
                np.arange(first_rows[3] - (first_rows[2] + 1)) % seg.size
            ]
        if first_rows[3] + 1 < n:
            seg = np.array(first_order, dtype=np.int64)
            keys[first_rows[3] + 1 :] = seg[np.arange(n - (first_rows[3] + 1)) % seg.size]

        # Stamp first-seen rows (must not be overwritten).
        for k, row in zip(first_order, first_rows):
            keys[row] = k

        # Sanity: verify first-seen rows are exactly as intended.
        for k, row in zip(first_order, first_rows):
            first_idx = int(np.flatnonzero(keys == k)[0])
            assert first_idx == row

        vals[:] = np.random.random(n).astype(np.float64)

        # Force group 99 to be all-NaN.
        vals[keys == 99] = np.nan

        return pd.DataFrame({"key": keys, "val": vals})

    @pytest.mark.parametrize("agg", ["sum", "mean", "min", "max", "count", "prod"])
    def test_sort_false_preserves_first_seen_float(self, agg: AggFunc):
        df = self._make_df()
        booster_result = cast(BoosterAccessor, df.booster).groupby("key", "val", agg, sort=False)
        pandas_result = getattr(df.groupby("key", sort=False)["val"], agg)()

        # Order must match Pandas appearance order.
        assert booster_result.index.tolist() == pandas_result.index.tolist() == [2, 99, 1, 5]

        pd.testing.assert_series_equal(
            booster_result,
            pandas_result,
            check_exact=(agg == "count"),
            rtol=(0.0 if agg == "count" else 1e-10),
        )

    @pytest.mark.parametrize("agg", ["sum", "mean", "min", "max", "count", "prod"])
    def test_sort_false_preserves_first_seen_int(self, agg: AggFunc):
        n = 200_000
        keys = np.array([2, 99, 1, 5], dtype=np.int64)
        keys = np.tile(keys, (n + len(keys) - 1) // len(keys))[:n]
        vals = (np.arange(n, dtype=np.int64) % 7) - 3
        df = pd.DataFrame({"key": keys, "val": vals})

        booster_result = cast(BoosterAccessor, df.booster).groupby("key", "val", agg, sort=False)
        pandas_result = getattr(df.groupby("key", sort=False)["val"], agg)()

        assert booster_result.index.tolist() == pandas_result.index.tolist() == [2, 99, 1, 5]
        pd.testing.assert_series_equal(
            booster_result,
            pandas_result,
            check_exact=(agg != "mean"),
            check_dtype=True,
            rtol=(1e-10 if agg == "mean" else 0.0),
        )

    def test_mean_all_nan_group(self):
        n = 200_000
        df = pd.DataFrame(
            {
                "key": np.concatenate([np.ones(n // 2, dtype=int), np.full(n // 2, 2, dtype=int)]),
                "val": np.concatenate([np.full(n // 2, np.nan), np.random.random(n // 2)]),
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

    def test_min_all_nan_group(self):
        n = 200_000
        df = pd.DataFrame(
            {
                "key": np.concatenate([np.ones(n // 2, dtype=int), np.full(n // 2, 2, dtype=int)]),
                "val": np.concatenate([np.full(n // 2, np.nan), np.random.random(n // 2)]),
            }
        )

        booster_result = cast(BoosterAccessor, df.booster).groupby("key", "val", "min")
        pandas_result = df.groupby("key")["val"].min()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    def test_max_all_nan_group(self):
        n = 200_000
        df = pd.DataFrame(
            {
                "key": np.concatenate([np.ones(n // 2, dtype=int), np.full(n // 2, 2, dtype=int)]),
                "val": np.concatenate([np.full(n // 2, np.nan), np.random.random(n // 2)]),
            }
        )

        booster_result = cast(BoosterAccessor, df.booster).groupby("key", "val", "max")
        pandas_result = df.groupby("key")["val"].max()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )


class TestFirstSeenOrderSortFalse:
    def _make_ordered_df(self) -> pd.DataFrame:
        # Must be above fallback threshold to force acceleration.
        np.random.seed(123)
        n = 200_000
        keys = np.empty(n, dtype=np.int64)
        pattern = np.array([5, 1, 3, 0], dtype=np.int64)
        keys[: pattern.size] = pattern
        # Fill remaining with the same key set (no new groups introduced).
        keys[pattern.size :] = np.tile(
            pattern, (n - pattern.size + pattern.size - 1) // pattern.size
        )[: n - pattern.size]
        vals = np.random.random(n).astype(np.float64)
        mask = np.random.random(n) < 0.05
        vals[mask] = np.nan
        return pd.DataFrame({"key": keys, "val": vals})

    @pytest.mark.parametrize("agg", ["sum", "mean", "min", "max", "count", "prod"])
    def test_booster_sort_false_preserves_first_seen_single_key(self, agg: AggFunc):
        df = self._make_ordered_df()
        booster_result = cast(BoosterAccessor, df.booster).groupby("key", "val", agg, sort=False)
        pandas_grouped = df.groupby("key", sort=False)["val"]
        pandas_result = getattr(pandas_grouped, agg)()

        pd.testing.assert_series_equal(
            booster_result,
            pandas_result,
            check_exact=(agg == "count"),
            rtol=(0.0 if agg == "count" else 1e-10),
        )

    @pytest.mark.parametrize("agg", ["sum", "mean", "min", "max", "count", "prod"])
    def test_proxy_sort_false_preserves_first_seen_single_key(self, agg: AggFunc):
        import pandas_booster

        df = self._make_ordered_df()
        pandas_grouped = df.groupby("key", sort=False)["val"]
        pandas_result = getattr(pandas_grouped, agg)()

        pandas_booster.activate()
        try:
            proxy_grouped = df.groupby("key", sort=False)["val"]
            proxy_result = getattr(proxy_grouped, agg)()
        finally:
            pandas_booster.deactivate()

        pd.testing.assert_series_equal(
            proxy_result,
            pandas_result,
            check_exact=(agg == "count"),
            rtol=(0.0 if agg == "count" else 1e-10),
        )
