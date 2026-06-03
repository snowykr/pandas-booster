"""Multi-key first-seen ordering behavior tests."""

from typing import Literal, cast

import numpy as np
import pandas as pd
import pytest
from pandas_booster.accessor import BoosterAccessor

AggFunc = Literal["sum", "mean", "min", "max", "count", "prod"]
StdVarAgg = Literal["std", "var"]


class TestFirstSeenOrderSortFalse:
    def _make_ordered_multi_df(self) -> pd.DataFrame:
        n = 200_000
        # Four groups with deterministic first-seen order, spaced far apart to
        # stress first-seen tracking across parallel partitions/chunks.
        first_order = [(1, 10), (2, 20), (3, 30), (4, 40)]
        first_rows = [0, 60_000, 120_000, 180_000]

        k1 = np.empty(n, dtype=np.int64)
        k2 = np.empty(n, dtype=np.int64)

        # Default: only first group appears.
        k1[:] = first_order[0][0]
        k2[:] = first_order[0][1]

        def fill_segment(start: int, end: int, groups: list[tuple[int, int]]) -> None:
            if start >= end:
                return
            g1 = np.array([g[0] for g in groups], dtype=np.int64)
            g2 = np.array([g[1] for g in groups], dtype=np.int64)
            idx = np.arange(end - start) % len(groups)
            k1[start:end] = g1[idx]
            k2[start:end] = g2[idx]

        # Between first-seen points, only allow groups that have already been seen.
        fill_segment(first_rows[1] + 1, first_rows[2], first_order[:2])
        fill_segment(first_rows[2] + 1, first_rows[3], first_order[:3])
        fill_segment(first_rows[3] + 1, n, first_order[:4])

        # Stamp the first-seen rows (must not be overwritten).
        for (a, b), row in zip(first_order, first_rows):
            k1[row] = a
            k2[row] = b

        # Sanity: verify first-seen rows are exactly as intended.
        for (a, b), row in zip(first_order, first_rows):
            mask = (k1 == a) & (k2 == b)
            first_idx = int(np.flatnonzero(mask)[0])
            assert first_idx == row

        vals = np.random.random(n).astype(np.float64)

        # Force one group to be all-NaN to ensure the group is not dropped in the
        # first-seen path (matches the single-key coverage).
        vals[(k1 == 2) & (k2 == 20)] = np.nan
        return pd.DataFrame({"k1": k1, "k2": k2, "val": vals})

    @pytest.mark.parametrize("agg", ["sum", "mean", "min", "max", "count", "prod"])
    def test_booster_sort_false_preserves_first_seen_multi_key(self, agg: AggFunc):
        df = self._make_ordered_multi_df()
        booster_result = cast(BoosterAccessor, df.booster).groupby(
            ["k1", "k2"], "val", agg, sort=False
        )
        pandas_grouped = df.groupby(["k1", "k2"], sort=False)["val"]
        pandas_result = getattr(pandas_grouped, agg)()

        pd.testing.assert_series_equal(
            booster_result,
            pandas_result,
            check_exact=(agg == "count"),
            rtol=(0.0 if agg == "count" else 1e-10),
        )

    @pytest.mark.parametrize("agg", ["sum", "mean", "min", "max", "count", "prod"])
    def test_proxy_sort_false_preserves_first_seen_multi_key(self, agg: AggFunc):
        import pandas_booster

        df = self._make_ordered_multi_df()
        pandas_grouped = df.groupby(["k1", "k2"], sort=False)["val"]
        pandas_result = getattr(pandas_grouped, agg)()

        pandas_booster.activate()
        try:
            proxy_grouped = df.groupby(["k1", "k2"], sort=False)["val"]
            proxy_result = getattr(proxy_grouped, agg)()
        finally:
            pandas_booster.deactivate()

        pd.testing.assert_series_equal(
            proxy_result,
            pandas_result,
            check_exact=(agg == "count"),
            rtol=(0.0 if agg == "count" else 1e-10),
        )

    @pytest.mark.parametrize("agg", ["sum", "mean", "min", "max", "count", "prod"])
    def test_booster_sort_false_preserves_first_seen_multi_key_int(self, agg: AggFunc):
        df = self._make_ordered_multi_df()[["k1", "k2"]].copy()
        n = len(df)
        df["val"] = (np.arange(n, dtype=np.int64) % 17) - 8

        booster_result = cast(BoosterAccessor, df.booster).groupby(
            ["k1", "k2"], "val", agg, sort=False
        )
        pandas_grouped = df.groupby(["k1", "k2"], sort=False)["val"]
        pandas_result = getattr(pandas_grouped, agg)()

        expected_order = [(1, 10), (2, 20), (3, 30), (4, 40)]
        assert booster_result.index.tolist() == pandas_result.index.tolist() == expected_order

        pd.testing.assert_series_equal(
            booster_result,
            pandas_result,
            check_exact=(agg != "mean"),
            check_dtype=True,
            rtol=(1e-10 if agg == "mean" else 0.0),
        )

    def test_booster_sort_false_threshold_small_materialized_path_matches_pandas(self):
        # n_groups * n_keys (2) = 180_000 <= SMALL_DIRECT_THRESHOLD_ELEMS (200_000)
        # so radix first-seen reorder should materialize and return perm=None internally.
        n_groups = 90_000
        n_rows = 120_000
        rng = np.random.default_rng(2026)
        order = rng.permutation(n_groups).astype(np.int64)

        k1_unique = order
        k2_unique = order * 17 + 11
        v_unique = (order % 101).astype(np.float64)

        extra = n_rows - n_groups
        k1 = np.concatenate([k1_unique, k1_unique[:extra]])
        k2 = np.concatenate([k2_unique, k2_unique[:extra]])
        v = np.concatenate([v_unique, np.ones(extra, dtype=np.float64)])

        df = pd.DataFrame({"k1": k1, "k2": k2, "val": v})
        booster_result = cast(BoosterAccessor, df.booster).groupby(
            ["k1", "k2"], "val", "sum", sort=False
        )
        pandas_result = df.groupby(["k1", "k2"], sort=False)["val"].sum()

        assert len(booster_result) == n_groups
        pd.testing.assert_series_equal(booster_result, pandas_result, check_exact=False, rtol=1e-10)

    def test_booster_sort_false_threshold_large_perm_path_matches_pandas(self):
        # n_groups * n_keys (2) = 240_000 > SMALL_DIRECT_THRESHOLD_ELEMS (200_000)
        # so radix first-seen reorder should defer via perm=Some internally.
        n_groups = 120_000
        rng = np.random.default_rng(2027)
        order = rng.permutation(n_groups).astype(np.int64)

        k1 = order
        k2 = order * 19 + 5
        v = (order % 97).astype(np.float64)

        df = pd.DataFrame({"k1": k1, "k2": k2, "val": v})
        booster_result = cast(BoosterAccessor, df.booster).groupby(
            ["k1", "k2"], "val", "sum", sort=False
        )
        pandas_result = df.groupby(["k1", "k2"], sort=False)["val"].sum()

        assert len(booster_result) == n_groups
        pd.testing.assert_series_equal(booster_result, pandas_result, check_exact=False, rtol=1e-10)
