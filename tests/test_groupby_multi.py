"""Tests for multi-column groupby functionality.

These tests verify that the multi-key groupby operations produce results
that match native Pandas groupby behavior.
"""

from typing import Any, Literal, cast

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


@pytest.fixture
def large_multi_df():
    """Large DataFrame for multi-key groupby tests (above fallback threshold)."""
    np.random.seed(42)
    n = 500_000
    return pd.DataFrame(
        {
            "k1": np.random.randint(0, 100, size=n),
            "k2": np.random.randint(0, 50, size=n),
            "k3": np.random.randint(0, 20, size=n),
            "val_int": np.random.randint(0, 100, size=n),
            "val_float": np.random.random(size=n) * 100,
        }
    )


@pytest.fixture
def small_multi_df():
    """Small DataFrame that should trigger fallback."""
    return pd.DataFrame(
        {
            "k1": [1, 1, 2, 2, 1],
            "k2": [10, 20, 10, 20, 10],
            "val_int": [10, 20, 30, 40, 50],
            "val_float": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


class TestMultiKeyGroupBySum:
    """Test multi-key groupby sum operations."""

    def test_two_keys_sum_f64_matches_pandas(self, large_multi_df):
        booster_result = cast(BoosterAccessor, large_multi_df.booster).groupby(
            ["k1", "k2"], "val_float", "sum"
        )
        pandas_result = large_multi_df.groupby(["k1", "k2"])["val_float"].sum()

        booster_sorted = booster_result.sort_index()
        pandas_sorted = pandas_result.sort_index()

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=False,
            rtol=1e-10,
        )

    def test_three_keys_sum_f64_matches_pandas(self, large_multi_df):
        booster_result = cast(BoosterAccessor, large_multi_df.booster).groupby(
            ["k1", "k2", "k3"], "val_float", "sum"
        )
        pandas_result = large_multi_df.groupby(["k1", "k2", "k3"])["val_float"].sum()

        booster_sorted = booster_result.sort_index()
        pandas_sorted = pandas_result.sort_index()

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=False,
            rtol=1e-10,
        )

    def test_two_keys_sum_i64_matches_pandas(self, large_multi_df):
        booster_result = cast(BoosterAccessor, large_multi_df.booster).groupby(
            ["k1", "k2"], "val_int", "sum"
        )
        pandas_result = large_multi_df.groupby(["k1", "k2"])["val_int"].sum()

        booster_sorted = booster_result.sort_index()
        pandas_sorted = pandas_result.sort_index()

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=True,
        )


class TestMultiKeyGroupByMean:
    """Test multi-key groupby mean operations."""

    def test_two_keys_mean_f64_matches_pandas(self, large_multi_df):
        booster_result = cast(BoosterAccessor, large_multi_df.booster).groupby(
            ["k1", "k2"], "val_float", "mean"
        )
        pandas_result = large_multi_df.groupby(["k1", "k2"])["val_float"].mean()

        booster_sorted = booster_result.sort_index()
        pandas_sorted = pandas_result.sort_index()

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=False,
            rtol=1e-10,
        )

    def test_three_keys_mean_f64_matches_pandas(self, large_multi_df):
        booster_result = cast(BoosterAccessor, large_multi_df.booster).groupby(
            ["k1", "k2", "k3"], "val_float", "mean"
        )
        pandas_result = large_multi_df.groupby(["k1", "k2", "k3"])["val_float"].mean()

        booster_sorted = booster_result.sort_index()
        pandas_sorted = pandas_result.sort_index()

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=False,
            rtol=1e-10,
        )


class TestMultiKeyGroupByStdVar:
    def _make_ordered_two_key_std_var_df(self) -> pd.DataFrame:
        n = 200_000
        first_order = [(4, 40), (1, 10), (3, 30), (2, 20)]
        first_rows = [0, 60_000, 120_000, 180_000]

        k1 = np.empty(n, dtype=np.int64)
        k2 = np.empty(n, dtype=np.int64)

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

        fill_segment(first_rows[1] + 1, first_rows[2], first_order[:2])
        fill_segment(first_rows[2] + 1, first_rows[3], first_order[:3])
        fill_segment(first_rows[3] + 1, n, first_order[:4])

        for (a, b), row in zip(first_order, first_rows):
            k1[row] = a
            k2[row] = b

        base = np.arange(n, dtype=np.float64)
        val_float = ((base % 41.0) - 20.0) / 7.0 + (k1 * 0.125) - (k2 * 0.01)
        val_int = ((np.arange(n, dtype=np.int64) % 53) - 26) + (k1 * 3) - (k2 // 10)

        return pd.DataFrame(
            {
                "k1": k1,
                "k2": k2,
                "val_float": val_float,
                "val_int": val_int,
            }
        )

    def _make_three_key_std_var_df(self) -> pd.DataFrame:
        n = 240_000
        groups = [
            (3, 30, 300),
            (1, 10, 100),
            (4, 40, 401),
            (2, 20, 200),
            (4, 40, 400),
            (1, 10, 101),
        ]
        g = np.array(groups, dtype=np.int64)
        idx = np.arange(n) % len(groups)
        k1 = g[idx, 0].copy()
        k2 = g[idx, 1].copy()
        k3 = g[idx, 2].copy()

        base = np.arange(n, dtype=np.float64)
        val_float = ((base % 37.0) - 18.0) / 5.0 + (k1 * 0.05) + (k3 * 0.0005)
        val_int = ((np.arange(n, dtype=np.int64) % 47) - 23) + (k1 * 2) - (k3 % 7)

        return pd.DataFrame(
            {
                "k1": k1,
                "k2": k2,
                "k3": k3,
                "val_float": val_float,
                "val_int": val_int,
            }
        )

    def _assert_std_var_series_equal(
        self, left: pd.Series, right: pd.Series, *, expected_names: list[str]
    ) -> None:
        assert isinstance(left.index, pd.MultiIndex)
        assert left.index.names == expected_names
        assert left.dtype == np.float64

        pd.testing.assert_series_equal(
            left,
            right,
            check_exact=False,
            check_dtype=True,
            rtol=1e-10,
        )

    def _patch_groupby_fallback_to_fail(
        self, monkeypatch: pytest.MonkeyPatch, agg: StdVarAgg
    ) -> None:
        series_groupby_cls = type(pd.DataFrame({"k": [0, 0], "v": [1.0, 2.0]}).groupby("k")["v"])

        def _unexpected_fallback(*_args: object, **_kwargs: object) -> pd.Series:
            raise AssertionError(f"proxy unexpectedly used pandas SeriesGroupBy.{agg} fallback")

        monkeypatch.setattr(series_groupby_cls, agg, _unexpected_fallback)

    @pytest.mark.parametrize("agg", ["std", "var"])
    @pytest.mark.parametrize("target", ["val_float", "val_int"])
    def test_accessor_two_keys_sort_true_std_var_matches_pandas(
        self, agg: StdVarAgg, target: str
    ) -> None:
        df = self._make_ordered_two_key_std_var_df()

        booster_result = cast(Any, cast(BoosterAccessor, df.booster)).groupby(
            ["k1", "k2"], target, agg, sort=True
        )
        pandas_result = getattr(df.groupby(["k1", "k2"], sort=True)[target], agg)()

        assert booster_result.index.tolist() == pandas_result.index.tolist()
        self._assert_std_var_series_equal(
            booster_result,
            pandas_result,
            expected_names=["k1", "k2"],
        )

    @pytest.mark.parametrize("agg", ["std", "var"])
    @pytest.mark.parametrize("target", ["val_float", "val_int"])
    def test_accessor_two_keys_sort_false_std_var_preserves_first_seen_order(
        self, agg: StdVarAgg, target: str
    ) -> None:
        df = self._make_ordered_two_key_std_var_df()

        booster_result = cast(Any, cast(BoosterAccessor, df.booster)).groupby(
            ["k1", "k2"], target, agg, sort=False
        )
        pandas_result = getattr(df.groupby(["k1", "k2"], sort=False)[target], agg)()

        expected_order = [(4, 40), (1, 10), (3, 30), (2, 20)]
        assert booster_result.index.tolist() == pandas_result.index.tolist() == expected_order
        self._assert_std_var_series_equal(
            booster_result,
            pandas_result,
            expected_names=["k1", "k2"],
        )

    @pytest.mark.parametrize("agg", ["std", "var"])
    @pytest.mark.parametrize("target", ["val_float", "val_int"])
    def test_accessor_three_keys_sort_true_std_var_matches_pandas_multiindex_order(
        self, agg: StdVarAgg, target: str
    ) -> None:
        df = self._make_three_key_std_var_df()

        booster_result = cast(Any, cast(BoosterAccessor, df.booster)).groupby(
            ["k1", "k2", "k3"], target, agg, sort=True
        )
        pandas_result = getattr(df.groupby(["k1", "k2", "k3"], sort=True)[target], agg)()

        assert booster_result.index.tolist() == pandas_result.index.tolist()
        self._assert_std_var_series_equal(
            booster_result,
            pandas_result,
            expected_names=["k1", "k2", "k3"],
        )

    @pytest.mark.parametrize("agg", ["std", "var"])
    @pytest.mark.parametrize("target", ["val_float", "val_int"])
    def test_proxy_two_keys_sort_false_std_var_preserves_first_seen_order(
        self, monkeypatch: pytest.MonkeyPatch, agg: StdVarAgg, target: str
    ) -> None:
        import pandas_booster

        df = self._make_ordered_two_key_std_var_df()
        pandas_result = getattr(df.groupby(["k1", "k2"], sort=False)[target], agg)()

        expected_order = [(4, 40), (1, 10), (3, 30), (2, 20)]
        assert pandas_result.index.tolist() == expected_order

        self._patch_groupby_fallback_to_fail(monkeypatch, agg)

        pandas_booster.activate()
        try:
            proxy_result = getattr(df.groupby(["k1", "k2"], sort=False)[target], agg)()
        finally:
            pandas_booster.deactivate()

        assert proxy_result.index.tolist() == expected_order
        self._assert_std_var_series_equal(proxy_result, pandas_result, expected_names=["k1", "k2"])

    @pytest.mark.parametrize("agg", ["std", "var"])
    @pytest.mark.parametrize("target", ["val_float", "val_int"])
    def test_proxy_three_keys_sort_true_std_var_matches_pandas_multiindex_order(
        self, monkeypatch: pytest.MonkeyPatch, agg: StdVarAgg, target: str
    ) -> None:
        import pandas_booster

        df = self._make_three_key_std_var_df()
        pandas_result = getattr(df.groupby(["k1", "k2", "k3"], sort=True)[target], agg)()

        self._patch_groupby_fallback_to_fail(monkeypatch, agg)

        pandas_booster.activate()
        try:
            proxy_result = getattr(df.groupby(["k1", "k2", "k3"], sort=True)[target], agg)()
        finally:
            pandas_booster.deactivate()

        assert proxy_result.index.tolist() == pandas_result.index.tolist()
        self._assert_std_var_series_equal(
            proxy_result,
            pandas_result,
            expected_names=["k1", "k2", "k3"],
        )


class TestMultiKeyGroupByMinMax:
    """Test multi-key groupby min/max operations."""

    def test_two_keys_min_f64_matches_pandas(self, large_multi_df):
        booster_result = cast(BoosterAccessor, large_multi_df.booster).groupby(
            ["k1", "k2"], "val_float", "min"
        )
        pandas_result = large_multi_df.groupby(["k1", "k2"])["val_float"].min()

        booster_sorted = booster_result.sort_index()
        pandas_sorted = pandas_result.sort_index()

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=False,
            rtol=1e-10,
        )

    def test_two_keys_max_f64_matches_pandas(self, large_multi_df):
        booster_result = cast(BoosterAccessor, large_multi_df.booster).groupby(
            ["k1", "k2"], "val_float", "max"
        )
        pandas_result = large_multi_df.groupby(["k1", "k2"])["val_float"].max()

        booster_sorted = booster_result.sort_index()
        pandas_sorted = pandas_result.sort_index()

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=False,
            rtol=1e-10,
        )

    def test_three_keys_min_i64_matches_pandas(self, large_multi_df):
        booster_result = cast(BoosterAccessor, large_multi_df.booster).groupby(
            ["k1", "k2", "k3"], "val_int", "min"
        )
        pandas_result = large_multi_df.groupby(["k1", "k2", "k3"])["val_int"].min()

        booster_sorted = booster_result.sort_index()
        pandas_sorted = pandas_result.sort_index()

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=True,
        )


class TestMultiKeyFallback:
    """Test fallback to Pandas for multi-key groupby."""

    def test_small_df_uses_fallback(self, small_multi_df):
        booster_result = cast(BoosterAccessor, small_multi_df.booster).groupby(
            ["k1", "k2"], "val_float", "sum"
        )
        pandas_result = small_multi_df.groupby(["k1", "k2"])["val_float"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )


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
                "k1": pd.array(np.random.randint(0, 100, n), dtype="Int64"),
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

    def test_too_many_keys_raises(self, large_multi_df):
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


class TestMultiKeyMultiIndex:
    """Test that multi-key results have proper MultiIndex structure."""

    def test_multiindex_names(self, large_multi_df):
        """MultiIndex should have correct level names."""
        result = cast(BoosterAccessor, large_multi_df.booster).groupby(
            ["k1", "k2"], "val_float", "sum"
        )

        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == ["k1", "k2"]

    def test_multiindex_three_levels(self, large_multi_df):
        """Three-key groupby should have 3-level MultiIndex."""
        result = cast(BoosterAccessor, large_multi_df.booster).groupby(
            ["k1", "k2", "k3"], "val_float", "sum"
        )

        assert isinstance(result.index, pd.MultiIndex)
        assert len(result.index.names) == 3
        assert result.index.names == ["k1", "k2", "k3"]

    def test_series_name_preserved(self, large_multi_df):
        """Result Series should have target column name."""
        result = cast(BoosterAccessor, large_multi_df.booster).groupby(
            ["k1", "k2"], "val_float", "sum"
        )

        assert result.name == "val_float"


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

class TestProdMultiKey:
    def test_composite_key_prod_sort_true_and_false_order(self):
        base = pd.DataFrame(
            {
                "k1": ["b", "a", "b", "a", "a"],
                "k2": [2, 1, 2, 2, 1],
                "val": [2.0, 3.0, 5.0, 7.0, np.nan],
            }
        )
        repeats = 40_000
        df = pd.concat([base] * repeats, ignore_index=True)
        df["k1_code"] = df["k1"].map({"a": 1, "b": 2}).astype(np.int64)

        for sort, expected_order in (
            (True, [(1, 1), (1, 2), (2, 2)]),
            (False, [(2, 2), (1, 1), (1, 2)]),
        ):
            booster_result = cast(BoosterAccessor, df.booster).groupby(
                ["k1_code", "k2"], "val", "prod", sort=sort
            )
            pandas_result = df.groupby(["k1_code", "k2"], sort=sort)["val"].prod()
            assert booster_result.index.tolist() == pandas_result.index.tolist() == expected_order
            pd.testing.assert_series_equal(
                booster_result,
                pandas_result,
                check_exact=False,
                rtol=1e-10,
                atol=0.0,
            )

    def test_multi_key_prod_int_exactness_sort_modes(self):
        n = 200_000
        df = pd.DataFrame(
            {
                "k1": np.tile(np.array([3, 1, 3, 2], dtype=np.int64), n // 4),
                "k2": np.tile(np.array([30, 10, 31, 20], dtype=np.int64), n // 4),
                "val": np.tile(np.array([-2, 1, 2, 3], dtype=np.int64), n // 4),
            }
        )
        for sort in (True, False):
            booster_result = cast(BoosterAccessor, df.booster).groupby(
                ["k1", "k2"], "val", "prod", sort=sort
            )
            pandas_result = df.groupby(["k1", "k2"], sort=sort)["val"].prod()
            pd.testing.assert_series_equal(booster_result, pandas_result, check_exact=True)

    def test_multi_key_prod_all_nan_group_retained_as_one(self):
        n = 200_000
        df = pd.DataFrame(
            {
                "k1": np.repeat(np.array([1, 2, 3, 4], dtype=np.int64), n // 4),
                "k2": np.repeat(np.array([10, 20, 30, 40], dtype=np.int64), n // 4),
                "val": np.ones(n, dtype=np.float64) * 1.001,
            }
        )
        df.loc[(df["k1"] == 2) & (df["k2"] == 20), "val"] = np.nan

        booster_result = cast(BoosterAccessor, df.booster).groupby(
            ["k1", "k2"], "val", "prod", sort=False
        )
        pandas_result = df.groupby(["k1", "k2"], sort=False)["val"].prod()

        assert (2, 20) in booster_result.index
        assert booster_result.loc[(2, 20)] == pandas_result.loc[(2, 20)] == 1.0
        pd.testing.assert_series_equal(booster_result, pandas_result, check_exact=False, rtol=1e-10)
