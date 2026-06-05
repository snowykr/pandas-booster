"""Multi-key std and var behavior tests."""

from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import pytest
from pandas_booster.accessor import BoosterAccessor

AggFunc = Literal["sum", "mean", "min", "max", "count", "prod"]
StdVarAgg = Literal["std", "var"]


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
