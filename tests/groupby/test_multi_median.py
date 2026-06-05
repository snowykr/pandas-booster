"""Multi-key median behavior tests."""

from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import pytest
from pandas_booster.accessor import BoosterAccessor

AggFunc = Literal["sum", "mean", "min", "max", "count", "prod"]
StdVarAgg = Literal["std", "var"]


class TestMultiKeyGroupByMedian:
    def _make_ordered_two_key_df(self) -> pd.DataFrame:
        n = 200_000
        groups = [
            (4, 40),
            (1, 10),
            (3, 30),
            (2, 20),
            (5, 50),
        ]
        g = np.array(groups, dtype=np.int64)
        idx = np.arange(n) % len(groups)
        k1 = g[idx, 0].copy()
        k2 = g[idx, 1].copy()

        val_float = ((np.arange(n, dtype=np.float64) % 12.0) - 6.0) / 3.0
        val_int = (np.arange(n, dtype=np.int64) % 12) - 6

        val_float[(k1 == 1) & (k2 == 10)] = np.nan
        mixed_nan = (k1 == 3) & (k2 == 30) & ((np.arange(n) % 2) == 0)
        val_float[mixed_nan] = np.nan

        return pd.DataFrame({"k1": k1, "k2": k2, "val_float": val_float, "val_int": val_int})

    def _make_three_key_df(self) -> pd.DataFrame:
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

        val_float = ((np.arange(n, dtype=np.float64) % 15.0) - 7.0) / 4.0
        val_int = (np.arange(n, dtype=np.int64) % 15) - 7

        return pd.DataFrame(
            {
                "k1": k1,
                "k2": k2,
                "k3": k3,
                "val_float": val_float,
                "val_int": val_int,
            }
        )

    def _assert_median_series_equal(
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

    @pytest.mark.parametrize("target", ["val_float", "val_int"])
    def test_accessor_two_keys_sort_true_matches_pandas(self, target: str) -> None:
        df = self._make_ordered_two_key_df()

        booster_result = cast(Any, cast(BoosterAccessor, df.booster)).groupby(
            ["k1", "k2"], target, "median", sort=True
        )
        pandas_result = df.groupby(["k1", "k2"], sort=True)[target].median()

        assert booster_result.index.tolist() == pandas_result.index.tolist()
        self._assert_median_series_equal(
            booster_result,
            pandas_result,
            expected_names=["k1", "k2"],
        )

    @pytest.mark.parametrize("target", ["val_float", "val_int"])
    def test_accessor_two_keys_sort_false_preserves_first_seen_order(self, target: str) -> None:
        df = self._make_ordered_two_key_df()

        booster_result = cast(Any, cast(BoosterAccessor, df.booster)).groupby(
            ["k1", "k2"], target, "median", sort=False
        )
        pandas_result = df.groupby(["k1", "k2"], sort=False)[target].median()

        expected_order = [(4, 40), (1, 10), (3, 30), (2, 20), (5, 50)]
        assert booster_result.index.tolist() == pandas_result.index.tolist() == expected_order
        self._assert_median_series_equal(
            booster_result,
            pandas_result,
            expected_names=["k1", "k2"],
        )

    @pytest.mark.parametrize("target", ["val_float", "val_int"])
    def test_accessor_three_keys_sort_true_matches_pandas_multiindex_order(
        self, target: str
    ) -> None:
        df = self._make_three_key_df()

        booster_result = cast(Any, cast(BoosterAccessor, df.booster)).groupby(
            ["k1", "k2", "k3"], target, "median", sort=True
        )
        pandas_result = df.groupby(["k1", "k2", "k3"], sort=True)[target].median()

        assert booster_result.index.tolist() == pandas_result.index.tolist()
        self._assert_median_series_equal(
            booster_result,
            pandas_result,
            expected_names=["k1", "k2", "k3"],
        )
