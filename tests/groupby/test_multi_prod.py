"""Multi-key product behavior tests."""

from typing import Literal, cast

import numpy as np
import pandas as pd
import pytest
from pandas_booster.accessor import BoosterAccessor

AggFunc = Literal["sum", "mean", "min", "max", "count", "prod"]
StdVarAgg = Literal["std", "var"]


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

    @pytest.mark.parametrize("sort", [True, False])
    def test_multi_key_float_prod_order_sensitive_chunks_remains_rust_eligible(
        self, monkeypatch: pytest.MonkeyPatch, sort: bool
    ):
        import pandas_booster._rust as rust

        threshold = rust.get_fallback_threshold()
        n = threshold
        df = pd.DataFrame(
            {
                "k1": np.ones(n, dtype=np.int64),
                "k2": np.ones(n, dtype=np.int64),
                "val": np.r_[
                    np.full(n // 2, 0.5),
                    np.full(n - n // 2, 2.0),
                ].astype(np.float64),
            }
        )
        expected = df.groupby(["k1", "k2"], sort=sort)["val"].prod()
        calls: list[str] = []

        symbol = "groupby_multi_prod_f64_sorted" if sort else "groupby_multi_prod_f64_firstseen_u32"
        original = getattr(rust, symbol)

        def wrapped(*args, **kwargs):
            calls.append(symbol)
            return original(*args, **kwargs)

        monkeypatch.setattr(rust, symbol, wrapped, raising=False)

        result = cast(BoosterAccessor, df.booster).groupby(["k1", "k2"], "val", "prod", sort=sort)

        assert calls == [symbol]
        pd.testing.assert_series_equal(result, expected, check_exact=True)

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
