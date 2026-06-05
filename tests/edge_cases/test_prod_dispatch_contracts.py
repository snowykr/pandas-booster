"""Product threshold and dispatch contract tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestProdDispatchContracts:
    def _assert_float_prod_equal(self, result: pd.Series, expected: pd.Series) -> None:
        pd.testing.assert_series_equal(
            result.sort_index(),
            expected.sort_index(),
            check_exact=False,
            rtol=1e-10,
            atol=0.0,
        )
        result_sorted = result.sort_index()
        expected_sorted = expected.sort_index()
        zero_mask = (expected_sorted.to_numpy() == 0.0) & ~pd.isna(expected_sorted.to_numpy())
        if zero_mask.any():
            np.testing.assert_array_equal(
                np.signbit(result_sorted.to_numpy()[zero_mask]),
                np.signbit(expected_sorted.to_numpy()[zero_mask]),
            )

    def test_below_threshold_prod_falls_back_without_rust_symbol(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        import pandas_booster._rust as rust

        threshold = rust.get_fallback_threshold()
        n = max(1, threshold - 1)
        df = pd.DataFrame(
            {
                "key": np.resize(np.array([1, 2], dtype=np.int64), n),
                "val": np.linspace(1.001, 1.01, n, dtype=np.float64),
            }
        )

        def _boom(*_args, **_kwargs):
            raise AssertionError("below-threshold prod should use pandas fallback")

        for suffix in ("", "_sorted", "_firstseen_u32", "_firstseen_u64"):
            monkeypatch.setattr(rust, f"groupby_prod_f64{suffix}", _boom, raising=False)

        result = df.booster.groupby("key", "val", "prod")
        expected = df.groupby("key")["val"].prod()
        self._assert_float_prod_equal(result, expected)

    @pytest.mark.parametrize("sort", [True, False])
    def test_at_threshold_single_key_float_prod_invokes_expected_rust_symbol(
        self, monkeypatch: pytest.MonkeyPatch, sort: bool
    ):
        import pandas_booster._rust as rust

        threshold = rust.get_fallback_threshold()
        n = threshold
        df = pd.DataFrame(
            {
                "key": np.resize(np.array([1, 2], dtype=np.int64), n),
                "val": np.linspace(1.001, 1.01, n, dtype=np.float64),
            }
        )
        expected = df.groupby("key", sort=sort)["val"].prod()
        calls: list[str] = []

        symbol = "groupby_prod_f64_sorted" if sort else "groupby_prod_f64_firstseen_u32"

        def fake_kernel(_keys, _values):
            calls.append(symbol)
            return expected.index.to_numpy(dtype=np.int64), expected.to_numpy(dtype=np.float64)

        monkeypatch.setattr(rust, symbol, fake_kernel, raising=False)

        result = df.booster.groupby("key", "val", "prod", sort=sort)

        assert calls == [symbol]
        self._assert_float_prod_equal(result, expected)

    @pytest.mark.parametrize("sort", [True, False])
    def test_multi_key_below_threshold_prod_falls_back_without_rust_symbol(
        self, monkeypatch: pytest.MonkeyPatch, sort: bool
    ):
        import pandas_booster._rust as rust

        threshold = rust.get_fallback_threshold()
        n = max(1, threshold - 1)
        df = pd.DataFrame(
            {
                "k1": np.resize(np.array([1, 2], dtype=np.int64), n),
                "k2": np.resize(np.array([10, 20, 30], dtype=np.int64), n),
                "val": np.linspace(1.001, 1.01, n, dtype=np.float64),
            }
        )

        def _boom(*_args, **_kwargs):
            raise AssertionError("below-threshold multi-key prod should use pandas fallback")

        for suffix in ("", "_sorted", "_firstseen_u32", "_firstseen_u64"):
            monkeypatch.setattr(rust, f"groupby_multi_prod_f64{suffix}", _boom, raising=False)

        result = df.booster.groupby(["k1", "k2"], "val", "prod", sort=sort)
        expected = df.groupby(["k1", "k2"], sort=sort)["val"].prod()
        self._assert_float_prod_equal(result, expected)

    @pytest.mark.parametrize(
        ("sort", "symbol"),
        [
            pytest.param(True, "groupby_multi_prod_f64_sorted", id="sorted"),
            pytest.param(False, "groupby_multi_prod_f64_firstseen_u32", id="firstseen-u32"),
        ],
    )
    def test_multi_key_at_threshold_prod_invokes_expected_rust_symbol(
        self, monkeypatch: pytest.MonkeyPatch, sort: bool, symbol: str
    ):
        import pandas_booster._rust as rust

        threshold = rust.get_fallback_threshold()
        n = threshold
        df = pd.DataFrame(
            {
                "k1": np.resize(np.array([1, 2], dtype=np.int64), n),
                "k2": np.resize(np.array([10, 20, 30], dtype=np.int64), n),
                "val": np.linspace(1.001, 1.01, n, dtype=np.float64),
            }
        )
        expected = df.groupby(["k1", "k2"], sort=sort)["val"].prod()
        calls: list[str] = []

        def fake_kernel(_key_arrays, _values):
            calls.append(symbol)
            return [
                expected.index.get_level_values(0).to_numpy(dtype=np.int64),
                expected.index.get_level_values(1).to_numpy(dtype=np.int64),
            ], expected.to_numpy(dtype=np.float64)

        monkeypatch.setattr(rust, symbol, fake_kernel, raising=False)
        result = df.booster.groupby(["k1", "k2"], "val", "prod", sort=sort)

        assert calls == [symbol]
        self._assert_float_prod_equal(result, expected)
