"""Single-key product behavior tests."""

from typing import Literal, cast

import numpy as np
import pandas as pd
import pytest
from pandas_booster.accessor import BoosterAccessor

AggFunc = Literal["sum", "mean", "min", "max", "count", "prod"]


def _assert_prod_series_matches_pandas(
    booster_result: pd.Series,
    pandas_result: pd.Series,
    *,
    check_dtype: bool = True,
) -> None:
    pd.testing.assert_series_equal(
        booster_result,
        pandas_result,
        check_exact=False,
        check_dtype=check_dtype,
        rtol=1e-10,
        atol=0.0,
    )
    zero_mask = (pandas_result.to_numpy() == 0.0) & ~pd.isna(pandas_result.to_numpy())
    if zero_mask.any():
        np.testing.assert_array_equal(
            np.signbit(booster_result.to_numpy()[zero_mask]),
            np.signbit(pandas_result.to_numpy()[zero_mask]),
        )


class TestProdSingleKey:
    def test_prod_f64_matches_pandas_sort_modes(self):
        n = 200_000
        idx = np.arange(n, dtype=np.int64)
        df = pd.DataFrame(
            {
                "key": np.tile(np.array([3, 1, 2, 5], dtype=np.int64), n // 4),
                "val": (1.0 + (idx % 7).astype(np.float64) / 1000.0),
            }
        )
        df.loc[df.index % 997 == 0, "val"] = np.nan

        for sort in (True, False):
            booster_result = cast(BoosterAccessor, df.booster).groupby(
                "key", "val", "prod", sort=sort
            )
            pandas_result = df.groupby("key", sort=sort)["val"].prod()
            _assert_prod_series_matches_pandas(booster_result, pandas_result)

    @pytest.mark.parametrize("sort", [True, False])
    def test_single_key_float_prod_order_sensitive_chunks_use_rust_path(
        self, monkeypatch: pytest.MonkeyPatch, sort: bool
    ):
        import pandas_booster._rust as rust

        monkeypatch.delenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", raising=False)
        monkeypatch.delenv("PANDAS_BOOSTER_FORCE_PANDAS_SORT", raising=False)

        n = rust.get_fallback_threshold()
        df = pd.DataFrame(
            {
                "key": np.ones(n, dtype=np.int64),
                "val": np.r_[
                    np.full(n // 2, 0.5),
                    np.full(n - n // 2, 2.0),
                ].astype(np.float64),
            }
        )
        expected = df.groupby("key", sort=sort)["val"].prod()
        called = False

        def _fake_rust_prod(*_args, **_kwargs):
            nonlocal called
            called = True
            return expected.index.to_numpy(dtype=np.int64), expected.to_numpy(dtype=np.float64)

        suffix = "_sorted" if sort else "_firstseen_u32"
        monkeypatch.setattr(rust, f"groupby_prod_f64{suffix}", _fake_rust_prod, raising=False)

        result = cast(BoosterAccessor, df.booster).groupby("key", "val", "prod", sort=sort)

        pd.testing.assert_series_equal(result, expected, check_exact=True)
        assert called is True

    def test_prod_i64_matches_pandas_sort_modes(self):
        n = 200_000
        vals = np.tile(np.array([-2, -1, 1, 2, 3], dtype=np.int64), n // 5)
        df = pd.DataFrame(
            {
                "key": np.tile(np.array([9, 4, 9, 1, 4], dtype=np.int64), n // 5),
                "val": vals,
            }
        )

        for sort in (True, False):
            booster_result = cast(BoosterAccessor, df.booster).groupby(
                "key", "val", "prod", sort=sort
            )
            pandas_result = df.groupby("key", sort=sort)["val"].prod()
            pd.testing.assert_series_equal(booster_result, pandas_result, check_exact=True)

    def test_prod_float_special_values_match_pandas_observables(self):
        pairs = [
            (10, np.nan),
            (10, np.nan),
            (20, np.nan),
            (20, 2.0),
            (30, np.inf),
            (30, 0.0),
            (40, -np.inf),
            (40, 0.0),
            (50, -0.0),
            (50, 2.0),
            (60, 0.0),
            (60, -2.0),
            (70, 1e308),
            (70, 1e308),
            (80, 1e-308),
            (80, 1e-308),
        ]
        repeats = 12_500
        df = pd.DataFrame(
            {
                "key": np.array([k for k, _ in pairs] * repeats, dtype=np.int64),
                "val": np.array([v for _, v in pairs] * repeats, dtype=np.float64),
            }
        )

        booster_result = cast(BoosterAccessor, df.booster).groupby("key", "val", "prod", sort=True)
        pandas_result = df.groupby("key", sort=True)["val"].prod()

        _assert_prod_series_matches_pandas(booster_result, pandas_result)
        assert booster_result.loc[10] == pandas_result.loc[10] == 1.0
        assert np.isnan(booster_result.loc[30]) and np.isnan(pandas_result.loc[30])
        assert np.isnan(booster_result.loc[40]) and np.isnan(pandas_result.loc[40])
        assert np.signbit(booster_result.loc[50]) == np.signbit(pandas_result.loc[50])
        assert np.signbit(booster_result.loc[60]) == np.signbit(pandas_result.loc[60])

    def test_single_key_float_prod_respects_force_pandas_escape_hatch(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        import pandas_booster._rust as rust

        n = 120_000
        df = pd.DataFrame(
            {
                "key": np.repeat(np.array([1, 2], dtype=np.int64), n // 2),
                "val": np.linspace(1.001, 1.01, n, dtype=np.float64),
            }
        )
        expected = df.groupby("key")["val"].prod()

        def _boom(*_args, **_kwargs):
            raise AssertionError(
                "float prod should use pandas when force-pandas escape hatch is set"
            )

        monkeypatch.setenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", "1")
        for suffix in ("", "_sorted", "_firstseen_u32", "_firstseen_u64"):
            monkeypatch.setattr(rust, f"groupby_prod_f64{suffix}", _boom, raising=False)

        result = cast(BoosterAccessor, df.booster).groupby("key", "val", "prod", sort=True)
        pd.testing.assert_series_equal(result, expected)

    @pytest.mark.parametrize("sort", [True, False])
    def test_single_key_float_prod_without_ordered_abi_marker_falls_back_to_pandas(
        self, monkeypatch: pytest.MonkeyPatch, sort: bool
    ):
        import pandas_booster._abi_compat as abi
        import pandas_booster.accessor as accessor_mod

        class StaleRust:
            @staticmethod
            def get_fallback_threshold() -> int:
                return 100_000

            @staticmethod
            def groupby_prod_f64_sorted(*_args, **_kwargs):
                raise AssertionError("stale single-key float prod kernel must not be called")

            @staticmethod
            def groupby_prod_f64_firstseen_u32(*_args, **_kwargs):
                raise AssertionError("stale single-key float prod kernel must not be called")

        monkeypatch.setattr(
            accessor_mod.BoosterAccessor,
            "_get_rust_module",
            staticmethod(lambda: StaleRust),
        )
        monkeypatch.delenv("PANDAS_BOOSTER_STRICT_ABI", raising=False)
        monkeypatch.delenv("PANDAS_BOOSTER_ABI_SKEW_NOTICE", raising=False)
        monkeypatch.setattr(abi, "_WARNED_ABI_SKEW", False)

        n = 100_000
        df = pd.DataFrame(
            {
                "key": np.resize(np.array([1, 2], dtype=np.int64), n),
                "val": np.resize(np.array([1e308, 1e308, 1e-308, 1e-308], dtype=np.float64), n),
            }
        )
        expected = df.groupby("key", sort=sort)["val"].prod()

        result = cast(BoosterAccessor, df.booster).groupby("key", "val", "prod", sort=sort)

        pd.testing.assert_series_equal(result, expected, check_exact=True)

    def test_single_key_float_prod_without_ordered_abi_marker_hard_fails_in_strict_abi(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        import pandas_booster._abi_compat as abi
        import pandas_booster.accessor as accessor_mod

        class StaleRust:
            @staticmethod
            def get_fallback_threshold() -> int:
                return 100_000

            @staticmethod
            def groupby_prod_f64_sorted(*_args, **_kwargs):
                raise AssertionError("stale single-key float prod kernel must not be called")

        monkeypatch.setattr(
            accessor_mod.BoosterAccessor,
            "_get_rust_module",
            staticmethod(lambda: StaleRust),
        )
        monkeypatch.setenv("PANDAS_BOOSTER_STRICT_ABI", "1")
        monkeypatch.delenv("PANDAS_BOOSTER_ABI_SKEW_NOTICE", raising=False)
        monkeypatch.setattr(abi, "_WARNED_ABI_SKEW", False)

        n = 100_000
        df = pd.DataFrame(
            {
                "key": np.resize(np.array([1, 2], dtype=np.int64), n),
                "val": np.linspace(1.001, 1.01, n, dtype=np.float64),
            }
        )

        with pytest.raises(
            abi.PandasBoosterKeyShapeSkewError,
            match="has_ordered_single_key_float_prod_abi",
        ):
            cast(BoosterAccessor, df.booster).groupby("key", "val", "prod", sort=True)
