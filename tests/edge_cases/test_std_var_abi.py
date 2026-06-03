"""Std/var ABI-skew fallback and strict-mode contract tests."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from ._helpers import (
    _accessor_groupby_result,
    _delete_groupby_kernel_symbols,
)


class TestStdVarAbiContracts:
    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_missing_single_key_std_var_symbols_fall_back_as_abi_skew(
        self, monkeypatch: pytest.MonkeyPatch, agg: str
    ):
        import pandas_booster
        import pandas_booster._abi_compat as abi
        import pandas_booster._rust as rust
        from pandas_booster.accessor import BoosterAccessor

        monkeypatch.delenv("PANDAS_BOOSTER_STRICT_ABI", raising=False)
        monkeypatch.delenv("PANDAS_BOOSTER_ABI_SKEW_NOTICE", raising=False)
        monkeypatch.delenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", raising=False)

        df = pd.DataFrame(
            {
                "key": np.repeat([1, 2, 3], 4),
                "val": np.array([1.0, 2.0, 4.0, 8.0, 2.5, 3.5, 6.5, 7.5, 5.0, 6.0, 9.0, 10.0]),
            }
        )
        expected = getattr(df.groupby("key", sort=True)["val"], agg)()

        _delete_groupby_kernel_symbols(monkeypatch, rust, agg, kernel="f64", multi=False)
        monkeypatch.setattr(abi, "_WARNED_ABI_SKEW", False)

        booster = df.booster
        assert isinstance(booster, BoosterAccessor)
        fallback_called = {"n": 0}
        orig_fallback = booster._pandas_fallback

        def wrapped_fallback(by_cols, target, wrapped_agg, *, sort: bool):
            fallback_called["n"] += 1
            return orig_fallback(by_cols, target, wrapped_agg, sort=sort)

        monkeypatch.setattr(booster, "_pandas_fallback", wrapped_fallback)

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            accessor_result = _accessor_groupby_result(df, "key", "val", agg)

            pandas_booster.activate()
            try:
                proxy_result = getattr(df.groupby("key", sort=True)["val"], agg)()
            finally:
                pandas_booster.deactivate()

        assert fallback_called["n"] == 1
        pd.testing.assert_series_equal(accessor_result, expected, check_exact=False, rtol=1e-12)
        pd.testing.assert_series_equal(proxy_result, expected, check_exact=False, rtol=1e-12)

        abi_warnings = [w for w in rec if issubclass(w.category, abi.PandasBoosterAbiSkewWarning)]
        abi_warnings = [w for w in abi_warnings if abi.ABI_SKEW_PREFIX in str(w.message)]
        assert len(abi_warnings) == 1
        assert f"missing Rust kernel symbol 'groupby_{agg}_f64'" in str(abi_warnings[0].message)

    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_missing_multi_key_std_var_symbols_fall_back_as_abi_skew(
        self, monkeypatch: pytest.MonkeyPatch, agg: str
    ):
        import pandas_booster
        import pandas_booster._abi_compat as abi
        import pandas_booster._rust as rust
        from pandas_booster.accessor import BoosterAccessor

        monkeypatch.delenv("PANDAS_BOOSTER_STRICT_ABI", raising=False)
        monkeypatch.delenv("PANDAS_BOOSTER_ABI_SKEW_NOTICE", raising=False)
        monkeypatch.delenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", raising=False)

        df = pd.DataFrame(
            {
                "k1": [1, 1, 1, 2, 2, 2],
                "k2": [10, 10, 20, 10, 10, 20],
                "val": [1.0, 5.0, 9.0, 2.0, 6.0, 10.0],
            }
        )
        by_cols = ["k1", "k2"]
        expected = getattr(df.groupby(by_cols, sort=True)["val"], agg)()

        _delete_groupby_kernel_symbols(monkeypatch, rust, agg, kernel="f64", multi=True)
        monkeypatch.setattr(abi, "_WARNED_ABI_SKEW", False)

        booster = df.booster
        assert isinstance(booster, BoosterAccessor)
        fallback_called = {"n": 0}
        orig_fallback = booster._pandas_fallback

        def wrapped_fallback(wrapped_by_cols, target, wrapped_agg, *, sort: bool):
            fallback_called["n"] += 1
            return orig_fallback(wrapped_by_cols, target, wrapped_agg, sort=sort)

        monkeypatch.setattr(booster, "_pandas_fallback", wrapped_fallback)

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            accessor_result = _accessor_groupby_result(df, by_cols, "val", agg)

            pandas_booster.activate()
            try:
                proxy_result = getattr(df.groupby(by_cols, sort=True)["val"], agg)()
            finally:
                pandas_booster.deactivate()

        assert fallback_called["n"] == 1
        pd.testing.assert_series_equal(accessor_result, expected, check_exact=False, rtol=1e-12)
        pd.testing.assert_series_equal(proxy_result, expected, check_exact=False, rtol=1e-12)

        abi_warnings = [w for w in rec if issubclass(w.category, abi.PandasBoosterAbiSkewWarning)]
        abi_warnings = [w for w in abi_warnings if abi.ABI_SKEW_PREFIX in str(w.message)]
        assert len(abi_warnings) == 1
        assert f"missing Rust kernel symbol 'groupby_multi_{agg}_f64'" in str(
            abi_warnings[0].message
        )

    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_missing_single_key_std_var_symbols_hard_fail_in_strict_abi(
        self, monkeypatch: pytest.MonkeyPatch, agg: str
    ):
        import pandas_booster
        import pandas_booster._abi_compat as abi
        import pandas_booster._rust as rust

        monkeypatch.setenv("PANDAS_BOOSTER_STRICT_ABI", "1")
        monkeypatch.delenv("PANDAS_BOOSTER_ABI_SKEW_NOTICE", raising=False)
        monkeypatch.delenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", raising=False)

        df = pd.DataFrame(
            {
                "key": np.repeat([1, 2, 3], 4),
                "val": np.array([1.0, 2.0, 4.0, 8.0, 2.5, 3.5, 6.5, 7.5, 5.0, 6.0, 9.0, 10.0]),
            }
        )

        _delete_groupby_kernel_symbols(monkeypatch, rust, agg, kernel="f64", multi=False)
        monkeypatch.setattr(abi, "_WARNED_ABI_SKEW", False)

        with pytest.raises(
            abi.PandasBoosterKeyShapeSkewError,
            match=f"missing Rust kernel symbol 'groupby_{agg}_f64'",
        ):
            _ = _accessor_groupby_result(df, "key", "val", agg)

        abi._WARNED_ABI_SKEW = False
        pandas_booster.activate()
        try:
            with pytest.raises(
                abi.PandasBoosterKeyShapeSkewError,
                match=f"missing Rust kernel symbol 'groupby_{agg}_f64'",
            ):
                _ = getattr(df.groupby("key", sort=True)["val"], agg)()
        finally:
            pandas_booster.deactivate()

    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_missing_multi_key_std_var_symbols_hard_fail_in_strict_abi(
        self, monkeypatch: pytest.MonkeyPatch, agg: str
    ):
        import pandas_booster
        import pandas_booster._abi_compat as abi
        import pandas_booster._rust as rust

        monkeypatch.setenv("PANDAS_BOOSTER_STRICT_ABI", "1")
        monkeypatch.delenv("PANDAS_BOOSTER_ABI_SKEW_NOTICE", raising=False)
        monkeypatch.delenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", raising=False)

        df = pd.DataFrame(
            {
                "k1": [1, 1, 1, 2, 2, 2],
                "k2": [10, 10, 20, 10, 10, 20],
                "val": [1.0, 5.0, 9.0, 2.0, 6.0, 10.0],
            }
        )
        by_cols = ["k1", "k2"]

        _delete_groupby_kernel_symbols(monkeypatch, rust, agg, kernel="f64", multi=True)
        monkeypatch.setattr(abi, "_WARNED_ABI_SKEW", False)

        with pytest.raises(
            abi.PandasBoosterKeyShapeSkewError,
            match=f"missing Rust kernel symbol 'groupby_multi_{agg}_f64'",
        ):
            _ = _accessor_groupby_result(df, by_cols, "val", agg)

        abi._WARNED_ABI_SKEW = False
        pandas_booster.activate()
        try:
            with pytest.raises(
                abi.PandasBoosterKeyShapeSkewError,
                match=f"missing Rust kernel symbol 'groupby_multi_{agg}_f64'",
            ):
                _ = getattr(df.groupby(by_cols, sort=True)["val"], agg)()
        finally:
            pandas_booster.deactivate()
