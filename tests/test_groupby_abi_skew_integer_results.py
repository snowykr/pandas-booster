from __future__ import annotations

import warnings
from typing import Literal, cast

import numpy as np
import pandas as pd
import pytest

AggFunc = Literal["sum", "min", "max", "count"]
ValueKernel = Literal["i64", "f64"]


def _make_single_df(n: int = 120_000, *, is_val_int: bool = True) -> pd.DataFrame:
    vals = np.zeros(n, dtype=np.int64 if is_val_int else np.float64)
    vals[0] = 2**53 + 1 if is_val_int else 1.5
    vals[n // 2 :] = 3 if is_val_int else 3.5
    return pd.DataFrame({"key": np.repeat([1, 2], n // 2), "val": vals})


def _make_multi_df(n: int = 120_000, *, is_val_int: bool = True) -> pd.DataFrame:
    vals = np.zeros(n, dtype=np.int64 if is_val_int else np.float64)
    vals[0] = 2**53 + 1 if is_val_int else 1.5
    vals[n // 2 :] = 3 if is_val_int else 3.5
    return pd.DataFrame(
        {
            "k1": np.repeat([1, 2], n // 2),
            "k2": np.repeat([10, 20], n // 2),
            "val": vals,
        }
    )


def _patch_single_to_return_float_results(
    monkeypatch: pytest.MonkeyPatch,
    rust: object,
    expected: pd.Series,
    agg: AggFunc,
    *,
    kernel: ValueKernel,
) -> None:
    def stale_single(_keys_arr, _values_arr):
        return (
            np.asarray(expected.index.to_numpy(), dtype=np.int64),
            np.asarray(expected.to_numpy(), dtype=np.float64),
        )

    monkeypatch.setattr(rust, f"groupby_{agg}_{kernel}", stale_single)
    monkeypatch.setattr(rust, f"groupby_{agg}_{kernel}_sorted", stale_single)


def _patch_multi_to_return_float_results(
    monkeypatch: pytest.MonkeyPatch,
    rust: object,
    expected: pd.Series,
    agg: AggFunc,
    *,
    kernel: ValueKernel,
) -> None:
    def stale_multi(_key_arrays, _values_arr):
        keys_cols = [
            np.asarray(expected.index.get_level_values(0), dtype=np.int64),
            np.asarray(expected.index.get_level_values(1), dtype=np.int64),
        ]
        return keys_cols, np.asarray(expected.to_numpy(), dtype=np.float64)

    monkeypatch.setattr(rust, f"groupby_multi_{agg}_{kernel}", stale_multi)
    monkeypatch.setattr(rust, f"groupby_multi_{agg}_{kernel}_sorted", stale_multi)


def _assert_single_non_strict_fallback(
    monkeypatch: pytest.MonkeyPatch, agg: AggFunc, *, is_val_int: bool = True
) -> None:
    import pandas_booster
    import pandas_booster._abi_compat as abi
    import pandas_booster._rust as rust
    from pandas_booster.accessor import BoosterAccessor

    monkeypatch.delenv("PANDAS_BOOSTER_STRICT_ABI", raising=False)
    monkeypatch.delenv("PANDAS_BOOSTER_ABI_SKEW_NOTICE", raising=False)

    df = _make_single_df(is_val_int=is_val_int)
    expected = cast(pd.Series, getattr(df.groupby("key", sort=True)["val"], agg)())

    _patch_single_to_return_float_results(
        monkeypatch,
        rust,
        expected,
        agg,
        kernel="i64" if is_val_int else "f64",
    )
    monkeypatch.setattr(abi, "_WARNED_ABI_SKEW", False)

    booster = cast(BoosterAccessor, df.booster)
    fallback_called = {"n": 0}
    orig_fallback = booster._pandas_fallback

    def wrapped_fallback(by_cols, target, wrapped_agg, *, sort: bool):
        fallback_called["n"] += 1
        return orig_fallback(by_cols, target, wrapped_agg, sort=sort)

    monkeypatch.setattr(booster, "_pandas_fallback", wrapped_fallback)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        accessor_out = booster.groupby("key", "val", agg, sort=True)

        pandas_booster.activate()
        try:
            proxy_out = cast(pd.Series, getattr(df.groupby("key", sort=True)["val"], agg)())
        finally:
            pandas_booster.deactivate()

    assert fallback_called["n"] == 1
    pd.testing.assert_series_equal(accessor_out, expected, check_exact=True)
    pd.testing.assert_series_equal(proxy_out, expected, check_exact=True)

    abi_warnings = [w for w in rec if issubclass(w.category, abi.PandasBoosterAbiSkewWarning)]
    abi_warnings = [w for w in abi_warnings if abi.ABI_SKEW_PREFIX in str(w.message)]
    assert len(abi_warnings) == 1
    assert f"expected integer result dtype for agg={agg}" in str(abi_warnings[0].message)


def _assert_multi_non_strict_fallback(
    monkeypatch: pytest.MonkeyPatch, agg: AggFunc, *, is_val_int: bool = True
) -> None:
    import pandas_booster
    import pandas_booster._abi_compat as abi
    import pandas_booster._rust as rust
    from pandas_booster.accessor import BoosterAccessor

    monkeypatch.delenv("PANDAS_BOOSTER_STRICT_ABI", raising=False)
    monkeypatch.delenv("PANDAS_BOOSTER_ABI_SKEW_NOTICE", raising=False)

    df = _make_multi_df(is_val_int=is_val_int)
    by_cols = ["k1", "k2"]
    expected = cast(pd.Series, getattr(df.groupby(by_cols, sort=True)["val"], agg)())

    _patch_multi_to_return_float_results(
        monkeypatch,
        rust,
        expected,
        agg,
        kernel="i64" if is_val_int else "f64",
    )
    monkeypatch.setattr(abi, "_WARNED_ABI_SKEW", False)

    booster = cast(BoosterAccessor, df.booster)
    fallback_called = {"n": 0}
    orig_fallback = booster._pandas_fallback

    def wrapped_fallback(wrapped_by_cols, target, wrapped_agg, *, sort: bool):
        fallback_called["n"] += 1
        return orig_fallback(wrapped_by_cols, target, wrapped_agg, sort=sort)

    monkeypatch.setattr(booster, "_pandas_fallback", wrapped_fallback)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        accessor_out = booster.groupby(by_cols, "val", agg, sort=True)

        pandas_booster.activate()
        try:
            proxy_out = cast(pd.Series, getattr(df.groupby(by_cols, sort=True)["val"], agg)())
        finally:
            pandas_booster.deactivate()

    assert fallback_called["n"] == 1
    pd.testing.assert_series_equal(accessor_out, expected, check_exact=True)
    pd.testing.assert_series_equal(proxy_out, expected, check_exact=True)

    abi_warnings = [w for w in rec if issubclass(w.category, abi.PandasBoosterAbiSkewWarning)]
    abi_warnings = [w for w in abi_warnings if abi.ABI_SKEW_PREFIX in str(w.message)]
    assert len(abi_warnings) == 1
    assert f"expected integer result dtype for agg={agg}" in str(abi_warnings[0].message)


def _assert_single_strict_fail(
    monkeypatch: pytest.MonkeyPatch, agg: AggFunc, *, is_val_int: bool = True
) -> None:
    import pandas_booster
    import pandas_booster._abi_compat as abi
    import pandas_booster._rust as rust
    from pandas_booster.accessor import BoosterAccessor

    monkeypatch.setenv("PANDAS_BOOSTER_STRICT_ABI", "1")
    monkeypatch.delenv("PANDAS_BOOSTER_ABI_SKEW_NOTICE", raising=False)

    df = _make_single_df(is_val_int=is_val_int)
    expected = cast(pd.Series, getattr(df.groupby("key", sort=True)["val"], agg)())

    _patch_single_to_return_float_results(
        monkeypatch,
        rust,
        expected,
        agg,
        kernel="i64" if is_val_int else "f64",
    )
    monkeypatch.setattr(abi, "_WARNED_ABI_SKEW", False)

    booster = cast(BoosterAccessor, df.booster)
    with pytest.raises(
        abi.PandasBoosterKeyShapeSkewError, match=f"expected integer result dtype for agg={agg}"
    ):
        _ = booster.groupby("key", "val", agg, sort=True)

    abi._WARNED_ABI_SKEW = False
    pandas_booster.activate()
    try:
        with pytest.raises(
            abi.PandasBoosterKeyShapeSkewError, match=f"expected integer result dtype for agg={agg}"
        ):
            _ = getattr(df.groupby("key", sort=True)["val"], agg)()
    finally:
        pandas_booster.deactivate()


def _assert_multi_strict_fail(
    monkeypatch: pytest.MonkeyPatch, agg: AggFunc, *, is_val_int: bool = True
) -> None:
    import pandas_booster
    import pandas_booster._abi_compat as abi
    import pandas_booster._rust as rust
    from pandas_booster.accessor import BoosterAccessor

    monkeypatch.setenv("PANDAS_BOOSTER_STRICT_ABI", "1")
    monkeypatch.delenv("PANDAS_BOOSTER_ABI_SKEW_NOTICE", raising=False)

    df = _make_multi_df(is_val_int=is_val_int)
    by_cols = ["k1", "k2"]
    expected = cast(pd.Series, getattr(df.groupby(by_cols, sort=True)["val"], agg)())

    _patch_multi_to_return_float_results(
        monkeypatch,
        rust,
        expected,
        agg,
        kernel="i64" if is_val_int else "f64",
    )
    monkeypatch.setattr(abi, "_WARNED_ABI_SKEW", False)

    booster = cast(BoosterAccessor, df.booster)
    with pytest.raises(
        abi.PandasBoosterKeyShapeSkewError, match=f"expected integer result dtype for agg={agg}"
    ):
        _ = booster.groupby(by_cols, "val", agg, sort=True)

    abi._WARNED_ABI_SKEW = False
    pandas_booster.activate()
    try:
        with pytest.raises(
            abi.PandasBoosterKeyShapeSkewError, match=f"expected integer result dtype for agg={agg}"
        ):
            _ = getattr(df.groupby(by_cols, sort=True)["val"], agg)()
    finally:
        pandas_booster.deactivate()


@pytest.mark.parametrize("agg", ["sum", "min", "max"])
def test_single_key_integer_result_dtype_skew_falls_back_and_warns(monkeypatch, agg: AggFunc):
    _assert_single_non_strict_fallback(monkeypatch, agg)


@pytest.mark.parametrize("agg", ["sum", "min", "max"])
def test_single_key_integer_result_dtype_skew_strict_abi_hard_fails(monkeypatch, agg: AggFunc):
    _assert_single_strict_fail(monkeypatch, agg)


@pytest.mark.parametrize("agg", ["sum", "min", "max"])
def test_multi_key_integer_result_dtype_skew_falls_back_and_warns(monkeypatch, agg: AggFunc):
    _assert_multi_non_strict_fallback(monkeypatch, agg)


@pytest.mark.parametrize("agg", ["sum", "min", "max"])
def test_multi_key_integer_result_dtype_skew_strict_abi_hard_fails(monkeypatch, agg: AggFunc):
    _assert_multi_strict_fail(monkeypatch, agg)


@pytest.mark.parametrize("is_val_int", [True, False], ids=["int-values", "float-values"])
def test_single_key_count_result_dtype_skew_falls_back_and_warns(monkeypatch, is_val_int: bool):
    _assert_single_non_strict_fallback(monkeypatch, "count", is_val_int=is_val_int)


@pytest.mark.parametrize("is_val_int", [True, False], ids=["int-values", "float-values"])
def test_single_key_count_result_dtype_skew_strict_abi_hard_fails(monkeypatch, is_val_int: bool):
    _assert_single_strict_fail(monkeypatch, "count", is_val_int=is_val_int)


@pytest.mark.parametrize("is_val_int", [True, False], ids=["int-values", "float-values"])
def test_multi_key_count_result_dtype_skew_falls_back_and_warns(monkeypatch, is_val_int: bool):
    _assert_multi_non_strict_fallback(monkeypatch, "count", is_val_int=is_val_int)


@pytest.mark.parametrize("is_val_int", [True, False], ids=["int-values", "float-values"])
def test_multi_key_count_result_dtype_skew_strict_abi_hard_fails(monkeypatch, is_val_int: bool):
    _assert_multi_strict_fail(monkeypatch, "count", is_val_int=is_val_int)
