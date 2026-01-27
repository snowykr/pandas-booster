from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pandas as pd
import pytest
from pandas_booster.accessor import BoosterAccessor

AggFunc = Literal["sum", "mean", "min", "max", "count"]


def _assert_groupby_series_equal(
    booster_result: pd.Series, pandas_result: pd.Series, *, agg: AggFunc
) -> None:
    # For floating point reductions, allow small numerical differences.
    # Parallel reductions can differ by operation order (non-associativity).
    if agg in {"min", "max", "count"}:
        pd.testing.assert_series_equal(booster_result, pandas_result, check_exact=True)
        return

    pd.testing.assert_series_equal(
        booster_result,
        pandas_result,
        check_exact=False,
        rtol=1e-9,
        atol=1e-12,
    )


@pytest.mark.parametrize("agg", ["sum", "mean", "min", "max", "count"])
def test_single_key_sort_true_matches_pandas_order(monkeypatch: pytest.MonkeyPatch, agg: AggFunc):
    monkeypatch.delenv("PANDAS_BOOSTER_FORCE_PANDAS_SORT", raising=False)

    np.random.seed(123)
    n = 200_000
    df = pd.DataFrame(
        {
            "key": np.random.randint(-10_000, 10_000, size=n, dtype=np.int64),
            "val": np.random.random(size=n).astype(np.float64),
        }
    )

    booster_result = cast(BoosterAccessor, df.booster).groupby("key", "val", agg, sort=True)
    pandas_result = getattr(df.groupby("key", sort=True)["val"], agg)()

    _assert_groupby_series_equal(booster_result, pandas_result, agg=agg)


def test_single_key_uint32_boundary_order(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("PANDAS_BOOSTER_FORCE_PANDAS_SORT", raising=False)

    np.random.seed(456)
    n = 200_000
    low = (1 << 32) - 10_000
    high = (1 << 32) - 1
    df = pd.DataFrame(
        {
            "key": np.random.randint(low, high, size=n, dtype=np.int64),
            "val": np.random.random(size=n).astype(np.float64),
        }
    )

    booster_result = cast(BoosterAccessor, df.booster).groupby("key", "val", "sum", sort=True)
    pandas_result = df.groupby("key", sort=True)["val"].sum()

    _assert_groupby_series_equal(booster_result, pandas_result, agg="sum")


@pytest.mark.parametrize("n_keys", [2, 3, 4, 5, 10])
def test_multi_key_sort_true_matches_pandas_order(monkeypatch: pytest.MonkeyPatch, n_keys: int):
    monkeypatch.delenv("PANDAS_BOOSTER_FORCE_PANDAS_SORT", raising=False)

    np.random.seed(42 + n_keys)
    n = 200_000
    data: dict[str, np.ndarray] = {}
    for i in range(n_keys):
        data[f"k{i}"] = np.random.randint(-50, 50, size=n, dtype=np.int64)
    data["val"] = np.random.random(size=n).astype(np.float64)
    df = pd.DataFrame(data)

    by_cols = [f"k{i}" for i in range(n_keys)]

    booster_result = cast(BoosterAccessor, df.booster).groupby(by_cols, "val", "sum", sort=True)
    pandas_result = df.groupby(by_cols, sort=True)["val"].sum()

    _assert_groupby_series_equal(booster_result, pandas_result, agg="sum")


def test_multi_key_mixed_int_dtypes_preserved(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("PANDAS_BOOSTER_FORCE_PANDAS_SORT", raising=False)

    np.random.seed(999)
    n = 200_000
    df = pd.DataFrame(
        {
            "k1": np.random.randint(-100, 100, size=n, dtype=np.int32),
            "k2": np.random.randint(-1000, 1000, size=n, dtype=np.int64),
            "val": np.random.random(size=n).astype(np.float64),
        }
    )

    booster_result = cast(BoosterAccessor, df.booster).groupby(
        ["k1", "k2"], "val", "sum", sort=True
    )
    pandas_result = df.groupby(["k1", "k2"], sort=True)["val"].sum()

    _assert_groupby_series_equal(booster_result, pandas_result, agg="sum")

    assert isinstance(booster_result.index, pd.MultiIndex)
    assert booster_result.index.names == ["k1", "k2"]
    assert booster_result.index.levels[0].dtype == np.dtype(np.int32)
    assert booster_result.index.levels[1].dtype == np.dtype(np.int64)


def test_default_no_python_sort_when_rust_sorted(monkeypatch: pytest.MonkeyPatch):
    import pandas_booster

    rust = getattr(pandas_booster, "_rust", None)
    if rust is None or not hasattr(rust, "groupby_multi_sum_f64_sorted"):
        pytest.skip(
            "Rust sorted kernels not available (likely Python/Rust wheel mismatch); "
            "panic-button test requires groupby_multi_sum_f64_sorted"
        )

    np.random.seed(2024)
    n = 200_000
    df = pd.DataFrame(
        {
            "k1": np.random.randint(0, 100, size=n, dtype=np.int64),
            "k2": np.random.randint(0, 50, size=n, dtype=np.int64),
            "val": np.random.random(size=n).astype(np.float64),
        }
    )

    # Compute the Pandas baseline before monkeypatching. This keeps the test
    # robust against potential future Pandas internal changes that might call
    # Series.sort_index() as part of groupby(sort=True) execution.
    pandas_on = df.groupby(["k1", "k2"], sort=True)["val"].sum()

    def _raise_sort_index(_self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        _ = (_self, _args, _kwargs)
        raise AssertionError("Series.sort_index() should not be called")

    monkeypatch.delenv("PANDAS_BOOSTER_FORCE_PANDAS_SORT", raising=False)
    monkeypatch.setattr(pd.Series, "sort_index", _raise_sort_index, raising=True)

    booster_on = cast(BoosterAccessor, df.booster).groupby(["k1", "k2"], "val", "sum", sort=True)
    _assert_groupby_series_equal(booster_on, pandas_on, agg="sum")

    pandas_booster.activate()
    try:
        proxy_on = df.groupby(["k1", "k2"], sort=True)["val"].sum()
    finally:
        pandas_booster.deactivate()

    _assert_groupby_series_equal(proxy_on, pandas_on, agg="sum")

    # Force Python sort: should call sort_index() in Python builder.
    monkeypatch.setenv("PANDAS_BOOSTER_FORCE_PANDAS_SORT", "1")
    with pytest.raises(AssertionError, match="should not be called"):
        cast(BoosterAccessor, df.booster).groupby(["k1", "k2"], "val", "sum", sort=True)


@pytest.mark.parametrize("env_value", ["0", "false"])
def test_force_pandas_sort_false_does_not_call_sort_index(
    monkeypatch: pytest.MonkeyPatch, env_value: str
):
    import pandas_booster

    rust = getattr(pandas_booster, "_rust", None)
    if rust is None or not hasattr(rust, "groupby_multi_sum_f64_sorted"):
        pytest.skip(
            "Rust sorted kernels not available (likely Python/Rust wheel mismatch); "
            "panic-button test requires groupby_multi_sum_f64_sorted"
        )

    np.random.seed(2026)
    n = 200_000
    df = pd.DataFrame(
        {
            "k1": np.random.randint(0, 100, size=n, dtype=np.int64),
            "k2": np.random.randint(0, 50, size=n, dtype=np.int64),
            "val": np.random.random(size=n).astype(np.float64),
        }
    )

    pandas_on = df.groupby(["k1", "k2"], sort=True)["val"].sum()

    def _raise_sort_index(_self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        _ = (_self, _args, _kwargs)
        raise AssertionError("Series.sort_index() should not be called")

    monkeypatch.setenv("PANDAS_BOOSTER_FORCE_PANDAS_SORT", env_value)
    monkeypatch.setattr(pd.Series, "sort_index", _raise_sort_index, raising=True)

    booster_on = cast(BoosterAccessor, df.booster).groupby(["k1", "k2"], "val", "sum", sort=True)

    _assert_groupby_series_equal(booster_on, pandas_on, agg="sum")


@pytest.mark.parametrize("agg", ["mean", "min", "max", "count"])
def test_default_no_python_sort_for_other_aggs_when_rust_sorted(
    monkeypatch: pytest.MonkeyPatch, agg: AggFunc
) -> None:
    import pandas_booster

    rust = getattr(pandas_booster, "_rust", None)
    if rust is None:
        pytest.skip("Rust extension module not available")

    sorted_symbol = f"groupby_multi_{agg}_f64_sorted"
    if not hasattr(rust, sorted_symbol):
        pytest.skip(
            f"Rust sorted kernel {sorted_symbol} not available (likely Python/Rust wheel mismatch)"
        )

    np.random.seed(2025)
    n = 200_000
    df = pd.DataFrame(
        {
            "k1": np.random.randint(0, 100, size=n, dtype=np.int64),
            "k2": np.random.randint(0, 50, size=n, dtype=np.int64),
            "val": np.random.random(size=n).astype(np.float64),
        }
    )

    # Compute the Pandas baseline before monkeypatching (see sum test above).
    pandas_on = getattr(df.groupby(["k1", "k2"], sort=True)["val"], agg)()

    def _raise_sort_index(_self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        _ = (_self, _args, _kwargs)
        raise AssertionError("Series.sort_index() should not be called")

    monkeypatch.delenv("PANDAS_BOOSTER_FORCE_PANDAS_SORT", raising=False)
    monkeypatch.setattr(pd.Series, "sort_index", _raise_sort_index, raising=True)

    booster_on = cast(BoosterAccessor, df.booster).groupby(["k1", "k2"], "val", agg, sort=True)
    _assert_groupby_series_equal(booster_on, pandas_on, agg=agg)

    pandas_booster.activate()
    try:
        proxy_on = getattr(df.groupby(["k1", "k2"], sort=True)["val"], agg)()
    finally:
        pandas_booster.deactivate()

    _assert_groupby_series_equal(proxy_on, pandas_on, agg=agg)


def test_force_pandas_sort_matches_pandas_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PANDAS_BOOSTER_FORCE_PANDAS_SORT", "1")

    np.random.seed(2027)
    n = 200_000
    df = pd.DataFrame(
        {
            "k1": np.random.randint(0, 100, size=n, dtype=np.int64),
            "k2": np.random.randint(0, 50, size=n, dtype=np.int64),
            "val": np.random.random(size=n).astype(np.float64),
        }
    )

    booster_on = cast(BoosterAccessor, df.booster).groupby(["k1", "k2"], "val", "sum", sort=True)
    pandas_on = df.groupby(["k1", "k2"], sort=True)["val"].sum()

    _assert_groupby_series_equal(booster_on, pandas_on, agg="sum")
