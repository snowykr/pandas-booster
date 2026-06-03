"""Shared helpers for edge-case contract tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _accessor_groupby_result(
    df: pd.DataFrame,
    by: str | list[str],
    target: str,
    agg: str,
    *,
    sort: bool = True,
) -> pd.Series:
    import pandas_booster  # noqa: F401

    return df.booster.groupby(by, target, agg, sort=sort)


def _proxy_groupby_result(
    df: pd.DataFrame,
    by: str | list[str],
    target: str,
    agg: str,
    *,
    sort: bool = True,
) -> pd.Series:
    import pandas_booster

    pandas_booster.activate()
    try:
        return getattr(df.groupby(by, sort=sort)[target], agg)()
    finally:
        pandas_booster.deactivate()


def _patch_single_std_var_kernel(
    monkeypatch: pytest.MonkeyPatch,
    rust: object,
    expected: pd.Series,
    agg: str,
    *,
    kernel: str,
    result_dtype: np.dtype,
) -> None:
    def fake_groupby(_keys_arr, _values_arr):
        return (
            np.asarray(expected.index.to_numpy(), dtype=np.int64),
            np.asarray(expected.to_numpy(), dtype=result_dtype),
        )

    for suffix in ("", "_sorted", "_firstseen_u32", "_firstseen_u64"):
        monkeypatch.setattr(rust, f"groupby_{agg}_{kernel}{suffix}", fake_groupby, raising=False)


def _patch_multi_std_var_kernel(
    monkeypatch: pytest.MonkeyPatch,
    rust: object,
    expected: pd.Series,
    agg: str,
    *,
    kernel: str,
    result_dtype: np.dtype,
) -> None:
    def fake_groupby(_key_arrays, _values_arr):
        keys_cols = [
            np.asarray(expected.index.get_level_values(i), dtype=np.int64)
            for i in range(expected.index.nlevels)
        ]
        return keys_cols, np.asarray(expected.to_numpy(), dtype=result_dtype)

    for suffix in ("", "_sorted", "_firstseen_u32", "_firstseen_u64"):
        monkeypatch.setattr(
            rust,
            f"groupby_multi_{agg}_{kernel}{suffix}",
            fake_groupby,
            raising=False,
        )


def _patch_all_std_var_kernels_to_raise(
    monkeypatch: pytest.MonkeyPatch, rust: object, message: str
) -> None:
    def _boom(*_args, **_kwargs):
        raise AssertionError(message)

    for agg in ("std", "var"):
        for kernel in ("f64", "i64"):
            for prefix in ("groupby", "groupby_multi"):
                for suffix in ("", "_sorted", "_firstseen_u32", "_firstseen_u64"):
                    monkeypatch.setattr(
                        rust,
                        f"{prefix}_{agg}_{kernel}{suffix}",
                        _boom,
                        raising=False,
                    )


def _patch_all_i64_kernels_for_agg_to_raise(
    monkeypatch: pytest.MonkeyPatch, rust: object, agg: str, message: str
) -> None:
    def _boom(*_args, **_kwargs):
        raise AssertionError(message)

    for prefix in ("groupby", "groupby_multi"):
        for suffix in ("", "_sorted", "_firstseen_u32", "_firstseen_u64"):
            monkeypatch.setattr(rust, f"{prefix}_{agg}_i64{suffix}", _boom, raising=False)


def _patch_all_numeric_kernels_for_agg_to_raise(
    monkeypatch: pytest.MonkeyPatch, rust: object, agg: str, message: str
) -> None:
    def _boom(*_args, **_kwargs):
        raise AssertionError(message)

    for kernel in ("f64", "i64"):
        for prefix in ("groupby", "groupby_multi"):
            for suffix in ("", "_sorted", "_firstseen_u32", "_firstseen_u64"):
                monkeypatch.setattr(rust, f"{prefix}_{agg}_{kernel}{suffix}", _boom, raising=False)


def _delete_groupby_kernel_symbols(
    monkeypatch: pytest.MonkeyPatch,
    rust: object,
    agg: str,
    *,
    kernel: str,
    multi: bool,
) -> None:
    prefix = "groupby_multi" if multi else "groupby"
    for suffix in ("", "_sorted", "_firstseen_u32", "_firstseen_u64"):
        monkeypatch.delattr(rust, f"{prefix}_{agg}_{kernel}{suffix}", raising=False)


def _patch_pandas_series_groupby_agg_to_raise(
    monkeypatch: pytest.MonkeyPatch, agg: str, message: str
) -> None:
    from pandas.core.groupby.generic import SeriesGroupBy

    def _boom(self, *args, **kwargs):
        raise AssertionError(message)

    monkeypatch.setattr(SeriesGroupBy, agg, _boom, raising=True)


def _patch_single_std_var_firstseen_only_kernel(
    monkeypatch: pytest.MonkeyPatch,
    rust: object,
    expected: pd.Series,
    agg: str,
    *,
    kernel: str,
    result_dtype: np.dtype,
    calls: list[str],
) -> None:
    def fake_groupby(_keys_arr, _values_arr):
        calls.append("firstseen")
        return (
            np.asarray(expected.index.to_numpy(), dtype=np.int64),
            np.asarray(expected.to_numpy(), dtype=result_dtype),
        )

    def _boom(*_args, **_kwargs):
        raise AssertionError(f"sort=False {agg} should route through first-seen kernels")

    monkeypatch.setattr(rust, f"groupby_{agg}_{kernel}", _boom, raising=False)
    monkeypatch.setattr(rust, f"groupby_{agg}_{kernel}_sorted", _boom, raising=False)
    monkeypatch.setattr(rust, f"groupby_{agg}_{kernel}_firstseen_u32", fake_groupby, raising=False)
    monkeypatch.setattr(rust, f"groupby_{agg}_{kernel}_firstseen_u64", fake_groupby, raising=False)
