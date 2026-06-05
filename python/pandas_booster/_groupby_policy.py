from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

import pandas as pd

from . import _groupby_accel as _groupby_accel_mod
from ._config import (
    force_pandas_float_groupby_enabled,
    force_pandas_sort_enabled,
)
from ._groupby_accel import AggFunc

_FALLBACK_EXEMPT_AGGS: Final = frozenset({"std", "var", "median"})
_MAX_RUST_MULTI_KEYS: Final = 10


def resolve_rust_module(df: pd.DataFrame, *, context: str) -> Any:
    if context == "accessor":
        accessor = df.booster
        return accessor._get_rust_module()
    return _groupby_accel_mod.get_rust_module()


def should_fallback_groupby(
    df: pd.DataFrame,
    key_cols: Sequence[pd.Series],
    val_col: pd.Series,
    agg: AggFunc,
    *,
    context: str,
    multi: bool,
    sort: bool,
) -> bool:
    if multi and len(key_cols) > _MAX_RUST_MULTI_KEYS:
        return True

    if agg not in _FALLBACK_EXEMPT_AGGS and len(df) < _resolve_fallback_threshold(
        df, context=context
    ):
        return True

    compatibility = _groupby_accel_mod.classify_groupby_compatibility(
        key_cols=key_cols,
        val_col=val_col,
        agg=agg,
        force_pandas_float_groupby=force_pandas_float_groupby_enabled(),
    )
    if not compatibility.supported or compatibility.force_pandas:
        return True

    if agg != "median":
        return False

    return not _has_median_kernel(
        resolve_rust_module(df, context=context),
        df,
        val_col,
        multi=multi,
        sort=sort,
    )


def _resolve_fallback_threshold(df: pd.DataFrame, *, context: str) -> int:
    if context == "accessor":
        return int(resolve_rust_module(df, context=context).get_fallback_threshold())
    return _groupby_accel_mod.get_fallback_threshold()


def _has_median_kernel(
    rust: Any,
    df: pd.DataFrame,
    val_col: pd.Series,
    *,
    multi: bool,
    sort: bool,
) -> bool:
    kernel = "i64" if pd.api.types.is_integer_dtype(val_col) else "f64"
    prefix = "groupby_multi" if multi else "groupby"
    return _groupby_accel_mod.has_rust_groupby_func(
        rust,
        f"{prefix}_median_{kernel}",
        sort=sort,
        n_rows=len(df),
        force_pandas_sort=bool(sort) and force_pandas_sort_enabled(),
    )
