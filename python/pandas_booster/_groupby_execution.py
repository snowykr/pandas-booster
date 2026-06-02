from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd

from . import _abi_compat
from . import _groupby_accel as _groupby_accel_mod
from ._abi_compat import PandasBoosterKeyShapeSkewError, raise_abi_skew
from ._config import (
    force_pandas_sort_enabled,
    strict_abi_enabled,
)
from ._groupby_accel import AggFunc
from ._groupby_policy import resolve_rust_module, should_fallback_groupby
from ._groupby_result import (
    build_series_from_multi_result,
    build_series_from_single_result,
)


class GroupByValueInputs(NamedTuple):
    func_base: str
    is_val_int: bool
    value_dtype: np.dtype
    values: np.ndarray


def _prepare_groupby_value_inputs(
    val_col: pd.Series, *, agg: AggFunc, func_prefix: str
) -> GroupByValueInputs:
    value_dtype = _groupby_accel_mod.capture_value_numpy_dtype(val_col)
    is_val_int = pd.api.types.is_integer_dtype(val_col)
    if is_val_int:
        values = np.ascontiguousarray(val_col.to_numpy(dtype=np.int64))
        func_base = f"{func_prefix}{agg}_i64"
    else:
        values = np.ascontiguousarray(val_col.to_numpy(dtype=np.float64))
        func_base = f"{func_prefix}{agg}_f64"
    return GroupByValueInputs(
        func_base=func_base,
        is_val_int=is_val_int,
        value_dtype=value_dtype,
        values=values,
    )


def execute_groupby_single(
    df: pd.DataFrame,
    key_col: pd.Series,
    val_col: pd.Series,
    agg: AggFunc,
    *,
    sort: bool,
    context: str,
    fallback: Callable[[], pd.Series],
) -> pd.Series:
    if should_fallback_groupby(
        df,
        [key_col],
        val_col,
        agg,
        context=context,
        multi=False,
        sort=sort,
    ):
        return fallback()

    rust = resolve_rust_module(df, context=context)
    strict = strict_abi_enabled()
    force_pandas_sort = bool(sort) and force_pandas_sort_enabled()
    key_dtype = _groupby_accel_mod.capture_key_numpy_dtype(key_col)
    value_inputs = _prepare_groupby_value_inputs(val_col, agg=agg, func_prefix="groupby_")
    keys = _groupby_accel_mod.to_i64_contiguous(key_col.to_numpy(copy=False))

    try:
        rust_func, needs_python_sort = _groupby_accel_mod.select_rust_groupby_func(
            rust,
            value_inputs.func_base,
            sort=sort,
            n_rows=len(df),
            force_pandas_sort=force_pandas_sort,
            context=context,
        )
        result_keys, result_values = rust_func(keys, value_inputs.values)
        result_values_arr = _abi_compat.normalize_result_values(
            result_values,
            agg=agg,
            is_val_int=value_inputs.is_val_int,
            context=context,
        )
        return build_series_from_single_result(
            np.asarray(result_keys),
            result_values_arr,
            name=val_col.name,
            index_name=key_col.name,
            index_dtype=key_dtype,
            value_dtype=value_inputs.value_dtype,
            agg=agg,
            is_val_int=value_inputs.is_val_int,
            sort=sort,
            needs_python_sort=needs_python_sort,
        )
    except PandasBoosterKeyShapeSkewError:
        if strict:
            raise
        return fallback()


def execute_groupby_multi(
    df: pd.DataFrame,
    key_cols: Sequence[tuple[str, pd.Series]],
    val_col: pd.Series,
    agg: AggFunc,
    *,
    sort: bool,
    context: str,
    fallback: Callable[[], pd.Series],
) -> pd.Series:
    by_cols = [col_name for col_name, _ in key_cols]
    key_series = [key_col for _, key_col in key_cols]
    if should_fallback_groupby(
        df,
        key_series,
        val_col,
        agg,
        context=context,
        multi=True,
        sort=sort,
    ):
        return fallback()

    rust = resolve_rust_module(df, context=context)
    strict = strict_abi_enabled()
    force_pandas_sort = bool(sort) and force_pandas_sort_enabled()
    key_dtypes = [_groupby_accel_mod.capture_key_numpy_dtype(key_col) for key_col in key_series]
    key_arrays = [
        _groupby_accel_mod.to_i64_contiguous(key_col.to_numpy(copy=False)) for key_col in key_series
    ]
    value_inputs = _prepare_groupby_value_inputs(val_col, agg=agg, func_prefix="groupby_multi_")

    try:
        rust_func, needs_python_sort = _groupby_accel_mod.select_rust_groupby_func(
            rust,
            value_inputs.func_base,
            sort=sort,
            n_rows=len(df),
            force_pandas_sort=force_pandas_sort,
            context=context,
        )
        rust_result = rust_func(key_arrays, value_inputs.values)
        if not isinstance(rust_result, tuple) or len(rust_result) != 2:
            raise_abi_skew(
                context=context,
                detail=(
                    "expected Rust groupby_multi result as (keys_cols, result_values) tuple; "
                    f"got type={type(rust_result)!r}."
                    if not isinstance(rust_result, tuple)
                    else f"expected 2-tuple (keys_cols, result_values); got len={len(rust_result)}."
                ),
            )
        keys_cols, result_values = rust_result
        result_values_arr = _abi_compat.normalize_result_values(
            result_values,
            agg=agg,
            is_val_int=value_inputs.is_val_int,
            context=context,
        )
        keys_cols_arr = _abi_compat.normalize_multi_keys_cols(
            keys_cols,
            n_groups=result_values_arr.shape[0],
            n_keys=len(by_cols),
            context=context,
            strict=strict,
        )
        return build_series_from_multi_result(
            keys_cols_arr,
            result_values_arr,
            by_cols=by_cols,
            key_dtypes=key_dtypes,
            name=val_col.name,
            value_dtype=value_inputs.value_dtype,
            agg=agg,
            is_val_int=value_inputs.is_val_int,
            sort=sort,
            needs_python_sort=needs_python_sort,
        )
    except PandasBoosterKeyShapeSkewError:
        if strict:
            raise
        return fallback()
