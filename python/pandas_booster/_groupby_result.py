from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import pandas as pd


def build_series_from_single_result(
    keys_1d: np.ndarray,
    result_values: np.ndarray,
    *,
    name: Hashable | None,
    index_name: Hashable | None,
    index_dtype: np.dtype,
    value_dtype: np.dtype,
    agg: str,
    is_val_int: bool,
    sort: bool,
    needs_python_sort: bool = False,
) -> pd.Series:
    if keys_1d.ndim != 1:
        raise ValueError(f"keys_1d must be 1D, got ndim={keys_1d.ndim}")
    if result_values.ndim != 1:
        raise ValueError(f"result_values must be 1D, got ndim={result_values.ndim}")
    if keys_1d.shape[0] != result_values.shape[0]:
        raise ValueError(
            f"keys_1d length {keys_1d.shape[0]} != result_values length {result_values.shape[0]}"
        )

    if keys_1d.shape[0] == 0:
        idx = pd.Index([], dtype=index_dtype, name=index_name)
        out_dtype = (
            np.int64
            if agg == "count"
            else value_dtype
            if is_val_int and agg in {"sum", "prod", "min", "max"}
            else np.float64
        )
        return pd.Series([], index=idx, name=name, dtype=out_dtype)

    keys_arr = np.asarray(keys_1d).astype(index_dtype, copy=False)
    idx = pd.Index(keys_arr, dtype=index_dtype, name=index_name, copy=False)

    values_arr = np.asarray(result_values)
    if agg == "count":
        values_arr = values_arr.astype(np.int64, copy=False)
    elif is_val_int and agg in {"sum", "prod", "min", "max"}:
        values_arr = values_arr.astype(value_dtype, copy=False)
    else:
        values_arr = values_arr.astype(np.float64, copy=False)

    result = pd.Series(values_arr, index=idx, name=name)

    if sort and needs_python_sort:
        result = result.sort_index()
    return result


def build_series_from_multi_result(
    keys_cols: list[np.ndarray],
    result_values: np.ndarray,
    *,
    by_cols: list[str],
    key_dtypes: list[np.dtype],
    name: Hashable | None,
    value_dtype: np.dtype,
    agg: str,
    is_val_int: bool,
    sort: bool,
    needs_python_sort: bool = False,
) -> pd.Series:
    n_keys = len(by_cols)

    if len(key_dtypes) != n_keys:
        raise ValueError(f"key_dtypes length mismatch: expected {n_keys}, got {len(key_dtypes)}")

    if len(keys_cols) != n_keys:
        raise ValueError(f"keys_cols length mismatch: expected {n_keys}, got {len(keys_cols)}")

    if result_values.ndim != 1:
        raise ValueError(f"result_values must be 1D, got ndim={result_values.ndim}")

    n_groups = result_values.shape[0]
    if n_groups == 0:
        empty_arrays = [np.array([], dtype=key_dtypes[i]) for i in range(n_keys)]
        idx = pd.MultiIndex.from_arrays(empty_arrays, names=by_cols)

        out_dtype = (
            np.int64
            if agg == "count"
            else value_dtype
            if is_val_int and agg in {"sum", "prod", "min", "max"}
            else np.float64
        )
        return pd.Series([], index=idx, name=name, dtype=out_dtype)

    # Cast each level array BEFORE MultiIndex construction to ensure level dtype preservation.
    # keys_cols is expected to be per-column 1D arrays (from Rust or normalization).
    index_arrays: list[np.ndarray] = []
    for i in range(n_keys):
        arr = np.asarray(keys_cols[i])
        if arr.ndim != 1:
            raise ValueError(f"keys_cols[{i}] must be 1D, got ndim={arr.ndim}")
        if arr.shape[0] != n_groups:
            raise ValueError(
                f"keys_cols[{i}] length {arr.shape[0]} != result_values length {n_groups}"
            )
        index_arrays.append(arr.astype(key_dtypes[i], copy=False))
    idx = pd.MultiIndex.from_arrays(index_arrays, names=by_cols)

    values_arr = np.asarray(result_values)
    if agg == "count":
        values_arr = values_arr.astype(np.int64, copy=False)
    elif is_val_int and agg in {"sum", "prod", "min", "max"}:
        values_arr = values_arr.astype(value_dtype, copy=False)
    else:
        values_arr = values_arr.astype(np.float64, copy=False)

    result = pd.Series(values_arr, index=idx, name=name)

    if sort and needs_python_sort:
        result = result.sort_index()
    return result
