from __future__ import annotations

from typing import Hashable, Literal

import numpy as np
import pandas as pd

AggFunc = Literal["sum", "mean", "min", "max", "count"]


def should_fallback_for_key_dtype(key_col: pd.Series) -> bool:
    """Return True if we should not accelerate due to key dtype.

    Reasons:
    - Extension dtypes (e.g., pandas nullable Int64) are not supported.
    - Unsigned uint64 keys may overflow when cast to int64.
      Unsigned <= 32-bit keys (uint8/uint16/uint32) are safe to cast to int64.
    """
    if pd.api.types.is_extension_array_dtype(key_col):
        return True

    arr = np.asarray(key_col.to_numpy(copy=False))
    dtype = arr.dtype
    if dtype.kind != "u":
        return False

    # uint64 can contain values > int64 max; casting to int64 would wrap.
    # We don't attempt to prove safety here; just fall back.
    if dtype.itemsize == 8:
        return True

    return False


def capture_key_numpy_dtype(key_col: pd.Series) -> np.dtype:
    """Capture the numpy dtype that pandas would preserve for the key.

    Callers must ensure `key_col` is NOT an extension dtype.
    """
    arr = np.asarray(key_col.to_numpy(copy=False))
    return arr.dtype


def to_i64_contiguous(arr: np.ndarray) -> np.ndarray:
    """Convert to contiguous int64 ndarray for Rust."""
    return np.ascontiguousarray(arr.astype(np.int64, copy=False))


def build_series_from_single_result(
    keys_1d: np.ndarray,
    result_values: np.ndarray,
    *,
    name: Hashable | None,
    index_name: Hashable | None,
    index_dtype: np.dtype,
    agg: str,
    sort: bool,
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
        out_dtype = np.int64 if agg == "count" else np.float64
        return pd.Series([], index=idx, name=name, dtype=out_dtype)

    keys_arr = np.asarray(keys_1d).astype(index_dtype, copy=False)
    idx = pd.Index(keys_arr, dtype=index_dtype, name=index_name, copy=False)

    values_arr = np.asarray(result_values)
    if agg != "count":
        values_arr = values_arr.astype(np.float64, copy=False)

    result = pd.Series(values_arr, index=idx, name=name)
    if agg == "count":
        result = result.astype(np.int64)

    if sort:
        result = result.sort_index()
    return result


def build_series_from_multi_result(
    keys_2d: np.ndarray,
    result_values: np.ndarray,
    *,
    by_cols: list[str],
    key_dtypes: list[np.dtype],
    name: Hashable | None,
    agg: str,
    sort: bool,
) -> pd.Series:
    n_keys = len(by_cols)

    if len(key_dtypes) != n_keys:
        raise ValueError(
            f"key_dtypes length mismatch: expected {n_keys}, got {len(key_dtypes)}"
        )

    if keys_2d.ndim != 2:
        raise ValueError(f"keys_2d must be 2D, got ndim={keys_2d.ndim}")

    if keys_2d.shape[1] != n_keys:
        raise ValueError(
            f"keys_2d column mismatch: expected {n_keys}, got {keys_2d.shape[1]}"
        )

    if keys_2d.shape[0] == 0:
        empty_arrays = [np.array([], dtype=key_dtypes[i]) for i in range(n_keys)]
        idx = pd.MultiIndex.from_arrays(empty_arrays, names=by_cols)

        out_dtype = np.int64 if agg == "count" else np.float64
        return pd.Series([], index=idx, name=name, dtype=out_dtype)

    # Cast each level array BEFORE MultiIndex construction to ensure level dtype preservation.
    index_arrays = [
        np.ascontiguousarray(keys_2d[:, i]).astype(key_dtypes[i], copy=False)
        for i in range(n_keys)
    ]
    idx = pd.MultiIndex.from_arrays(index_arrays, names=by_cols)

    result = pd.Series(result_values, index=idx, name=name)
    if agg == "count":
        result = result.astype(np.int64)

    if sort:
        result = result.sort_index()
    return result
