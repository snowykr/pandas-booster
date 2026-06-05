from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from typing import Any, Final, Literal, NamedTuple

import numpy as np
import pandas as pd

from ._abi_compat import raise_abi_skew
from ._groupby_result import (
    build_series_from_multi_result,
    build_series_from_single_result,
)

__all__ = [
    "AggFunc",
    "GroupByCompatibility",
    "build_series_from_multi_result",
    "build_series_from_single_result",
    "capture_key_numpy_dtype",
    "capture_value_numpy_dtype",
    "classify_groupby_compatibility",
    "firstseen_suffix",
    "get_fallback_threshold",
    "get_rust_module",
    "has_nullable_na",
    "has_rust_groupby_func",
    "is_supported_value_dtype",
    "select_rust_groupby_func",
    "should_fallback_for_key_dtype",
    "to_i64_contiguous",
]

AggFunc = Literal["sum", "mean", "prod", "min", "max", "count", "std", "var", "median"]
ORDERED_SINGLE_KEY_FLOAT_PROD_ABI_MARKER: Final = "has_ordered_single_key_float_prod_abi"
_RUST: Any | None = None
_FALLBACK_THRESHOLD: int | None = None


class GroupByCompatibility(NamedTuple):
    supported: bool
    force_pandas: bool


def get_rust_module() -> Any:
    global _RUST
    if _RUST is None:
        _RUST = importlib.import_module("pandas_booster._rust")
    return _RUST


def get_fallback_threshold() -> int:
    global _FALLBACK_THRESHOLD
    if _FALLBACK_THRESHOLD is None:
        _FALLBACK_THRESHOLD = int(get_rust_module().get_fallback_threshold())
    return _FALLBACK_THRESHOLD


def select_rust_groupby_func(
    rust: Any,
    func_base: str,
    *,
    sort: bool,
    n_rows: int,
    force_pandas_sort: bool,
    context: str | None = None,
) -> tuple[Callable[..., Any], bool]:
    """Resolve the Rust kernel for a groupby call.

    Returns (callable, needs_python_sort).
    """
    if func_base == "groupby_prod_f64" and not hasattr(
        rust, ORDERED_SINGLE_KEY_FLOAT_PROD_ABI_MARKER
    ):
        if context is None:
            raise AttributeError(ORDERED_SINGLE_KEY_FLOAT_PROD_ABI_MARKER)
        raise_abi_skew(
            context=context,
            detail=(
                "missing Rust capability marker "
                f"{ORDERED_SINGLE_KEY_FLOAT_PROD_ABI_MARKER!r} while resolving "
                "ordered single-key float 'prod'."
            ),
        )

    def _lookup(symbol: str) -> Callable[..., Any]:
        try:
            return getattr(rust, symbol)
        except AttributeError:
            if context is None:
                raise
            raise_abi_skew(
                context=context,
                detail=(f"missing Rust kernel symbol {symbol!r} while resolving {func_base!r}."),
            )

    if not sort:
        suffix = firstseen_suffix(sort=False, n_rows=n_rows)
        return _lookup(f"{func_base}{suffix}"), False

    if force_pandas_sort:
        return _lookup(func_base), True

    try:
        return getattr(rust, f"{func_base}_sorted"), False
    except AttributeError:
        # Python/Rust wheel mismatch (or older extension): fall back to the
        # legacy path and let Python sort_index() handle ordering.
        return _lookup(func_base), True


def has_rust_groupby_func(
    rust: Any,
    func_base: str,
    *,
    sort: bool,
    n_rows: int,
    force_pandas_sort: bool,
) -> bool:
    """Return whether a Rust groupby kernel is available without warning.

    This mirrors `select_rust_groupby_func` symbol selection but treats missing
    symbols as a normal negative result. It is used for staged Python dispatch
    where a compatibility policy can be certified before the matching Rust
    kernel ships.
    """
    if func_base == "groupby_prod_f64" and not hasattr(
        rust, ORDERED_SINGLE_KEY_FLOAT_PROD_ABI_MARKER
    ):
        return False

    if not sort:
        suffix = firstseen_suffix(sort=False, n_rows=n_rows)
        return hasattr(rust, f"{func_base}{suffix}")

    if force_pandas_sort:
        return hasattr(rust, func_base)

    return hasattr(rust, f"{func_base}_sorted") or hasattr(rust, func_base)


def firstseen_suffix(*, sort: bool, n_rows: int) -> str:
    if sort:
        return ""
    return "_firstseen_u32" if n_rows < (1 << 32) else "_firstseen_u64"


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
    return dtype.itemsize == 8


def capture_key_numpy_dtype(key_col: pd.Series) -> np.dtype:
    """Capture the numpy dtype that pandas would preserve for the key.

    Callers must ensure `key_col` is NOT an extension dtype.
    """
    arr = np.asarray(key_col.to_numpy(copy=False))
    return arr.dtype


def capture_value_numpy_dtype(value_col: pd.Series) -> np.dtype:
    arr = np.asarray(value_col.to_numpy(copy=False))
    return arr.dtype


def has_nullable_na(series: pd.Series) -> bool:
    if pd.api.types.is_extension_array_dtype(series):
        return bool(series.isna().any())
    return False


def is_supported_value_dtype(value_col: pd.Series, *, agg: AggFunc) -> bool:
    if pd.api.types.is_extension_array_dtype(value_col):
        return False
    if has_nullable_na(value_col):
        return False
    if pd.api.types.is_bool_dtype(value_col):
        return False
    if not (pd.api.types.is_integer_dtype(value_col) or pd.api.types.is_float_dtype(value_col)):
        return False

    value_dtype = capture_value_numpy_dtype(value_col)
    if value_dtype.kind == "u" and value_dtype.itemsize == 8:
        return False
    if agg == "prod" and value_dtype.kind == "u":
        return False
    if agg == "prod" and value_dtype.kind == "i" and value_dtype.itemsize < 8:
        return False
    return not (agg in {"std", "var"} and value_dtype.kind == "u")


def classify_groupby_compatibility(
    *,
    key_cols: Sequence[pd.Series],
    val_col: pd.Series,
    agg: AggFunc,
    force_pandas_float_groupby: bool,
) -> GroupByCompatibility:
    for key_col in key_cols:
        if not pd.api.types.is_integer_dtype(key_col):
            return GroupByCompatibility(False, False)
        if should_fallback_for_key_dtype(key_col):
            return GroupByCompatibility(False, False)
        if has_nullable_na(key_col):
            return GroupByCompatibility(False, False)

    if not is_supported_value_dtype(val_col, agg=agg):
        return GroupByCompatibility(False, False)

    if (
        len(key_cols) == 1
        and agg in {"sum", "mean", "prod", "std", "var", "median"}
        and pd.api.types.is_float_dtype(val_col)
        and force_pandas_float_groupby
    ):
        return GroupByCompatibility(True, True)

    return GroupByCompatibility(True, False)


def to_i64_contiguous(arr: np.ndarray) -> np.ndarray:
    """Convert to contiguous int64 ndarray for Rust."""
    return np.ascontiguousarray(arr.astype(np.int64, copy=False))
