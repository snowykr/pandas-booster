from __future__ import annotations

"""Pandas DataFrame accessor for Rust-accelerated groupby operations."""

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pandas import DataFrame, Series

AggFunc = Literal["sum", "mean", "min", "max"]


@pd.api.extensions.register_dataframe_accessor("booster")
class BoosterAccessor:
    """Pandas DataFrame accessor providing Rust-accelerated groupby operations.

    Automatically falls back to native Pandas when:
    - Dataset has fewer than 100,000 rows
    - Key column is not integer dtype
    - Value column is not numeric (int/float)
    - Columns contain nullable NA values (pd.NA)

    Examples:
        >>> df = pd.DataFrame({"key": [1, 2, 1], "val": [10.0, 20.0, 30.0]})
        >>> df.booster.groupby("key", "val", "sum")
    """

    _SUPPORTED_AGGS: set[str] = {"sum", "mean", "min", "max"}

    def __init__(self, pandas_obj: DataFrame) -> None:
        self._df = pandas_obj
        self._rust = None
        self._fallback_threshold = None

    def _get_rust_module(self):
        if self._rust is None:
            from pandas_booster import _rust

            self._rust = _rust
        return self._rust

    def _get_fallback_threshold(self) -> int:
        if self._fallback_threshold is None:
            self._fallback_threshold = self._get_rust_module().get_fallback_threshold()
        return self._fallback_threshold

    def groupby(
        self,
        by: str,
        target: str,
        agg: AggFunc,
    ) -> Series:
        """Perform accelerated groupby aggregation.

        Args:
            by: Column name to group by (must be integer dtype for acceleration).
            target: Column name to aggregate (must be numeric).
            agg: Aggregation function - one of "sum", "mean", "min", "max".

        Returns:
            Series indexed by unique group keys with aggregated values.

        Raises:
            ValueError: If agg is not a supported aggregation function.
        """
        if agg not in self._SUPPORTED_AGGS:
            raise ValueError(f"Unsupported aggregation: {agg}. Use one of {self._SUPPORTED_AGGS}")

        key_col = self._df[by]
        val_col = self._df[target]

        if len(self._df) < self._get_fallback_threshold():
            return self._pandas_fallback(by, target, agg)

        if not pd.api.types.is_integer_dtype(key_col):
            return self._pandas_fallback(by, target, agg)

        if not self._is_numeric_dtype(val_col):
            return self._pandas_fallback(by, target, agg)

        if self._has_nullable_na(key_col) or self._has_nullable_na(val_col):
            return self._pandas_fallback(by, target, agg)

        return self._rust_groupby(key_col, val_col, agg)

    def _is_numeric_dtype(self, series: Series) -> bool:
        return pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series)

    def _has_nullable_na(self, series: Series) -> bool:
        if pd.api.types.is_extension_array_dtype(series):
            return series.isna().any()
        return False

    def _pandas_fallback(self, by: str, target: str, agg: str) -> Series:
        grouped = self._df.groupby(by)[target]
        return getattr(grouped, agg)()

    def _rust_groupby(self, key_col: Series, val_col: Series, agg: str) -> Series:
        rust = self._get_rust_module()

        keys = np.ascontiguousarray(key_col.to_numpy(dtype=np.int64))
        is_val_int = pd.api.types.is_integer_dtype(val_col)

        if is_val_int:
            values = np.ascontiguousarray(val_col.to_numpy(dtype=np.int64))
            func_name = f"groupby_{agg}_i64"
        else:
            values = np.ascontiguousarray(val_col.to_numpy(dtype=np.float64))
            func_name = f"groupby_{agg}_f64"

        rust_func = getattr(rust, func_name)
        result_dict = rust_func(keys, values)

        result = pd.Series(result_dict, name=val_col.name)
        result.index.name = key_col.name
        return result

    def thread_count(self) -> int:
        """Return the number of threads used by the Rust parallel runtime."""
        rust = self._get_rust_module()
        return rust.get_thread_count()
