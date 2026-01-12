from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pandas import DataFrame, Series

AggFunc = Literal["sum", "mean", "min", "max", "count"]


@pd.api.extensions.register_dataframe_accessor("booster")
class BoosterAccessor:
    """Pandas DataFrame accessor providing Rust-accelerated groupby operations.

    Automatically falls back to native Pandas when:
    - Dataset has fewer than 100,000 rows
    - Key column is not integer dtype
    - Value column is not numeric (int/float)
    - Columns contain nullable NA values (pd.NA)

    Examples:
        Single key:
        >>> df = pd.DataFrame({"key": [1, 2, 1], "val": [10.0, 20.0, 30.0]})
        >>> df.booster.groupby("key", "val", "sum")

        Multiple keys:
        >>> df = pd.DataFrame({"k1": [1, 1, 2], "k2": [10, 20, 10], "val": [1.0, 2.0, 3.0]})
        >>> df.booster.groupby(["k1", "k2"], "val", "sum")
    """

    _SUPPORTED_AGGS: set[str] = {"sum", "mean", "min", "max", "count"}
    _MAX_MULTI_KEYS: int = 10

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
        by: str | Sequence[str],
        target: str,
        agg: AggFunc,
        *,
        sort: bool = True,
    ) -> Series:
        """Perform accelerated groupby aggregation.

        Args:
            by: Column name(s) to group by. Can be a single string or a list of strings.
                All key columns must be integer dtype for acceleration.
            target: Column name to aggregate (must be numeric).
            agg: Aggregation function - one of "sum", "mean", "min", "max".
            sort: If True (default), sort the result by the group keys to match
                Pandas default behavior. If False, the result order is arbitrary
                but may be faster for large datasets.

        Returns:
            Series indexed by unique group keys with aggregated values.
            For multi-key groupby, returns Series with MultiIndex.

        Raises:
            ValueError: If agg is not a supported aggregation function.
            ValueError: If more than 10 key columns are specified.
        """
        if agg not in self._SUPPORTED_AGGS:
            raise ValueError(f"Unsupported aggregation: {agg}. Use one of {self._SUPPORTED_AGGS}")

        by_cols = [by] if isinstance(by, str) else list(by)

        if len(by_cols) > self._MAX_MULTI_KEYS:
            raise ValueError(
                f"Too many key columns: {len(by_cols)}. Maximum is {self._MAX_MULTI_KEYS}"
            )

        if len(by_cols) == 1:
            return self._groupby_single(by_cols[0], target, agg, sort=sort)
        return self._groupby_multi(by_cols, target, agg, sort=sort)

    def _groupby_single(self, by: str, target: str, agg: str, *, sort: bool) -> Series:
        key_col = self._df[by]
        val_col = self._df[target]

        if len(self._df) < self._get_fallback_threshold():
            return self._pandas_fallback([by], target, agg, sort=sort)

        if not pd.api.types.is_integer_dtype(key_col):
            return self._pandas_fallback([by], target, agg, sort=sort)

        if not self._is_numeric_dtype(val_col):
            return self._pandas_fallback([by], target, agg, sort=sort)

        if self._has_nullable_na(key_col) or self._has_nullable_na(val_col):
            return self._pandas_fallback([by], target, agg, sort=sort)

        return self._rust_groupby_single(key_col, val_col, agg, sort=sort)

    def _groupby_multi(self, by_cols: list[str], target: str, agg: str, *, sort: bool) -> Series:
        val_col = self._df[target]

        if len(self._df) < self._get_fallback_threshold():
            return self._pandas_fallback(by_cols, target, agg, sort=sort)

        for col_name in by_cols:
            col = self._df[col_name]
            if not pd.api.types.is_integer_dtype(col):
                return self._pandas_fallback(by_cols, target, agg, sort=sort)
            if self._has_nullable_na(col):
                return self._pandas_fallback(by_cols, target, agg, sort=sort)

        if not self._is_numeric_dtype(val_col):
            return self._pandas_fallback(by_cols, target, agg, sort=sort)

        if self._has_nullable_na(val_col):
            return self._pandas_fallback(by_cols, target, agg, sort=sort)

        return self._rust_groupby_multi(by_cols, val_col, agg, sort=sort)

    def _is_numeric_dtype(self, series: Series) -> bool:
        return pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series)

    def _has_nullable_na(self, series: Series) -> bool:
        if pd.api.types.is_extension_array_dtype(series):
            return series.isna().any()
        return False

    def _pandas_fallback(self, by_cols: list[str], target: str, agg: str, *, sort: bool) -> Series:
        grouped = self._df.groupby(by_cols, sort=sort)[target]
        return getattr(grouped, agg)()

    def _rust_groupby_single(
        self, key_col: Series, val_col: Series, agg: str, *, sort: bool
    ) -> Series:
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
        if agg == "count":
            result = result.astype(np.int64)

        if sort:
            result = result.sort_index()
        return result

    def _rust_groupby_multi(
        self, by_cols: list[str], val_col: Series, agg: str, *, sort: bool
    ) -> Series:
        rust = self._get_rust_module()

        key_arrays = [
            np.ascontiguousarray(self._df[col].to_numpy(dtype=np.int64)) for col in by_cols
        ]

        is_val_int = pd.api.types.is_integer_dtype(val_col)

        if is_val_int:
            values = np.ascontiguousarray(val_col.to_numpy(dtype=np.int64))
            func_name = f"groupby_multi_{agg}_i64"
        else:
            values = np.ascontiguousarray(val_col.to_numpy(dtype=np.float64))
            func_name = f"groupby_multi_{agg}_f64"

        rust_func = getattr(rust, func_name)
        keys_2d, result_values = rust_func(key_arrays, values)

        if keys_2d.shape[0] == 0:
            idx = pd.MultiIndex.from_arrays([[] for _ in by_cols], names=by_cols)
            return pd.Series([], index=idx, name=val_col.name, dtype=np.float64)

        index_arrays = [keys_2d[:, i] for i in range(keys_2d.shape[1])]
        idx = pd.MultiIndex.from_arrays(index_arrays, names=by_cols)

        result = pd.Series(result_values, index=idx, name=val_col.name)
        if agg == "count":
            result = result.astype(np.int64)

        if sort:
            result = result.sort_index()
        return result

    def thread_count(self) -> int:
        """Return the number of threads used by the Rust parallel runtime."""
        rust = self._get_rust_module()
        return rust.get_thread_count()
