from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pandas import DataFrame, Series
    from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy

_ACCELERATED_AGGS = frozenset({"sum", "mean", "min", "max", "count"})
_FALLBACK_THRESHOLD = 100_000


class BoosterSeriesGroupBy:
    __slots__ = ("_obj", "_df", "_by_cols", "_target", "_sort")

    def __init__(
        self,
        original_groupby: SeriesGroupBy,
        df: DataFrame,
        by_cols: list[str],
        target: str,
        sort: bool,
    ) -> None:
        self._obj = original_groupby
        self._df = df
        self._by_cols = by_cols
        self._target = target
        self._sort = sort

    def _can_accelerate(self) -> bool:
        if len(self._df) < _FALLBACK_THRESHOLD:
            return False

        for col_name in self._by_cols:
            col = self._df[col_name]
            if not pd.api.types.is_integer_dtype(col):
                return False
            if pd.api.types.is_extension_array_dtype(col) and col.isna().any():
                return False

        val_col = self._df[self._target]
        if not (pd.api.types.is_integer_dtype(val_col) or pd.api.types.is_float_dtype(val_col)):
            return False
        if pd.api.types.is_extension_array_dtype(val_col) and val_col.isna().any():
            return False

        return True

    def _rust_aggregate(self, agg: str) -> Series:
        from pandas_booster import _rust

        by_cols = self._by_cols
        target = self._target
        sort = self._sort

        val_col = self._df[target]
        is_val_int = pd.api.types.is_integer_dtype(val_col)

        if len(by_cols) == 1:
            keys = np.ascontiguousarray(self._df[by_cols[0]].to_numpy(dtype=np.int64))
            if is_val_int:
                values = np.ascontiguousarray(val_col.to_numpy(dtype=np.int64))
                func_name = f"groupby_{agg}_i64"
            else:
                values = np.ascontiguousarray(val_col.to_numpy(dtype=np.float64))
                func_name = f"groupby_{agg}_f64"

            rust_func = getattr(_rust, func_name)
            result_dict = rust_func(keys, values)

            result = pd.Series(result_dict, name=val_col.name)
            result.index.name = by_cols[0]
            if sort:
                result = result.sort_index()
            return result
        else:
            key_arrays = [
                np.ascontiguousarray(self._df[col].to_numpy(dtype=np.int64)) for col in by_cols
            ]

            if is_val_int:
                values = np.ascontiguousarray(val_col.to_numpy(dtype=np.int64))
                func_name = f"groupby_multi_{agg}_i64"
            else:
                values = np.ascontiguousarray(val_col.to_numpy(dtype=np.float64))
                func_name = f"groupby_multi_{agg}_f64"

            rust_func = getattr(_rust, func_name)
            keys_2d, result_values = rust_func(key_arrays, values)

            if keys_2d.shape[0] == 0:
                idx = pd.MultiIndex.from_arrays([[] for _ in by_cols], names=by_cols)
                return pd.Series([], index=idx, name=val_col.name, dtype=np.float64)

            index_arrays = [keys_2d[:, i] for i in range(keys_2d.shape[1])]
            idx = pd.MultiIndex.from_arrays(index_arrays, names=by_cols)

            result = pd.Series(result_values, index=idx, name=val_col.name)
            if sort:
                result = result.sort_index()
            return result

    def _try_accelerate(self, agg: str) -> Series:
        if agg in _ACCELERATED_AGGS and self._can_accelerate():
            result = self._rust_aggregate(agg)
            # count returns int64 in Pandas, but Rust returns float64
            if agg == "count":
                result = result.astype(np.int64)
            return result
        return getattr(self._obj, agg)()

    def sum(self, *args: Any, **kwargs: Any) -> Series:
        if args or kwargs:
            return self._obj.sum(*args, **kwargs)
        return self._try_accelerate("sum")

    def mean(self, *args: Any, **kwargs: Any) -> Series:
        if args or kwargs:
            return self._obj.mean(*args, **kwargs)
        return self._try_accelerate("mean")

    def min(self, *args: Any, **kwargs: Any) -> Series:
        if args or kwargs:
            return self._obj.min(*args, **kwargs)
        return self._try_accelerate("min")

    def max(self, *args: Any, **kwargs: Any) -> Series:
        if args or kwargs:
            return self._obj.max(*args, **kwargs)
        return self._try_accelerate("max")

    def count(self) -> Series:
        return self._try_accelerate("count")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._obj, name)

    def __repr__(self) -> str:
        return repr(self._obj)

    def __dir__(self) -> list[str]:
        return dir(self._obj)


class BoosterDataFrameGroupBy:
    __slots__ = ("_obj", "_df", "_by_cols", "_sort")

    def __init__(
        self,
        original_groupby: DataFrameGroupBy,
        df: DataFrame,
        by_cols: list[str],
        sort: bool,
    ) -> None:
        self._obj = original_groupby
        self._df = df
        self._by_cols = by_cols
        self._sort = sort

    def __getitem__(self, key: str | list[str]) -> BoosterSeriesGroupBy | DataFrameGroupBy:
        selected = self._obj[key]

        if isinstance(key, str):
            return BoosterSeriesGroupBy(
                original_groupby=selected,
                df=self._df,
                by_cols=self._by_cols,
                target=key,
                sort=self._sort,
            )

        return selected

    def __getattr__(self, name: str) -> Any:
        return getattr(self._obj, name)

    def __repr__(self) -> str:
        return repr(self._obj)

    def __dir__(self) -> list[str]:
        return dir(self._obj)

    def __iter__(self):
        return iter(self._obj)

    def __len__(self) -> int:
        return len(self._obj)
