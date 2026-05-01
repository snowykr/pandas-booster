from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import numpy as np
import pandas as pd

from ._abi_compat import (
    PandasBoosterKeyShapeSkewError,
    normalize_multi_keys_cols,
    normalize_result_values,
    raise_abi_skew,
)
from ._config import (
    force_pandas_float_groupby_enabled,
    force_pandas_sort_enabled,
    strict_abi_enabled,
)
from ._groupby_accel import (
    build_series_from_multi_result,
    build_series_from_single_result,
    capture_key_numpy_dtype,
    classify_groupby_compatibility,
    select_rust_groupby_func,
    to_i64_contiguous,
)

if TYPE_CHECKING:
    from pandas import DataFrame, Series


class _SeriesGroupByProto(Protocol):
    def sum(self, *args: Any, **kwargs: Any) -> Series: ...

    def mean(self, *args: Any, **kwargs: Any) -> Series: ...

    def min(self, *args: Any, **kwargs: Any) -> Series: ...

    def max(self, *args: Any, **kwargs: Any) -> Series: ...

    def count(self, *args: Any, **kwargs: Any) -> Series: ...

    def std(self, *args: Any, **kwargs: Any) -> Series: ...

    def var(self, *args: Any, **kwargs: Any) -> Series: ...

    def __getattr__(self, name: str) -> Any: ...


class _DataFrameGroupByProto(Protocol):
    def __getitem__(self, key: str | list[str]) -> Any: ...

    def __iter__(self) -> Iterator[tuple[object, DataFrame]]: ...

    def __len__(self) -> int: ...

    def __getattr__(self, name: str) -> Any: ...


_ACCELERATED_AGGS = frozenset({"sum", "mean", "min", "max", "count", "std", "var"})
_FALLBACK_THRESHOLD: int | None = None


def _get_fallback_threshold() -> int:
    global _FALLBACK_THRESHOLD
    if _FALLBACK_THRESHOLD is None:
        import importlib

        _rust = importlib.import_module("pandas_booster._rust")
        _FALLBACK_THRESHOLD = int(_rust.get_fallback_threshold())
    return _FALLBACK_THRESHOLD


AggFunc = Literal["sum", "mean", "min", "max", "count", "std", "var"]


class BoosterSeriesGroupBy:
    """Proxy for pandas SeriesGroupBy that optionally accelerates aggregations."""

    __slots__ = ("_obj", "_df", "_by_cols", "_target", "_sort")

    def __init__(
        self,
        original_groupby: _SeriesGroupByProto,
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

    def _can_accelerate(self, agg: AggFunc) -> bool:
        if agg not in {"std", "var"} and len(self._df) < _get_fallback_threshold():
            return False

        key_cols: list[pd.Series] = []
        for col_name in self._by_cols:
            col = self._df[col_name]
            if not isinstance(col, pd.Series):
                return False
            key_cols.append(cast(pd.Series, col))

        val_col = self._df[self._target]
        if not isinstance(val_col, pd.Series):
            return False
        val_col = cast(pd.Series, val_col)

        compatibility = classify_groupby_compatibility(
            key_cols=key_cols,
            val_col=val_col,
            agg=agg,
            force_pandas_float_groupby=force_pandas_float_groupby_enabled(),
        )
        return compatibility.supported and not compatibility.force_pandas

    def _rust_aggregate(self, agg: AggFunc) -> Series:
        import importlib

        _rust = importlib.import_module("pandas_booster._rust")

        by_cols = self._by_cols
        target = self._target
        sort = self._sort
        force_pandas_sort = bool(sort) and force_pandas_sort_enabled()

        strict = strict_abi_enabled()

        val_col = self._df[target]
        if not isinstance(val_col, pd.Series):
            return cast("Series", getattr(self._obj, agg)())
        is_val_int = pd.api.types.is_integer_dtype(val_col)

        if len(by_cols) == 1:
            key_col = self._df[by_cols[0]]
            if not isinstance(key_col, pd.Series):
                return cast("Series", getattr(self._obj, agg)())

            if not is_val_int and agg in {"sum", "mean"} and force_pandas_float_groupby_enabled():
                return cast("Series", getattr(self._obj, agg)())

            key_dtype = capture_key_numpy_dtype(key_col)
            value_dtype = np.asarray(val_col.to_numpy(copy=False)).dtype

            keys = to_i64_contiguous(key_col.to_numpy(copy=False))
            if is_val_int:
                values = np.ascontiguousarray(val_col.to_numpy(dtype=np.int64))
                func_base = f"groupby_{agg}_i64"
            else:
                values = np.ascontiguousarray(val_col.to_numpy(dtype=np.float64))
                func_base = f"groupby_{agg}_f64"

            try:
                rust_func, needs_python_sort = select_rust_groupby_func(
                    _rust,
                    func_base,
                    sort=sort,
                    n_rows=len(self._df),
                    force_pandas_sort=force_pandas_sort,
                    context="proxy",
                )
                result_keys, result_values = rust_func(keys, values)
                result_values_arr = normalize_result_values(
                    result_values,
                    agg=agg,
                    is_val_int=is_val_int,
                    context="proxy",
                )

                return build_series_from_single_result(
                    np.asarray(result_keys),
                    result_values_arr,
                    name=val_col.name,
                    index_name=key_col.name,
                    index_dtype=key_dtype,
                    value_dtype=value_dtype,
                    agg=agg,
                    is_val_int=is_val_int,
                    sort=sort,
                    needs_python_sort=needs_python_sort,
                )
            except PandasBoosterKeyShapeSkewError:
                if strict:
                    raise
                return cast("Series", getattr(self._obj, agg)())
        else:
            key_cols: list[pd.Series] = []
            for col_name in by_cols:
                col = self._df[col_name]
                if not isinstance(col, pd.Series):
                    return cast("Series", getattr(self._obj, agg)())
                key_cols.append(col)

            key_dtypes = [capture_key_numpy_dtype(col) for col in key_cols]
            value_dtype = np.asarray(val_col.to_numpy(copy=False)).dtype
            key_arrays = [to_i64_contiguous(col.to_numpy(copy=False)) for col in key_cols]

            if is_val_int:
                values = np.ascontiguousarray(val_col.to_numpy(dtype=np.int64))
                func_base = f"groupby_multi_{agg}_i64"
            else:
                values = np.ascontiguousarray(val_col.to_numpy(dtype=np.float64))
                func_base = f"groupby_multi_{agg}_f64"

            try:
                rust_func, needs_python_sort = select_rust_groupby_func(
                    _rust,
                    func_base,
                    sort=sort,
                    n_rows=len(self._df),
                    force_pandas_sort=force_pandas_sort,
                    context="proxy",
                )
                rust_result = rust_func(key_arrays, values)
                if not isinstance(rust_result, tuple) or len(rust_result) != 2:
                    raise_abi_skew(
                        context="proxy",
                        detail=(
                            "expected Rust groupby_multi result as "
                            "(keys_cols, result_values) tuple; "
                            f"got type={type(rust_result)!r}."
                            if not isinstance(rust_result, tuple)
                            else (
                                "expected 2-tuple (keys_cols, result_values); "
                                f"got len={len(rust_result)}."
                            )
                        ),
                    )
                keys_cols, result_values = rust_result
                result_values_arr = normalize_result_values(
                    result_values,
                    agg=agg,
                    is_val_int=is_val_int,
                    context="proxy",
                )
                keys_cols_arr = normalize_multi_keys_cols(
                    keys_cols,
                    n_groups=result_values_arr.shape[0],
                    n_keys=len(by_cols),
                    context="proxy",
                    strict=strict,
                )
                return build_series_from_multi_result(
                    keys_cols_arr,
                    result_values_arr,
                    by_cols=by_cols,
                    key_dtypes=key_dtypes,
                    name=val_col.name,
                    value_dtype=value_dtype,
                    agg=agg,
                    is_val_int=is_val_int,
                    sort=sort,
                    needs_python_sort=needs_python_sort,
                )
            except PandasBoosterKeyShapeSkewError:
                if strict:
                    raise
                return cast("Series", getattr(self._obj, agg)())

    def _try_accelerate(self, agg: AggFunc) -> Series:
        if agg in _ACCELERATED_AGGS and self._can_accelerate(agg):
            return self._rust_aggregate(agg)
        return cast("Series", getattr(self._obj, agg)())

    def sum(self, *args: Any, **kwargs: Any) -> Series:
        """Compute sum, using Rust acceleration when possible."""
        if args or kwargs:
            return cast("Series", self._obj.sum(*args, **kwargs))
        return self._try_accelerate("sum")

    def mean(self, *args: Any, **kwargs: Any) -> Series:
        """Compute mean, using Rust acceleration when possible."""
        if args or kwargs:
            return cast("Series", self._obj.mean(*args, **kwargs))
        return self._try_accelerate("mean")

    def min(self, *args: Any, **kwargs: Any) -> Series:
        """Compute min, using Rust acceleration when possible."""
        if args or kwargs:
            return cast("Series", self._obj.min(*args, **kwargs))
        return self._try_accelerate("min")

    def max(self, *args: Any, **kwargs: Any) -> Series:
        """Compute max, using Rust acceleration when possible."""
        if args or kwargs:
            return cast("Series", self._obj.max(*args, **kwargs))
        return self._try_accelerate("max")

    def count(self) -> Series:
        """Compute count, using Rust acceleration when possible."""
        return self._try_accelerate("count")

    def std(self, *args: Any, **kwargs: Any) -> Series:
        """Compute std, using Rust acceleration when possible."""
        if args or kwargs:
            return cast("Series", self._obj.std(*args, **kwargs))
        return self._try_accelerate("std")

    def var(self, *args: Any, **kwargs: Any) -> Series:
        """Compute var, using Rust acceleration when possible."""
        if args or kwargs:
            return cast("Series", self._obj.var(*args, **kwargs))
        return self._try_accelerate("var")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._obj, name)

    def __repr__(self) -> str:
        return repr(self._obj)

    def __dir__(self) -> list[str]:
        return dir(self._obj)


class BoosterDataFrameGroupBy:
    """Proxy for pandas DataFrameGroupBy that wraps Series selections."""

    __slots__ = ("_obj", "_df", "_by_cols", "_sort")

    def __init__(
        self,
        original_groupby: _DataFrameGroupByProto,
        df: DataFrame,
        by_cols: list[str],
        sort: bool,
    ) -> None:
        self._obj = original_groupby
        self._df = df
        self._by_cols = by_cols
        self._sort = sort

    def __getitem__(self, key: str | list[str]) -> BoosterSeriesGroupBy | Any:
        selected = self._obj[key]

        if isinstance(key, str):
            return BoosterSeriesGroupBy(
                original_groupby=cast(_SeriesGroupByProto, selected),
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

    def __iter__(self) -> Iterator[tuple[object, DataFrame]]:
        return iter(self._obj)

    def __len__(self) -> int:
        return len(self._obj)
