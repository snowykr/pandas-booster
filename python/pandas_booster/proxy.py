from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import pandas as pd

from ._groupby_execution import (
    execute_groupby_multi,
    execute_groupby_single,
)

if TYPE_CHECKING:
    from pandas import DataFrame, Series


class _SeriesGroupByProto(Protocol):
    def sum(self, *args: Any, **kwargs: Any) -> Series: ...

    def mean(self, *args: Any, **kwargs: Any) -> Series: ...

    def prod(self, *args: Any, **kwargs: Any) -> Series: ...

    def min(self, *args: Any, **kwargs: Any) -> Series: ...

    def max(self, *args: Any, **kwargs: Any) -> Series: ...

    def count(self, *args: Any, **kwargs: Any) -> Series: ...

    def std(self, *args: Any, **kwargs: Any) -> Series: ...

    def median(self, *args: Any, **kwargs: Any) -> Series: ...

    def var(self, *args: Any, **kwargs: Any) -> Series: ...

    def __getattr__(self, name: str) -> Any: ...


class _DataFrameGroupByProto(Protocol):
    def __getitem__(self, key: str | list[str]) -> Any: ...

    def __iter__(self) -> Iterator[tuple[object, DataFrame]]: ...

    def __len__(self) -> int: ...

    def __getattr__(self, name: str) -> Any: ...


_ACCELERATED_AGGS = frozenset(
    {"sum", "mean", "prod", "min", "max", "count", "std", "var", "median"}
)


AggFunc = Literal["sum", "mean", "prod", "min", "max", "count", "std", "median", "var"]


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

    def _accelerated_aggregate(self, agg: AggFunc) -> Series:
        by_cols = self._by_cols
        target = self._target
        sort = self._sort

        def fallback() -> Series:
            return cast("Series", getattr(self._obj, agg)())

        val_col = self._df[target]
        if not isinstance(val_col, pd.Series):
            return fallback()

        if len(by_cols) == 1:
            key_col = self._df[by_cols[0]]
            if not isinstance(key_col, pd.Series):
                return fallback()
            return execute_groupby_single(
                self._df,
                key_col,
                val_col,
                agg,
                sort=sort,
                context="proxy",
                fallback=fallback,
            )

        key_cols: list[tuple[str, pd.Series]] = []
        for col_name in by_cols:
            col = self._df[col_name]
            if not isinstance(col, pd.Series):
                return fallback()
            key_cols.append((col_name, col))

        return execute_groupby_multi(
            self._df,
            key_cols,
            val_col,
            agg,
            sort=sort,
            context="proxy",
            fallback=fallback,
        )

    def _try_accelerate(self, agg: AggFunc) -> Series:
        if agg in _ACCELERATED_AGGS:
            return self._accelerated_aggregate(agg)
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

    def prod(self, *args: Any, **kwargs: Any) -> Series:
        """Compute product, using Rust acceleration when possible."""
        if args or kwargs:
            return cast("Series", self._obj.prod(*args, **kwargs))
        return self._try_accelerate("prod")

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

    def median(self, *args: Any, **kwargs: Any) -> Series:
        """Compute median, preserving pandas call parity."""
        if args or kwargs:
            return cast("Series", self._obj.median(*args, **kwargs))
        return self._try_accelerate("median")

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
