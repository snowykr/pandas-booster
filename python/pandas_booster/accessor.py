from __future__ import annotations

# NOTE: Keep this module lightweight; it is on the hot path for groupby acceleration.
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import pandas as pd

from . import _groupby_accel as _groupby_accel_mod
from . import _groupby_execution as _groupby_execution_mod

if TYPE_CHECKING:
    from pandas import DataFrame, Series

AggFunc = Literal["sum", "mean", "prod", "min", "max", "count", "std", "var", "median"]


@pd.api.extensions.register_dataframe_accessor("booster")
class BoosterAccessor:
    """Pandas DataFrame accessor providing Rust-accelerated groupby operations.

    Automatically falls back to native Pandas when:
    - Dataset has fewer than 100,000 rows (legacy aggs only: sum, mean, prod, min, max, count)
    - Key column is not integer dtype
    - Value column is not numeric (int/float)
    - Columns are nullable extension dtypes (e.g. pandas Int64) or contain pd.NA

    Note: std and var use pandas-default semantics (ddof=1) and always return float64.
    Median is certified only for primitive NumPy-backed integer/float values with
    integer keys, and otherwise routes through the pandas fallback path.
    Unlike legacy aggs, supported std, var, and median are Rust-first by default regardless of
    dataset size. This provides significant speedups (up to 1.5x) for high-cardinality
    workloads, though standard-cardinality single-key cases may remain slower than
    pandas. Single-key float-input `prod` uses an ordered Rust path when the
    extension exposes the matching ABI marker, preserving row-order IEEE-754
    overflow/underflow semantics. The
    `PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY` environment toggle can be used
    to force a pandas fallback specifically for single-key float-input
    sum/mean/prod/std/var/median operations; it does not broaden to
    multi-key or int-backed paths.

    Examples:
        Single key:
        >>> df = pd.DataFrame({"key": [1, 2, 1], "val": [10.0, 20.0, 30.0]})
        >>> df.booster.groupby("key", "val", "sum")
        >>> df.booster.groupby("key", "val", "prod")
        >>> df.booster.groupby("key", "val", "std")

        Multiple keys:
        >>> df = pd.DataFrame({"k1": [1, 1, 2], "k2": [10, 20, 10], "val": [1.0, 2.0, 3.0]})
        >>> df.booster.groupby(["k1", "k2"], "val", "sum")
    """

    _SUPPORTED_AGGS: set[str] = {
        "sum",
        "mean",
        "prod",
        "min",
        "max",
        "count",
        "std",
        "var",
        "median",
    }
    _MAX_MULTI_KEYS: int = 10

    def __init__(self, pandas_obj: DataFrame) -> None:
        self._df: DataFrame = pandas_obj

    @staticmethod
    def _get_rust_module() -> object:
        return _groupby_accel_mod.get_rust_module()

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
            agg: Aggregation function - one of "sum", "mean", "prod", "min",
                "max", "count", "std", "var", "median".
            sort: If True (default), sort the result by the group keys to match
                Pandas default behavior. If False, preserve Pandas' "first-seen"
                group order (order of appearance in the input).

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

    def _groupby_single(self, by: str, target: str, agg: AggFunc, *, sort: bool) -> Series:
        key_col = self._df[by]
        val_col = self._df[target]

        if not isinstance(key_col, pd.Series) or not isinstance(val_col, pd.Series):
            return self._pandas_fallback([by], target, agg, sort=sort)

        return _groupby_execution_mod.execute_groupby_single(
            self._df,
            key_col,
            val_col,
            agg,
            sort=sort,
            context="accessor",
            fallback=lambda: self._pandas_fallback([by], target, agg, sort=sort),
        )

    def _groupby_multi(
        self, by_cols: list[str], target: str, agg: AggFunc, *, sort: bool
    ) -> Series:
        val_col = self._df[target]

        if not isinstance(val_col, pd.Series):
            return self._pandas_fallback(by_cols, target, agg, sort=sort)

        key_cols: dict[str, pd.Series] = {}
        for col_name in by_cols:
            col = self._df[col_name]
            if not isinstance(col, pd.Series):
                return self._pandas_fallback(by_cols, target, agg, sort=sort)
            key_cols[col_name] = col

        return _groupby_execution_mod.execute_groupby_multi(
            self._df,
            [(col_name, key_cols[col_name]) for col_name in by_cols],
            val_col,
            agg,
            sort=sort,
            context="accessor",
            fallback=lambda: self._pandas_fallback(by_cols, target, agg, sort=sort),
        )

    def _pandas_fallback(
        self, by_cols: list[str], target: str, agg: AggFunc, *, sort: bool
    ) -> Series:
        grouped = self._df.groupby(by_cols, sort=sort)[target]
        return getattr(grouped, agg)()

    def thread_count(self) -> int:
        """Return the number of threads used by the Rust parallel runtime."""
        return _groupby_accel_mod.get_rust_module().get_thread_count()
