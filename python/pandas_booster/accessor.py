from __future__ import annotations

# NOTE: Keep this module lightweight; it is on the hot path for groupby acceleration.
from collections.abc import Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd

from . import _abi_compat as _abi_compat
from . import _groupby_accel as _groupby_accel_mod
from ._config import (
    force_pandas_float_groupby_enabled,
    force_pandas_sort_enabled,
    strict_abi_enabled,
)

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
    pandas. The `PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY` environment toggle
    can be used to force a pandas fallback specifically for single-key float-input
    sum/mean/prod/std/var/median operations; it does not broaden to multi-key or int-backed paths.

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
        self._rust: ModuleType | None = None
        self._fallback_threshold: int | None = None

    def _get_rust_module(self) -> ModuleType:
        if self._rust is None:
            import importlib

            self._rust = importlib.import_module("pandas_booster._rust")
        return self._rust

    def _get_fallback_threshold(self) -> int:
        if self._fallback_threshold is None:
            self._fallback_threshold = self._get_rust_module().get_fallback_threshold()
        assert self._fallback_threshold is not None
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

        key_col = cast(pd.Series, key_col)
        val_col = cast(pd.Series, val_col)

        if agg not in {"std", "var", "median"} and len(self._df) < self._get_fallback_threshold():
            return self._pandas_fallback([by], target, agg, sort=sort)

        compatibility = _groupby_accel_mod.classify_groupby_compatibility(
            key_cols=[key_col],
            val_col=val_col,
            agg=agg,
            force_pandas_float_groupby=force_pandas_float_groupby_enabled(),
        )
        if not compatibility.supported or compatibility.force_pandas:
            return self._pandas_fallback([by], target, agg, sort=sort)

        if agg == "median" and not self._has_median_kernel(val_col, multi=False, sort=sort):
            return self._pandas_fallback([by], target, agg, sort=sort)

        return self._rust_groupby_single(by, target, key_col, val_col, agg, sort=sort)

    def _groupby_multi(
        self, by_cols: list[str], target: str, agg: AggFunc, *, sort: bool
    ) -> Series:
        val_col = self._df[target]

        if not isinstance(val_col, pd.Series):
            return self._pandas_fallback(by_cols, target, agg, sort=sort)

        val_col = cast(pd.Series, val_col)

        key_cols: dict[str, pd.Series] = {}
        for col_name in by_cols:
            col = self._df[col_name]
            if not isinstance(col, pd.Series):
                return self._pandas_fallback(by_cols, target, agg, sort=sort)
            key_cols[col_name] = cast(pd.Series, col)

        if agg not in {"std", "var", "median"} and len(self._df) < self._get_fallback_threshold():
            return self._pandas_fallback(by_cols, target, agg, sort=sort)

        compatibility = _groupby_accel_mod.classify_groupby_compatibility(
            key_cols=[key_cols[col_name] for col_name in by_cols],
            val_col=val_col,
            agg=agg,
            force_pandas_float_groupby=force_pandas_float_groupby_enabled(),
        )
        if not compatibility.supported or compatibility.force_pandas:
            return self._pandas_fallback(by_cols, target, agg, sort=sort)

        if agg == "median" and not self._has_median_kernel(val_col, multi=True, sort=sort):
            return self._pandas_fallback(by_cols, target, agg, sort=sort)

        return self._rust_groupby_multi(by_cols, target, key_cols, val_col, agg, sort=sort)

    def _has_median_kernel(self, val_col: Series, *, multi: bool, sort: bool) -> bool:
        rust = self._get_rust_module()
        force_pandas_sort = bool(sort) and force_pandas_sort_enabled()
        kernel = "i64" if pd.api.types.is_integer_dtype(val_col) else "f64"
        prefix = "groupby_multi" if multi else "groupby"
        return _groupby_accel_mod.has_rust_groupby_func(
            rust,
            f"{prefix}_median_{kernel}",
            sort=sort,
            n_rows=len(self._df),
            force_pandas_sort=force_pandas_sort,
        )

    def _pandas_fallback(
        self, by_cols: list[str], target: str, agg: AggFunc, *, sort: bool
    ) -> Series:
        grouped = self._df.groupby(by_cols, sort=sort)[target]
        return getattr(grouped, agg)()

    def _rust_groupby_single(
        self, by: str, target: str, key_col: Series, val_col: Series, agg: AggFunc, *, sort: bool
    ) -> Series:
        rust = self._get_rust_module()
        strict = strict_abi_enabled()

        force_pandas_sort = bool(sort) and force_pandas_sort_enabled()

        key_dtype = _groupby_accel_mod.capture_key_numpy_dtype(key_col)
        value_dtype = _groupby_accel_mod.capture_value_numpy_dtype(val_col)

        keys = _groupby_accel_mod.to_i64_contiguous(key_col.to_numpy(copy=False))
        is_val_int = pd.api.types.is_integer_dtype(val_col)

        if is_val_int:
            values = np.ascontiguousarray(val_col.to_numpy(dtype=np.int64))
            func_base = f"groupby_{agg}_i64"
        else:
            values = np.ascontiguousarray(val_col.to_numpy(dtype=np.float64))
            func_base = f"groupby_{agg}_f64"

        try:
            rust_func, needs_python_sort = _groupby_accel_mod.select_rust_groupby_func(
                rust,
                func_base,
                sort=sort,
                n_rows=len(self._df),
                force_pandas_sort=force_pandas_sort,
                context="accessor",
            )
            result_keys, result_values = rust_func(keys, values)
            result_values_arr = _abi_compat.normalize_result_values(
                result_values,
                agg=agg,
                is_val_int=is_val_int,
                context="accessor",
            )

            return _groupby_accel_mod.build_series_from_single_result(
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
        except _abi_compat.PandasBoosterKeyShapeSkewError:
            if strict:
                raise
            return self._pandas_fallback([by], target, agg, sort=sort)

    def _rust_groupby_multi(
        self,
        by_cols: list[str],
        target: str,
        key_cols: dict[str, Series],
        val_col: Series,
        agg: AggFunc,
        *,
        sort: bool,
    ) -> Series:
        rust = self._get_rust_module()

        strict = strict_abi_enabled()

        force_pandas_sort = bool(sort) and force_pandas_sort_enabled()

        key_dtypes = [_groupby_accel_mod.capture_key_numpy_dtype(key_cols[col]) for col in by_cols]
        value_dtype = _groupby_accel_mod.capture_value_numpy_dtype(val_col)

        key_arrays = [
            _groupby_accel_mod.to_i64_contiguous(key_cols[col].to_numpy(copy=False))
            for col in by_cols
        ]

        is_val_int = pd.api.types.is_integer_dtype(val_col)

        if is_val_int:
            values = np.ascontiguousarray(val_col.to_numpy(dtype=np.int64))
            func_base = f"groupby_multi_{agg}_i64"
        else:
            values = np.ascontiguousarray(val_col.to_numpy(dtype=np.float64))
            func_base = f"groupby_multi_{agg}_f64"

        try:
            rust_func, needs_python_sort = _groupby_accel_mod.select_rust_groupby_func(
                rust,
                func_base,
                sort=sort,
                n_rows=len(self._df),
                force_pandas_sort=force_pandas_sort,
                context="accessor",
            )
            rust_result = rust_func(key_arrays, values)
            if not isinstance(rust_result, tuple) or len(rust_result) != 2:
                _abi_compat.raise_abi_skew(
                    context="accessor",
                    detail=(
                        "expected Rust groupby_multi result as (keys_cols, result_values) tuple; "
                        f"got type={type(rust_result)!r}."
                        if not isinstance(rust_result, tuple)
                        else (
                            "expected 2-tuple (keys_cols, result_values); "
                            f"got len={len(rust_result)}."
                        )
                    ),
                )
            keys_cols, result_values = rust_result
            result_values_arr = _abi_compat.normalize_result_values(
                result_values,
                agg=agg,
                is_val_int=is_val_int,
                context="accessor",
            )
            keys_cols_arr = _abi_compat.normalize_multi_keys_cols(
                keys_cols,
                n_groups=result_values_arr.shape[0],
                n_keys=len(by_cols),
                context="accessor",
                strict=strict,
            )
            return _groupby_accel_mod.build_series_from_multi_result(
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
        except _abi_compat.PandasBoosterKeyShapeSkewError:
            if strict:
                raise
            return self._pandas_fallback(by_cols, target, agg, sort=sort)

    def thread_count(self) -> int:
        """Return the number of threads used by the Rust parallel runtime."""
        rust = self._get_rust_module()
        return rust.get_thread_count()
