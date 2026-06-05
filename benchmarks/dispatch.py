"""Benchmark backend dispatch helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    pl = None  # type: ignore[assignment]
    HAS_POLARS = False

BENCHMARKS_DIR = Path(__file__).resolve().parent
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))


def build_polars_agg_expr(value_col: str, agg: str) -> Any:
    assert pl is not None

    agg_map = {
        "sum": pl.col(value_col).sum().alias(value_col),
        "mean": pl.col(value_col).mean().alias(value_col),
        "median": pl.col(value_col).median().alias(value_col),
        "prod": pl.col(value_col).product().alias(value_col),
        "std": pl.col(value_col).std().alias(value_col),
        "var": pl.col(value_col).var().alias(value_col),
        "min": pl.col(value_col).min().alias(value_col),
        "max": pl.col(value_col).max().alias(value_col),
        "count": pl.col(value_col).count().alias(value_col),
    }
    return agg_map[agg]


def describe_booster_execution(
    df: pd.DataFrame,
    key_cols: list[str],
    value_col: str,
    agg: str,
    sort: bool,
    *,
    ignore_force_pandas_sort: bool = False,
) -> str:
    return cast(
        str,
        resolve_booster_benchmark_dispatch(
            df,
            key_cols,
            value_col,
            agg,
            sort,
            ignore_force_pandas_sort=ignore_force_pandas_sort,
        )["execution"],
    )


def resolve_booster_benchmark_dispatch(
    df: pd.DataFrame,
    key_cols: list[str],
    value_col: str,
    agg: str,
    sort: bool,
    *,
    ignore_force_pandas_sort: bool = False,
) -> dict[str, Any]:
    import pandas_booster._rust as rust
    from pandas_booster import _groupby_accel as groupby_accel
    from pandas_booster._config import (
        force_pandas_float_groupby_enabled,
        force_pandas_sort_enabled,
    )

    val_col = cast(pd.Series, df[value_col])
    key_series = [cast(pd.Series, df[col]) for col in key_cols]

    if agg not in {"std", "var", "median"} and len(df) < rust.get_fallback_threshold():
        return {
            "execution": f"booster->pandas.groupby.{agg}",
            "rust_func": None,
            "needs_python_sort": False,
        }

    compatibility = groupby_accel.classify_groupby_compatibility(
        key_cols=key_series,
        val_col=val_col,
        agg=cast(Any, agg),
        force_pandas_float_groupby=force_pandas_float_groupby_enabled(),
    )
    is_val_int = pd.api.types.is_integer_dtype(val_col)
    prefix = "groupby_multi" if len(key_cols) > 1 else "groupby"
    suffix = "i64" if is_val_int else "f64"
    force_pandas_sort = (
        False if ignore_force_pandas_sort else bool(sort) and force_pandas_sort_enabled()
    )

    if not compatibility.supported or compatibility.force_pandas:
        return {
            "execution": f"booster->pandas.groupby.{agg}",
            "rust_func": None,
            "needs_python_sort": False,
        }
    if agg in {"median", "prod"} and not groupby_accel.has_rust_groupby_func(
        rust,
        f"{prefix}_{agg}_{suffix}",
        sort=sort,
        n_rows=len(df),
        force_pandas_sort=force_pandas_sort,
    ):
        return {
            "execution": f"booster->pandas.groupby.{agg}",
            "rust_func": None,
            "needs_python_sort": False,
        }

    if (
        len(key_cols) == 1
        and pd.api.types.is_float_dtype(val_col)
        and agg in {"sum", "mean"}
        and force_pandas_float_groupby_enabled()
    ):
        return {
            "execution": f"booster->pandas.groupby.{agg}",
            "rust_func": None,
            "needs_python_sort": False,
        }

    rust_func, needs_python_sort = groupby_accel.select_rust_groupby_func(
        rust,
        f"{prefix}_{agg}_{suffix}",
        sort=sort,
        n_rows=len(df),
        force_pandas_sort=force_pandas_sort,
        context="benchmark",
    )
    execution = f"booster->rust.{rust_func.__name__}"
    if needs_python_sort and sort:
        execution += "+python_sort"
    return {
        "execution": execution,
        "rust_func": rust_func,
        "needs_python_sort": needs_python_sort,
    }
