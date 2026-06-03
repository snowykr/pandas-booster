"""Benchmark worker execution."""

from __future__ import annotations

import time
from typing import Any, Callable, Literal, cast

import pandas as pd
from datasets import PRESETS, generate_multi_key_dataset
from dispatch import HAS_POLARS, build_polars_agg_expr, describe_booster_execution, pl
from reporting import BACKEND_DISPLAY_ORDER


def benchmark_worker(
    preset_name: str,
    backend: str,
    agg: Literal[
        "sum",
        "mean",
        "prod",
        "median",
        "std",
        "var",
        "min",
        "max",
        "count",
    ] = "sum",
    sort: bool = True,
    verify_correctness: bool = False,
    mode: Literal["cold", "warm"] = "cold",
) -> dict[str, Any]:
    if backend not in BACKEND_DISPLAY_ORDER:
        raise ValueError(f"Unsupported benchmark backend: {backend!r}")

    run_once: Callable[[], Any]
    execution = ""

    config = PRESETS[preset_name]
    df = generate_multi_key_dataset(**config)
    key_cols = [col for col, _ in config["key_configs"]]
    value_col = "value"

    if verify_correctness:
        for col in key_cols:
            if not pd.api.types.is_integer_dtype(df[col]):
                raise ValueError(f"Benchmark key column {col!r} must be integer dtype")
            if df[col].isna().any():
                raise ValueError(
                    f"Benchmark key column {col!r} contains nulls; semantics differ across engines"
                )

        if df[value_col].isna().any():
            raise ValueError(
                f"Benchmark value column {value_col!r} contains NaNs; "
                "semantics can diverge across engines"
            )

    if backend == "pandas":
        execution = f"pandas.groupby.{agg}"

        def run_once_pandas() -> Any:
            return getattr(df.groupby(key_cols, sort=sort)[value_col], agg)()

        run_once = run_once_pandas
    elif backend == "booster":
        from pandas_booster.accessor import BoosterAccessor

        execution = describe_booster_execution(df, key_cols, value_col, agg, sort)

        def run_once_booster() -> Any:
            return cast(BoosterAccessor, df.booster).groupby(key_cols, value_col, agg, sort=sort)

        run_once = run_once_booster
    elif backend == "polars":
        if not HAS_POLARS:
            raise ImportError("Polars is not installed")

        assert pl is not None
        execution = f"polars.group_by.agg({agg})"

        df_polars = pl.DataFrame(
            {**{col: df[col].values for col in key_cols}, value_col: df[value_col].values}
        )
        agg_expr = build_polars_agg_expr(value_col, agg)

        def run_once_polars() -> Any:
            result = df_polars.group_by(key_cols, maintain_order=not sort).agg(agg_expr)
            if sort:
                result = result.sort(key_cols)
            return result

        run_once = run_once_polars
    else:
        raise ValueError(f"Unsupported benchmark backend: {backend!r}")

    def pandas_baseline() -> pd.Series:
        return cast(pd.Series, getattr(df.groupby(key_cols, sort=sort)[value_col], agg)())

    def polars_to_pandas_series(result_df: Any) -> pd.Series:
        if not HAS_POLARS:
            raise ImportError("Polars is not installed")
        assert pl is not None

        if not isinstance(result_df, pl.DataFrame):
            raise TypeError(f"Expected polars DataFrame, got {type(result_df)}")

        try:
            pdf = result_df.to_pandas()
        except Exception:
            pdf = pd.DataFrame({col: result_df[col].to_numpy() for col in result_df.columns})
        if len(key_cols) == 1:
            s = pdf.set_index(key_cols[0])[value_col]
            s.index.name = key_cols[0]
        else:
            s = pdf.set_index(key_cols)[value_col]
            s.index.names = key_cols
        s.name = value_col
        return cast(pd.Series, s)

    def normalize_result_to_series(result_obj: Any, backend_name: str = backend) -> pd.Series:
        if backend_name in ("pandas", "booster"):
            if not isinstance(result_obj, pd.Series):
                raise TypeError(f"Expected pandas Series, got {type(result_obj)}")
            s = cast(pd.Series, result_obj)
            if s.name != value_col:
                s = s.copy()
                s.name = value_col
            if len(key_cols) == 1:
                if s.index.name != key_cols[0]:
                    s.index.name = key_cols[0]
            elif list(s.index.names) != key_cols:
                s.index.names = key_cols
            return s
        if backend_name == "polars":
            return polars_to_pandas_series(result_obj)

        raise ValueError(f"Unsupported benchmark backend: {backend_name!r}")

    def assert_matches_baseline(actual: pd.Series, expected: pd.Series) -> None:
        if agg == "count":
            pd.testing.assert_series_equal(
                actual,
                expected,
                check_exact=True,
                check_dtype=False,
            )
            return

        if pd.api.types.is_float_dtype(expected):
            pd.testing.assert_series_equal(
                actual,
                expected,
                check_exact=False,
                rtol=1e-9,
                atol=1e-12,
                check_dtype=False,
            )
            return

        pd.testing.assert_series_equal(
            actual,
            expected,
            check_exact=True,
            check_dtype=False,
        )

    if mode == "cold":
        start = time.perf_counter()
        cold_result = run_once()
        cold_time = time.perf_counter() - start

        correctness = "not_checked"
        if verify_correctness and backend in {"booster", "polars"}:
            try:
                expected = pandas_baseline()
                actual = normalize_result_to_series(cold_result)
                assert_matches_baseline(actual, expected)
                correctness = "pass"
            except AssertionError as e:
                correctness = f"fail: {str(e)[:100]}"
            except Exception as e:
                correctness = f"fail: {type(e).__name__}: {str(e)[:80]}"

        return {
            "cold_time_s": cold_time,
            "correctness": correctness,
            "execution": execution,
        }

    if mode == "warm":
        _ = run_once()
        _ = run_once()

        start = time.perf_counter()
        warm_result = run_once()
        warm_time = time.perf_counter() - start

        correctness = "not_checked"
        if verify_correctness and backend in {"booster", "polars"}:
            try:
                expected = pandas_baseline()
                actual = normalize_result_to_series(warm_result)
                assert_matches_baseline(actual, expected)
                correctness = "pass"
            except AssertionError as e:
                correctness = f"fail: {str(e)[:100]}"
            except Exception as e:
                correctness = f"fail: {type(e).__name__}: {str(e)[:80]}"

        return {
            "warm_time_s": warm_time,
            "correctness": correctness,
            "execution": execution,
        }

    raise ValueError(f"Unsupported benchmark mode: {mode!r}")
