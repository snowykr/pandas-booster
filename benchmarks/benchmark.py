"""Official benchmark script for pandas-booster with Cold/Warm measurement.

This script generates per-aggregation benchmark reports for the benchmark docs.
It measures single-key through five-key groupby operations with proper
cold/warm separation and statistical analysis.

Cold/Warm Definitions:
- Cold: First call in a fresh Python process (after import + data preparation)
- Warm: Steady-state performance after cold + 1 warmup run (discarded), 5 samples

Usage:
    # Run core benchmarks (standard + high cardinality, sorted + sort=False)
    python benchmarks/benchmark.py

    # Run only standard cardinality benchmarks
    python benchmarks/benchmark.py --cardinality standard

    # Run only high cardinality benchmarks
    python benchmarks/benchmark.py --cardinality high

    # Run threshold-neighborhood diagnostics (sort=False boundary checks)
    python benchmarks/benchmark.py --diagnostic threshold --sort-mode unsorted

    # Run full suite (core + diagnostics)
    python benchmarks/benchmark.py --cardinality all --diagnostic threshold --sort-mode unsorted

    # Run only sorted benchmarks
    python benchmarks/benchmark.py --sort-mode sorted

    # Run only sort=False benchmarks
    python benchmarks/benchmark.py --sort-mode unsorted

    # Combine options
    python benchmarks/benchmark.py --cardinality high --sort-mode sorted

    # Save per-aggregation Markdown reports
    python benchmarks/benchmark.py --output benchmarks/reports

    # Run only selected aggregation functions
    python benchmarks/benchmark.py --agg std --agg var
    python benchmarks/benchmark.py --agg median

    # Adjust sample count (applies to both cold and warm)
    python benchmarks/benchmark.py --samples 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

import numpy as np
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
from bench_utils import BenchmarkStats, compute_stats, run_cold_warm_benchmark  # noqa: E402
from datasets import PRESETS, generate_multi_key_dataset, get_dataset_info  # noqa: E402
from reporting import (  # noqa: E402
    BACKEND_DISPLAY_ORDER,
    BENCHMARK_INDEX_FILENAME,
    BENCHMARK_REPORT_GENERATED_MARKER,
    CORE_BENCHMARK_AGG,
    STATS_EVIDENCE_AGGS,
    STATS_EVIDENCE_PRESETS,
    STATS_EVIDENCE_SORTS,
    SUPPORTED_AGGS,
    benchmark_report_filename,
    format_benchmark_document,
    format_benchmark_index,
    format_correctness_section,
    format_performance_section,
    is_generated_benchmark_report,
    ordered_result_aggs,
    render_generated_markdown,
    render_high_table,
    render_standard_table,
    render_stats_evidence_section,
    render_threshold_table,
    save_results_md,
    validate_report_output_conflicts,
)

__all__ = [
    "BACKEND_DISPLAY_ORDER",
    "BENCHMARK_INDEX_FILENAME",
    "BENCHMARK_REPORT_GENERATED_MARKER",
    "CORE_BENCHMARK_AGG",
    "STATS_EVIDENCE_AGGS",
    "STATS_EVIDENCE_PRESETS",
    "STATS_EVIDENCE_SORTS",
    "SUPPORTED_AGGS",
    "benchmark_report_filename",
    "format_benchmark_document",
    "format_benchmark_index",
    "format_correctness_section",
    "format_performance_section",
    "is_generated_benchmark_report",
    "ordered_result_aggs",
    "render_generated_markdown",
    "render_high_table",
    "render_standard_table",
    "render_stats_evidence_section",
    "render_threshold_table",
    "save_results_md",
    "validate_report_output_conflicts",
]

if TYPE_CHECKING:
    from typing import Literal


def stats_evidence_workload_label(preset_name: str) -> str:
    if preset_name == STATS_EVIDENCE_PRESETS["standard"]:
        return "standard"
    if preset_name == STATS_EVIDENCE_PRESETS["high"]:
        return "high"
    return preset_name


def serialize_stats(stats: BenchmarkStats) -> dict[str, Any]:
    return stats.to_dict()


def serialize_phase_stats(phases: dict[str, BenchmarkStats]) -> dict[str, dict[str, Any]]:
    return {name: serialize_stats(stats) for name, stats in phases.items()}


def stats_mean_map(phases: dict[str, BenchmarkStats]) -> dict[str, float]:
    return {name: stats.mean for name, stats in phases.items()}


def summarize_profile_cases(cases: list[dict[str, Any]]) -> dict[str, Any] | None:
    profiled_cases = [case for case in cases if case["breakdown"] is not None]
    if not profiled_cases:
        return None

    phase_names = list(profiled_cases[0]["breakdown"]["phases"].keys())
    phase_means = {
        phase_name: sum(case["breakdown"]["phases"][phase_name].mean for case in profiled_cases)
        / len(profiled_cases)
        for phase_name in phase_names
    }
    first_breakdown = profiled_cases[0]["breakdown"]

    return {
        "preset": profiled_cases[0]["preset"],
        "workload": profiled_cases[0]["workload"],
        "sort": profiled_cases[0]["sort"],
        "aggs": [case["agg"] for case in profiled_cases],
        "phases": phase_means,
        "rust_total_s": sum(case["breakdown"]["rust_total_s"] for case in profiled_cases)
        / len(profiled_cases),
        "python_total_s": sum(case["breakdown"]["python_total_s"] for case in profiled_cases)
        / len(profiled_cases),
        "total_pipeline_s": sum(case["breakdown"]["total_pipeline_s"] for case in profiled_cases)
        / len(profiled_cases),
        "partial_group_total": first_breakdown["partial_group_total"],
        "final_group_count": first_breakdown["final_group_count"],
        "partial_to_final_ratio": first_breakdown["partial_to_final_ratio"],
        "per_agg": {
            case["agg"]: {
                "execution": case["breakdown"]["execution"],
                "phases": stats_mean_map(case["breakdown"]["phases"]),
                "rust_total_s": case["breakdown"]["rust_total_s"],
                "python_total_s": case["breakdown"]["python_total_s"],
                "total_pipeline_s": case["breakdown"]["total_pipeline_s"],
                "partial_group_total": case["breakdown"]["partial_group_total"],
                "final_group_count": case["breakdown"]["final_group_count"],
                "partial_to_final_ratio": case["breakdown"]["partial_to_final_ratio"],
            }
            for case in profiled_cases
        },
    }


def build_profile_json_payload(
    evidence: list[dict[str, Any]],
    *,
    cardinality: str,
    sort_mode: str,
    n_samples: int,
    selected_aggs: list[str] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "metadata": {
            "cardinality": cardinality,
            "sort_mode": sort_mode,
            "samples": n_samples,
            "selected_aggs": resolve_stats_evidence_aggs(selected_aggs),
        },
        "cases": [],
    }

    grouped: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    for item in evidence:
        grouped.setdefault((item["workload"], item["sort"]), []).append(item)
        breakdown = item["breakdown"]
        payload["cases"].append(
            {
                "preset": item["preset"],
                "workload": item["workload"],
                "agg": item["agg"],
                "sort": item["sort"],
                "execution": item["execution"],
                "result": {
                    "preset": item["result"]["preset"],
                    "n_rows": item["result"]["n_rows"],
                    "n_keys": item["result"]["n_keys"],
                    "key_cols": item["result"]["key_cols"],
                    "combo_cardinality": item["result"]["combo_cardinality"],
                    "group_ratio": item["result"]["group_ratio"],
                    "agg": item["result"]["agg"],
                    "sort": item["result"]["sort"],
                    "backends": {
                        backend_name: {
                            "cold_stats": serialize_stats(backend_data["cold_stats"]),
                            "warm_stats": serialize_stats(backend_data["warm_stats"]),
                            "cold_correctness": backend_data["cold_correctness"],
                            "warm_correctness": backend_data["warm_correctness"],
                        }
                        for backend_name, backend_data in item["result"]["backends"].items()
                    },
                },
                "breakdown": None
                if breakdown is None
                else {
                    "execution": breakdown["execution"],
                    "phases": serialize_phase_stats(breakdown["phases"]),
                    "phase_means": stats_mean_map(breakdown["phases"]),
                    "rust_total_s": breakdown["rust_total_s"],
                    "python_total_s": breakdown["python_total_s"],
                    "total_pipeline_s": breakdown["total_pipeline_s"],
                    "partial_group_total": breakdown["partial_group_total"],
                    "final_group_count": breakdown["final_group_count"],
                    "partial_to_final_ratio": breakdown["partial_to_final_ratio"],
                },
            }
        )

    for workload in ("standard", "high"):
        for sort in (False, True):
            cases = grouped.get((workload, sort), [])
            if not cases:
                continue
            suffix = "unsorted" if not sort else "sorted"
            summary = summarize_profile_cases(cases)
            if summary is not None:
                payload[f"single_key_{suffix}_{workload}"] = summary

    if "single_key_unsorted_high" in payload:
        payload["single_key_unsorted"] = payload["single_key_unsorted_high"]
    elif "single_key_unsorted_standard" in payload:
        payload["single_key_unsorted"] = payload["single_key_unsorted_standard"]

    if "single_key_sorted_high" in payload:
        payload["single_key_sorted"] = payload["single_key_sorted_high"]
    elif "single_key_sorted_standard" in payload:
        payload["single_key_sorted"] = payload["single_key_sorted_standard"]

    return payload


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


def measure_booster_single_key_breakdown(
    preset_name: str,
    agg: str,
    sort: bool,
    n_samples: int,
    *,
    ignore_force_pandas_sort: bool = False,
) -> dict[str, Any] | None:
    import pandas_booster._abi_compat as abi_compat
    import pandas_booster._rust as rust
    from pandas_booster import _groupby_accel as groupby_accel

    config = PRESETS[preset_name]
    df = generate_multi_key_dataset(**config)
    key_cols = [col for col, _ in config["key_configs"]]
    if len(key_cols) != 1:
        raise ValueError("Breakdown evidence only supports single-key presets")

    key_col = cast(pd.Series, df[key_cols[0]])
    val_col = cast(pd.Series, df["value"])
    key_dtype = groupby_accel.capture_key_numpy_dtype(key_col)
    value_dtype = groupby_accel.capture_value_numpy_dtype(val_col)
    is_val_int = pd.api.types.is_integer_dtype(val_col)
    dispatch = resolve_booster_benchmark_dispatch(
        df,
        key_cols,
        "value",
        agg,
        sort,
        ignore_force_pandas_sort=ignore_force_pandas_sort,
    )
    rust_func = dispatch["rust_func"]
    if rust_func is None:
        return None

    needs_python_sort = bool(dispatch["needs_python_sort"])
    if needs_python_sort and sort:
        return None

    profile_func_name = f"profile_{rust_func.__name__}"
    profile_func = getattr(rust, profile_func_name, None)
    if profile_func is None:
        return None

    phase_samples: dict[str, list[float]] = {
        "prepare_inputs_s": [],
        "local_build_s": [],
        "merge_s": [],
        "reorder_s": [],
        "materialize_s": [],
        "python_normalize_s": [],
        "python_series_build_s": [],
        "rust_total_s": [],
        "python_total_s": [],
        "total_pipeline_s": [],
    }
    partial_group_total = 0
    final_group_count = 0
    partial_to_final_ratio = 0.0

    for _ in range(n_samples):
        total_start = time.perf_counter()

        prepare_start = time.perf_counter()
        keys = groupby_accel.to_i64_contiguous(key_col.to_numpy(copy=False))
        if is_val_int:
            values = np.ascontiguousarray(val_col.to_numpy(dtype=np.int64))
        else:
            values = np.ascontiguousarray(val_col.to_numpy(dtype=np.float64))
        phase_samples["prepare_inputs_s"].append(time.perf_counter() - prepare_start)

        result_keys, result_values, profile = profile_func(keys, values)
        phase_samples["local_build_s"].append(float(profile["local_build_s"]))
        phase_samples["merge_s"].append(float(profile["merge_s"]))
        phase_samples["reorder_s"].append(float(profile["reorder_s"]))
        phase_samples["materialize_s"].append(float(profile["materialize_s"]))
        phase_samples["rust_total_s"].append(float(profile["rust_total_s"]))
        partial_group_total = int(profile["partial_group_total"])
        final_group_count = int(profile["final_group_count"])
        partial_to_final_ratio = float(profile["partial_to_final_ratio"])

        normalize_start = time.perf_counter()
        result_values_arr = abi_compat.normalize_result_values(
            result_values,
            agg=agg,
            is_val_int=is_val_int,
            context="benchmark",
        )
        normalize_duration = time.perf_counter() - normalize_start
        phase_samples["python_normalize_s"].append(normalize_duration)

        build_start = time.perf_counter()
        _ = groupby_accel.build_series_from_single_result(
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
        build_duration = time.perf_counter() - build_start
        phase_samples["python_series_build_s"].append(build_duration)
        phase_samples["python_total_s"].append(normalize_duration + build_duration)
        phase_samples["total_pipeline_s"].append(time.perf_counter() - total_start)

    stats = {name: compute_stats(samples) for name, samples in phase_samples.items()}
    return {
        "execution": dispatch["execution"],
        "phases": stats,
        "rust_total_s": stats["rust_total_s"].mean,
        "python_total_s": stats["python_total_s"].mean,
        "total_pipeline_s": stats["total_pipeline_s"].mean,
        "partial_group_total": partial_group_total,
        "final_group_count": final_group_count,
        "partial_to_final_ratio": partial_to_final_ratio,
    }


def resolve_selected_aggs(selected_aggs: list[str] | None) -> list[str] | None:
    if selected_aggs is None:
        return None

    deduped: list[str] = []
    for agg in selected_aggs:
        if agg not in deduped:
            deduped.append(agg)
    return deduped


def resolve_core_aggs(selected_aggs: list[str] | None) -> list[str]:
    aggs = resolve_selected_aggs(selected_aggs)
    if aggs is None:
        return [CORE_BENCHMARK_AGG]
    return aggs


def resolve_stats_evidence_aggs(selected_aggs: list[str] | None) -> list[str]:
    aggs = resolve_selected_aggs(selected_aggs)
    if aggs is None:
        return list(STATS_EVIDENCE_AGGS)
    return [agg for agg in STATS_EVIDENCE_AGGS if agg in aggs]


def collect_stats_evidence(
    n_samples: int,
    cardinality: str,
    sort_mode: str,
    selected_aggs: list[str] | None = None,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    evidence_aggs = resolve_stats_evidence_aggs(selected_aggs)
    if not evidence_aggs:
        return evidence

    preset_names: list[str] = []
    if cardinality in {"all", "standard"}:
        preset_names.append(STATS_EVIDENCE_PRESETS["standard"])
    if cardinality in {"all", "high"}:
        preset_names.append(STATS_EVIDENCE_PRESETS["high"])

    sorts = resolve_sorts(sort_mode)

    for preset_name in preset_names:
        config = PRESETS[preset_name]
        key_cols = [col for col, _ in config["key_configs"]]
        workload = stats_evidence_workload_label(preset_name)
        for agg in evidence_aggs:
            for sort in sorts:
                result = benchmark_single(
                    preset_name,
                    agg=agg,
                    sort=sort,
                    n_samples=n_samples,
                    verify_correctness=True,
                )
                df = generate_multi_key_dataset(**config)
                execution = {
                    "pandas": f"pandas.groupby.{agg}",
                    "booster": describe_booster_execution(df, key_cols, "value", agg, sort),
                }
                if HAS_POLARS:
                    execution["polars"] = f"polars.group_by.agg({agg})"
                evidence.append(
                    {
                        "preset": preset_name,
                        "workload": workload,
                        "agg": agg,
                        "sort": sort,
                        "result": result,
                        "execution": execution,
                        "breakdown": measure_booster_single_key_breakdown(
                            preset_name,
                            agg,
                            sort,
                            n_samples,
                        ),
                    }
                )
    return evidence


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
    """Worker function: runs in fresh process, measures cold OR warm.

    Measurement protocol:
    - mode="cold": Measure 1st run. Return cold_time_s.
    - mode="warm": Run 1st (cold, discard) -> Run 2nd (warmup, discard) -> Measure 3rd.
      Return warm_time_s.

    Args:
        preset_name: Dataset preset name.
        backend: Which backend to benchmark.
        agg: Aggregation function.
        sort: Whether to sort results.
        verify_correctness: Whether to verify correctness against a Pandas baseline.
            When enabled, Booster and (if installed) Polars results are validated.
        mode: "cold" or "warm".

    Returns:
        Dictionary with cold_time_s OR warm_time_s.
    """
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
            # Defensive: Pandas and Polars differ on null-group defaults.
            if df[col].isna().any():
                raise ValueError(
                    f"Benchmark key column {col!r} contains nulls; semantics differ across engines"
                )

        # Defensive: keep benchmark datasets free of NaNs in the value column.
        # While Pandas/Polars often align on NaN handling, edge cases can diverge
        # (especially once NaN interacts with casting/Arrow conversion). This is
        # outside the timed region, so we prefer failing fast.
        if df[value_col].isna().any():
            raise ValueError(
                f"Benchmark value column {value_col!r} contains NaNs; "
                "semantics can diverge across engines"
            )

        # Note: count semantics are especially sensitive to NaN vs null, but the
        # invariant above (no NaNs in value) already enforces alignment.

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
            # When sort=False, align semantics with Pandas/Booster by preserving
            # appearance order (first-seen group order).
            result = df_polars.group_by(key_cols, maintain_order=not sort).agg(agg_expr)
            if sort:
                result = result.sort(key_cols)
            return result

        run_once = run_once_polars

    else:
        raise ValueError(f"Unsupported benchmark backend: {backend!r}")

    def pandas_baseline() -> pd.Series:
        # Note: we keep Pandas defaults (e.g., dropna=True) to match Pandas semantics.
        # Our benchmark datasets use integer keys without nulls.
        return cast(pd.Series, getattr(df.groupby(key_cols, sort=sort)[value_col], agg)())

    def polars_to_pandas_series(result_df: Any) -> pd.Series:
        if not HAS_POLARS:
            raise ImportError("Polars is not installed")
        assert pl is not None

        if not isinstance(result_df, pl.DataFrame):
            raise TypeError(f"Expected polars DataFrame, got {type(result_df)}")

        # Preserve Polars output row order. Do NOT sort here; sort semantics are
        # handled by the Polars query itself (and we validate sort=False order).
        try:
            pdf = result_df.to_pandas()
        except Exception:
            # Fallback path if Arrow conversion is unavailable.
            # This is slower and more memory-heavy, but runs outside timed sections.
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
            # Ensure naming parity for comparisons.
            if s.name != value_col:
                s = s.copy()
                s.name = value_col
            if len(key_cols) == 1:
                if s.index.name != key_cols[0]:
                    s.index.name = key_cols[0]
            else:
                if list(s.index.names) != key_cols:
                    s.index.names = key_cols
            return s
        if backend_name == "polars":
            return polars_to_pandas_series(result_obj)

        raise ValueError(f"Unsupported benchmark backend: {backend_name!r}")

    def assert_matches_baseline(actual: pd.Series, expected: pd.Series) -> None:
        # Ordering semantics:
        # - sort=True: expect key-sorted results from all backends
        # - sort=False: validate Pandas appearance order (first-seen group order)
        #
        # Note: float min/max can differ only by signed-zero (-0.0 vs +0.0) across
        # engines even when numerically equivalent, so we use tolerance-based
        # comparisons for float-valued aggregations.

        if agg == "count":
            pd.testing.assert_series_equal(
                actual,
                expected,
                check_exact=True,
                check_dtype=False,
            )
            return

        is_float = pd.api.types.is_float_dtype(expected)
        if is_float:
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

    elif mode == "warm":
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


def benchmark_single(
    preset_name: str,
    agg: str = "sum",
    sort: bool = True,
    n_samples: int = 5,
    verify_correctness: bool = True,
) -> dict[str, Any]:
    """Run a single benchmark across Pandas/Booster/Polars.

    Args:
        preset_name: Name of the dataset preset from PRESETS.
        agg: Aggregation function ("sum", "mean", "prod", "std", "var", "min", "max", "count").
        sort: Whether to sort the result.
        n_samples: Number of samples (= number of fresh processes).
        verify_correctness: Whether to verify results match.

    Returns:
        Dictionary with benchmark results and statistics.
    """
    config = PRESETS[preset_name]
    df_temp = generate_multi_key_dataset(**config)
    key_cols = [col for col, _ in config["key_configs"]]
    dataset_info = get_dataset_info(df_temp, key_cols)
    del df_temp

    script_path = Path(__file__).resolve()
    backends = [b for b in BACKEND_DISPLAY_ORDER if b != "polars" or HAS_POLARS]

    results = {}
    for backend in backends:
        print(f"  [{backend.upper()}]")
        worker_args = {
            "preset_name": preset_name,
            "backend": backend,
            "agg": agg,
            "sort": sort,
            "verify_correctness": verify_correctness and backend != "pandas",
        }

        bench_result = run_cold_warm_benchmark(
            script_path,
            worker_args,
            n_samples=n_samples,
            timeout_per_worker=300,
        )

        results[backend] = {
            "cold_stats": bench_result["cold_stats"],
            "warm_stats": bench_result["warm_stats"],
            "cold_correctness": bench_result.get("cold_correctness", "not_checked"),
            "warm_correctness": bench_result.get("warm_correctness", "not_checked"),
        }

    return {
        "preset": preset_name,
        "n_rows": dataset_info["n_rows"],
        "n_keys": dataset_info["n_keys"],
        "key_cols": key_cols,
        "combo_cardinality": dataset_info["combo_cardinality"],
        "group_ratio": round(dataset_info["group_ratio"], 6),
        "agg": agg,
        "sort": sort,
        "backends": results,
    }


def resolve_core_presets(cardinality: str) -> list[str]:
    """Resolve core cardinality option to list of preset names.

    Args:
        cardinality: "all", "standard", or "high".

    Returns:
        List of preset names.
    """
    standard = ["1key", "2key", "3key", "4key", "5key"]
    high = ["high_cardinality_1key", "high_cardinality_2key", "high_cardinality_3key"]
    if cardinality == "standard":
        return standard
    elif cardinality == "high":
        return high
    elif cardinality == "all":
        return standard + high
    else:
        raise ValueError(f"Unknown cardinality: {cardinality}")


def resolve_diagnostic_presets(diagnostic: str) -> list[str]:
    """Resolve diagnostic option to list of preset names."""
    threshold = ["threshold_180k", "threshold_200k", "threshold_220k"]

    if diagnostic == "none":
        return []
    elif diagnostic == "threshold":
        return threshold
    else:
        raise ValueError(f"Unknown diagnostic: {diagnostic}")


def resolve_sorts(sort_mode: str) -> list[bool]:
    """Resolve sort-mode option to list of sort values.

    Args:
        sort_mode: "all", "sorted", or "unsorted".

    Returns:
        List of sort boolean values.
    """
    if sort_mode == "sorted":
        return [True]
    elif sort_mode == "unsorted":
        return [False]
    elif sort_mode == "all":
        return [True, False]
    else:
        raise ValueError(f"Unknown sort-mode: {sort_mode}")


def run_benchmarks(
    cardinality: str,
    diagnostic: str,
    sort_mode: str,
    n_samples: int = 5,
    aggs: list[str] | None = None,
) -> list[dict]:
    """Run benchmark suite based on cardinality and sort-mode.

    Args:
        cardinality: "all", "standard", or "high".
        diagnostic: "none" or "threshold".
        sort_mode: "all", "sorted", or "unsorted".
        n_samples: Number of samples per benchmark (applies to both cold and warm).

    Returns:
        List of benchmark result dictionaries.
    """
    core_presets = resolve_core_presets(cardinality)
    diagnostic_presets = resolve_diagnostic_presets(diagnostic)
    sorts = resolve_sorts(sort_mode)

    if diagnostic == "threshold" and sort_mode != "unsorted":
        raise ValueError(
            "--diagnostic threshold requires --sort-mode unsorted "
            "(threshold diagnostics are sort=False boundary checks)"
        )

    presets = core_presets + diagnostic_presets
    selected_aggs = resolve_core_aggs(aggs)

    cardinality_label = cardinality.capitalize()
    if cardinality == "all":
        cardinality_label = "Standard + High"
    diagnostics_label = "None" if diagnostic == "none" else "Threshold Neighborhood"

    print("=" * 90)
    print(f"Pandas-Booster Benchmarks: {cardinality_label} Cardinality")
    print("=" * 90)
    print(f"Diagnostics: {diagnostics_label}")
    print(f"Samples per benchmark: {n_samples} (fresh processes for both cold and warm)")
    print(f"Aggregations: {', '.join(selected_aggs)}")
    print()

    results = []

    for sort_val in sorts:
        sort_str = "Sorted" if sort_val else "Unsorted"
        print(f"\n--- Running {sort_str} Benchmarks ---")

        for preset in presets:
            for agg in selected_aggs:
                print(f"\n[{preset}] Running {sort_str} cold/warm benchmark for agg={agg}...")

                result = benchmark_single(
                    preset,
                    agg=agg,
                    sort=sort_val,
                    n_samples=n_samples,
                    verify_correctness=True,
                )
                results.append(result)

                backends = result["backends"]
                print(f"  Groups: {result['combo_cardinality']:,}")

                for backend_name in BACKEND_DISPLAY_ORDER:
                    if backend_name not in backends:
                        continue
                    backend_data = backends[backend_name]
                    cold_stats = backend_data.get("cold_stats")
                    warm_stats = backend_data.get("warm_stats")
                    cold_str = "n/a" if cold_stats is None else cold_stats.format_ms(2)
                    warm_str = "n/a" if warm_stats is None else warm_stats.format_ms(2)

                    correctness_str = ""
                    if backend_name != "pandas":
                        cold_corr = backend_data.get("cold_correctness", "not_checked")
                        warm_corr = backend_data.get("warm_correctness", "not_checked")
                        if cold_corr != "not_checked" or warm_corr != "not_checked":
                            correctness_str = f" | Correctness: cold={cold_corr}, warm={warm_corr}"

                    print(
                        f"  {backend_name:8s} | Cold: {cold_str} | "
                        f"Warm: {warm_str}{correctness_str}"
                    )

    print("\n" + "=" * 90)
    print("Performance Tables")
    print("=" * 90)
    print(format_performance_section(results, sort_mode, cardinality, diagnostic))
    print()

    return results


def save_profile_json(
    evidence: list[dict[str, Any]],
    output_path: str,
    *,
    cardinality: str,
    sort_mode: str,
    n_samples: int,
    selected_aggs: list[str] | None,
) -> None:
    path = Path(output_path)
    if not path.suffix:
        path = path.with_suffix(".json")
    elif path.suffix != ".json":
        print(f"Warning: Output path should have .json extension, got {path.suffix}")

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_profile_json_payload(
        evidence,
        cardinality=cardinality,
        sort_mode=sort_mode,
        n_samples=n_samples,
        selected_aggs=selected_aggs,
    )
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Profile JSON saved to: {path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Official pandas-booster benchmarks with Cold/Warm measurement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmarks/benchmark.py
      # Run default benchmarks (cardinality=all, diagnostic=none)
  python benchmarks/benchmark.py --cardinality all
      # Run core benchmarks only (standard + high)
  python benchmarks/benchmark.py --cardinality standard             # Standard only
  python benchmarks/benchmark.py --cardinality high                 # High only
  python benchmarks/benchmark.py --agg std --agg var                # Run only std/var benchmarks
  python benchmarks/benchmark.py --agg median                       # Run only median benchmarks
  python benchmarks/benchmark.py --agg prod                         # Run only product benchmarks
  python benchmarks/benchmark.py --agg min --agg max               # Run only min/max benchmarks
  python benchmarks/benchmark.py --diagnostic threshold --sort-mode unsorted
      # Add threshold diagnostics
  python benchmarks/benchmark.py --cardinality all --diagnostic threshold --sort-mode unsorted
      # Core + diagnostics
  python benchmarks/benchmark.py --sort-mode sorted                 # Sorted only
  python benchmarks/benchmark.py --cardinality high --sort-mode unsorted  # Combine
  python benchmarks/benchmark.py --output benchmarks/reports        # Save per-aggregation reports
  python benchmarks/benchmark.py --samples 10                       # Adjust sample count

Environment:
  PANDAS_BOOSTER_FORCE_PANDAS_SORT=1  # force Python sort_index() (panic button)
  PANDAS_BOOSTER_FORCE_PANDAS_SORT=0  # use Rust-side sorting (default)
        """,
    )
    parser.add_argument(
        "--cardinality",
        choices=["all", "standard", "high"],
        default="all",
        help="Workload cardinality suite to run (default: all = standard + high)",
    )
    parser.add_argument(
        "--diagnostic",
        choices=["none", "threshold"],
        default="none",
        help="Internal diagnostic suite to add (default: none)",
    )
    parser.add_argument(
        "--sort-mode",
        choices=["all", "sorted", "unsorted"],
        default="all",
        help="Which sort mode to run (default: all)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of samples per benchmark (applies to both cold and warm, default: 5)",
    )
    parser.add_argument(
        "--agg",
        dest="aggs",
        action="append",
        choices=SUPPORTED_AGGS,
        help=(
            "Aggregation function to benchmark. Repeatable. "
            "Defaults to core=sum for benchmark reports; profile JSON defaults to std/var evidence."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help=(
            "Output directory for per-aggregation Markdown reports (e.g., benchmarks/reports). "
            "Existing README.md/<agg>.md files must already be benchmark-generated or the command "
            "will fail to avoid overwriting user files."
        ),
    )
    parser.add_argument(
        "--profile-json",
        type=str,
        help="Internal-use profile JSON output path for single-key std/var evidence",
    )
    parser.add_argument(
        "--worker",
        type=str,
        help="Worker mode (internal use): JSON args for single benchmark run",
    )

    args = parser.parse_args()

    if args.diagnostic == "threshold" and args.sort_mode != "unsorted":
        parser.error(
            "--diagnostic threshold requires --sort-mode unsorted "
            "(threshold diagnostics are sort=False boundary checks)"
        )

    # Default to Rust-side sorting for sort=True benchmarks unless explicitly forced off.
    if not args.worker:
        os.environ.setdefault("PANDAS_BOOSTER_FORCE_PANDAS_SORT", "0")
        print(
            "PANDAS_BOOSTER_FORCE_PANDAS_SORT="
            f"{os.environ.get('PANDAS_BOOSTER_FORCE_PANDAS_SORT')}",
            file=sys.stderr,
        )
        print(
            "Note: benchmarks default Rust-side sort=True unless you set "
            "PANDAS_BOOSTER_FORCE_PANDAS_SORT=1",
            file=sys.stderr,
        )

    if args.worker:
        worker_args = json.loads(args.worker)
        output_file = worker_args.pop("output_file", None)

        try:
            result = benchmark_worker(**worker_args)
        except Exception as e:
            raise e

        if output_file:
            with open(output_file, "w") as f:
                json.dump(result, f)
        else:
            print(json.dumps(result))
        return

    all_results = run_benchmarks(
        cardinality=args.cardinality,
        diagnostic=args.diagnostic,
        sort_mode=args.sort_mode,
        n_samples=args.samples,
        aggs=args.aggs,
    )

    profile_stats_evidence: list[dict[str, Any]] = []
    if args.profile_json:
        profile_stats_evidence = collect_stats_evidence(
            args.samples,
            args.cardinality,
            args.sort_mode,
            args.aggs,
        )

    if args.output:
        result_aggs = {result["agg"] for result in all_results}
        output_stats_evidence = [
            item for item in profile_stats_evidence if item["agg"] in result_aggs
        ]
        if not args.profile_json and args.aggs is not None:
            output_stats_evidence = collect_stats_evidence(
                args.samples,
                args.cardinality,
                args.sort_mode,
                args.aggs,
            )

        save_results_md(
            all_results,
            output_stats_evidence,
            args.output,
            args.sort_mode,
            args.cardinality,
            args.diagnostic,
        )

    if args.profile_json:
        save_profile_json(
            profile_stats_evidence,
            args.profile_json,
            cardinality=args.cardinality,
            sort_mode=args.sort_mode,
            n_samples=args.samples,
            selected_aggs=args.aggs,
        )

    print("\n" + "=" * 90)
    print("Summary")
    print("=" * 90)
    print(f"Total benchmarks run: {len(all_results)}")

    return all_results


if __name__ == "__main__":
    main()
