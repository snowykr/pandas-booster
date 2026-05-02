"""Official benchmark script for pandas-booster with Cold/Warm measurement.

This script generates benchmark results for the README Performance section.
It measures single-key through five-key groupby operations with proper
cold/warm separation and statistical analysis.

Cold/Warm Definitions:
- Cold: First call in a fresh Python process (after import + data preparation)
- Warm: Steady-state performance after cold + 1 warmup run (discarded), 5 samples

Usage:
    # Run core benchmarks (standard + high cardinality, sorted + sort=False)
    python benches/benchmark.py

    # Run only standard cardinality benchmarks
    python benches/benchmark.py --cardinality standard

    # Run only high cardinality benchmarks
    python benches/benchmark.py --cardinality high

    # Run threshold-neighborhood diagnostics (sort=False boundary checks)
    python benches/benchmark.py --diagnostic threshold --sort-mode unsorted

    # Run full suite (core + diagnostics)
    python benches/benchmark.py --cardinality all --diagnostic threshold --sort-mode unsorted

    # Run only sorted benchmarks
    python benches/benchmark.py --sort-mode sorted

    # Run only sort=False benchmarks
    python benches/benchmark.py --sort-mode unsorted

    # Combine options
    python benches/benchmark.py --cardinality high --sort-mode sorted

    # Save results to markdown file
    python benches/benchmark.py --output results.md

    # Run only selected aggregation functions
    python benches/benchmark.py --agg std --agg var

    # Adjust sample count (applies to both cold and warm)
    python benches/benchmark.py --samples 10
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


# Keep benchmark progress output aligned with README table column order.
BACKEND_DISPLAY_ORDER: tuple[str, ...] = ("pandas", "polars", "booster")

sys.path.insert(0, str(Path(__file__).parent))
from bench_utils import BenchmarkStats, compute_stats, run_cold_warm_benchmark  # noqa: E402
from datasets import PRESETS, generate_multi_key_dataset, get_dataset_info  # noqa: E402

if TYPE_CHECKING:
    from typing import Literal


SUPPORTED_AGGS: tuple[str, ...] = ("sum", "mean", "prod", "std", "var", "min", "max", "count")
CORE_BENCHMARK_AGG = "sum"
STATS_EVIDENCE_AGGS: tuple[str, str] = ("std", "var")
STATS_EVIDENCE_SORTS: tuple[bool, bool] = (True, False)
STATS_EVIDENCE_PRESETS: dict[str, str] = {
    "standard": "1key",
    "high": "high_cardinality_1key",
}


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
        phase_name: sum(
            case["breakdown"]["phases"][phase_name].mean for case in profiled_cases
        )
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
        "total_pipeline_s": sum(
            case["breakdown"]["total_pipeline_s"] for case in profiled_cases
        )
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

    if agg not in {"std", "var"} and len(df) < rust.get_fallback_threshold():
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
    if not compatibility.supported or compatibility.force_pandas:
        return {
            "execution": f"booster->pandas.groupby.{agg}",
            "rust_func": None,
            "needs_python_sort": False,
        }

    if (
        len(key_cols) == 1
        and pd.api.types.is_float_dtype(val_col)
        and agg in {"sum", "mean", "prod"}
        and force_pandas_float_groupby_enabled()
    ):
        return {
            "execution": f"booster->pandas.groupby.{agg}",
            "rust_func": None,
            "needs_python_sort": False,
        }

    is_val_int = pd.api.types.is_integer_dtype(val_col)
    prefix = "groupby_multi" if len(key_cols) > 1 else "groupby"
    suffix = "i64" if is_val_int else "f64"
    rust_func, needs_python_sort = groupby_accel.select_rust_groupby_func(
        rust,
        f"{prefix}_{agg}_{suffix}",
        sort=sort,
        n_rows=len(df),
        force_pandas_sort=(
            False
            if ignore_force_pandas_sort
            else bool(sort) and force_pandas_sort_enabled()
        ),
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


def render_stats_evidence_section(evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return ""

    evidence_aggs = sorted({item["agg"] for item in evidence})
    if evidence_aggs == ["std", "var"]:
        heading = "## Single-Key `std`/`var` Evidence"
    elif len(evidence_aggs) == 1:
        heading = f"## Single-Key `{evidence_aggs[0]}` Evidence"
    else:
        agg_list = "/".join(f"`{agg}`" for agg in evidence_aggs)
        heading = f"## Single-Key {agg_list} Evidence"

    lines: list[str] = []
    lines.append(heading)
    lines.append("")
    agg_phrase = "`std` and `var`" if evidence_aggs == ["std", "var"] else ", ".join(
        f"`{agg}`" for agg in evidence_aggs
    )
    lines.append(
        f"{agg_phrase} scale on the Rust-first path because each worker accumulates "
        "mergeable `(count, mean, m2)` state and Rayon only merges those partial states at the end."
    )
    lines.append(
        "When the dataset is smaller or the group count is low, the fixed overhead from NumPy "
        "contiguity/copy work, the Python↔Rust boundary, and Series materialization "
        "eats into that gain even though the kernel itself remains parallel."
    )
    lines.append("")
    lines.append("| Workload | Agg | Sort | Backend | Execution | Cold | Warm |")
    lines.append("|----------|-----|------|---------|-----------|------|------|")

    for item in evidence:
        agg = item["agg"]
        sort = "True" if item["sort"] else "False"
        result = item["result"]
        execution = item["execution"]
        for backend_name in BACKEND_DISPLAY_ORDER:
            backend_data = result["backends"].get(backend_name)
            if backend_data is None:
                continue
            cold_stats: BenchmarkStats = backend_data["cold_stats"]
            warm_stats: BenchmarkStats = backend_data["warm_stats"]
            row = [
                item["workload"],
                f"`{agg}`",
                sort,
                backend_name,
                f"`{execution[backend_name]}`",
                cold_stats.format_ms(2),
                warm_stats.format_ms(2),
            ]
            lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("### Booster conversion vs compute breakdown")
    lines.append("")
    lines.append(
        "The table below isolates the Rust-first Booster path for the same single-key datasets. "
        "`local_build`, `merge`, `reorder`, and `materialize` come from the "
        "internal Rust profiling hook, while the Python-side phases measure "
        "post-kernel normalization and final pandas Series assembly."
    )
    lines.append("")
    breakdown_columns = [
        "Workload",
        "Agg",
        "Sort",
        "Execution",
        "Prepare inputs",
        "Local build",
        "Merge",
        "Reorder",
        "Materialize",
        "Python normalize",
        "Series build",
        "Rust total",
        "Total pipeline",
        "Partial groups",
        "Final groups",
        "Partial/final",
    ]
    breakdown_separators = ["-" * len(column) for column in breakdown_columns]
    lines.append("| " + " | ".join(breakdown_columns) + " |")
    separator_row = "|" + "|".join(
        f"{separator:-^{len(separator) + 2}}"
        for separator in breakdown_separators
    )
    lines.append(separator_row + "|")
    has_breakdown_rows = False
    for item in evidence:
        breakdown = item["breakdown"]
        if breakdown is None:
            continue
        has_breakdown_rows = True
        phases = breakdown["phases"]
        row = [
            item["workload"],
            f"`{item['agg']}`",
            "True" if item["sort"] else "False",
            f"`{breakdown['execution']}`",
            phases["prepare_inputs_s"].format_ms(2),
            phases["local_build_s"].format_ms(2),
            phases["merge_s"].format_ms(2),
            phases["reorder_s"].format_ms(2),
            phases["materialize_s"].format_ms(2),
            phases["python_normalize_s"].format_ms(2),
            phases["python_series_build_s"].format_ms(2),
            phases["rust_total_s"].format_ms(2),
            phases["total_pipeline_s"].format_ms(2),
            f"{breakdown['partial_group_total']:,}",
            f"{breakdown['final_group_count']:,}",
            f"{breakdown['partial_to_final_ratio']:.3f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    if not has_breakdown_rows:
        lines.append(
            "No Rust-only Booster breakdown rows were available for the selected evidence cases."
        )

    lines.append("")
    return "\n".join(lines)


def benchmark_worker(
    preset_name: str,
    backend: str,
    agg: Literal["sum", "mean", "prod", "std", "var", "min", "max", "count"] = "sum",
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


def render_standard_table(results: list[dict]) -> str:
    """Render standard cardinality table (Single-key through 5-key) with cold/warm stats.

    Args:
        results: List of benchmark results for standard presets (can include both sorted and
            unsorted).

    Returns:
        Markdown table string.
    """
    if not results:
        return ""

    preset_order = ["1key", "2key", "3key", "4key", "5key"]
    preset_labels = {
        "1key": "Single-key",
        "2key": "2-key",
        "3key": "3-key",
        "4key": "4-key",
        "5key": "5-key",
    }

    # Group results by preset and sort
    results_by_preset_sort = {}
    for r in results:
        key = (r["preset"], r["sort"])
        results_by_preset_sort[key] = r

    lines = []
    lines.append("| Operation | Groups | Sort | Type | Pandas | Polars | Booster |")
    lines.append("|-----------|--------|------|------|--------|--------|---------|")

    prev_label = None
    prev_groups = None
    prev_sort_str = None

    for preset in preset_order:
        for sort_val in [True, False]:
            key = (preset, sort_val)
            if key not in results_by_preset_sort:
                continue

            r = results_by_preset_sort[key]
            label = preset_labels[preset]
            groups = f"{r['combo_cardinality']:,}"
            sort_str = "True" if sort_val else "False"
            backends = r["backends"]

            if "pandas" not in backends:
                continue

            pandas_cold: BenchmarkStats = backends["pandas"]["cold_stats"]
            pandas_warm: BenchmarkStats = backends["pandas"]["warm_stats"]
            pandas_cold_mean = pandas_cold.mean
            pandas_warm_mean = pandas_warm.mean

            def fmt_cell_cold(name, *, backends=backends, pandas_cold_mean=pandas_cold_mean):
                if name not in backends:
                    return "-"
                cold_stats: BenchmarkStats = backends[name]["cold_stats"]

                cold_mean_ms = cold_stats.mean * 1000
                cold_std_ms = cold_stats.std * 1000
                cold_speedup = pandas_cold_mean / cold_stats.mean if cold_stats.mean > 0 else 0

                cold_str = f"{cold_mean_ms:.1f}±{cold_std_ms:.1f}ms"

                if name == "pandas":
                    return f"{cold_str} (1.0x)"

                cold_speedup_str = (
                    f"**{cold_speedup:.1f}x**" if cold_speedup >= 1.1 else f"{cold_speedup:.1f}x"
                )
                return f"{cold_str} ({cold_speedup_str})"

            def fmt_cell_warm(name, *, backends=backends, pandas_warm_mean=pandas_warm_mean):
                if name not in backends:
                    return "-"
                warm_stats: BenchmarkStats = backends[name]["warm_stats"]

                warm_mean_ms = warm_stats.mean * 1000
                warm_std_ms = warm_stats.std * 1000
                warm_speedup = pandas_warm_mean / warm_stats.mean if warm_stats.mean > 0 else 0

                warm_str = f"{warm_mean_ms:.1f}±{warm_std_ms:.1f}ms"

                if name == "pandas":
                    return f"{warm_str} (1.0x)"

                warm_speedup_str = (
                    f"**{warm_speedup:.1f}x**" if warm_speedup >= 1.1 else f"{warm_speedup:.1f}x"
                )
                return f"{warm_str} ({warm_speedup_str})"

            pandas_cold_str = fmt_cell_cold("pandas")
            polars_cold_str = fmt_cell_cold("polars")
            booster_cold_str = fmt_cell_cold("booster")

            pandas_warm_str = fmt_cell_warm("pandas")
            polars_warm_str = fmt_cell_warm("polars")
            booster_warm_str = fmt_cell_warm("booster")

            # First row (Cold)
            display_label = label if label != prev_label else ""
            display_groups = groups if groups != prev_groups else ""
            display_sort = sort_str if sort_str != prev_sort_str else ""

            lines.append(
                f"| {display_label} | {display_groups} | {display_sort} | Cold | "
                f"{pandas_cold_str} | {polars_cold_str} | {booster_cold_str} |"
            )

            # Second row (Warm) - always hide label, groups, sort
            lines.append(
                f"|  |  |  | Warm | {pandas_warm_str} | {polars_warm_str} | {booster_warm_str} |"
            )

            prev_label = label
            prev_groups = groups
            prev_sort_str = sort_str

    return "\n".join(lines)


def render_high_table(results: list[dict]) -> str:
    """Render high cardinality table with cold/warm stats.

    Args:
        results: List of benchmark results for high cardinality presets (can include both sorted
            and unsorted).

    Returns:
        Markdown table string.
    """
    if not results:
        return ""

    preset_order = ["high_cardinality_1key", "high_cardinality_2key", "high_cardinality_3key"]
    preset_labels = {
        "high_cardinality_1key": "Single-key",
        "high_cardinality_2key": "2-key",
        "high_cardinality_3key": "3-key",
    }

    # Group results by preset and sort
    results_by_preset_sort = {}
    for r in results:
        key = (r["preset"], r["sort"])
        results_by_preset_sort[key] = r

    lines = []
    lines.append("| Operation | Groups | Sort | Type | Pandas | Polars | Booster |")
    lines.append("|-----------|--------|------|------|--------|--------|---------|")

    prev_label = None
    prev_groups = None
    prev_sort_str = None

    for preset in preset_order:
        for sort_val in [True, False]:
            key = (preset, sort_val)
            if key not in results_by_preset_sort:
                continue

            r = results_by_preset_sort[key]
            label = preset_labels[preset]
            groups = f"{r['combo_cardinality']:,}"
            sort_str = "True" if sort_val else "False"
            backends = r["backends"]

            if "pandas" not in backends:
                continue

            pandas_cold: BenchmarkStats = backends["pandas"]["cold_stats"]
            pandas_warm: BenchmarkStats = backends["pandas"]["warm_stats"]
            pandas_cold_mean = pandas_cold.mean
            pandas_warm_mean = pandas_warm.mean

            def fmt_cell_cold(name, *, backends=backends, pandas_cold_mean=pandas_cold_mean):
                if name not in backends:
                    return "-"
                cold_stats: BenchmarkStats = backends[name]["cold_stats"]

                cold_mean_ms = cold_stats.mean * 1000
                cold_std_ms = cold_stats.std * 1000
                cold_speedup = pandas_cold_mean / cold_stats.mean if cold_stats.mean > 0 else 0

                cold_str = f"{cold_mean_ms:.1f}±{cold_std_ms:.1f}ms"

                if name == "pandas":
                    return f"{cold_str} (1.0x)"

                cold_speedup_str = (
                    f"**{cold_speedup:.1f}x**" if cold_speedup >= 1.1 else f"{cold_speedup:.1f}x"
                )
                return f"{cold_str} ({cold_speedup_str})"

            def fmt_cell_warm(name, *, backends=backends, pandas_warm_mean=pandas_warm_mean):
                if name not in backends:
                    return "-"
                warm_stats: BenchmarkStats = backends[name]["warm_stats"]

                warm_mean_ms = warm_stats.mean * 1000
                warm_std_ms = warm_stats.std * 1000
                warm_speedup = pandas_warm_mean / warm_stats.mean if warm_stats.mean > 0 else 0

                warm_str = f"{warm_mean_ms:.1f}±{warm_std_ms:.1f}ms"

                if name == "pandas":
                    return f"{warm_str} (1.0x)"

                warm_speedup_str = (
                    f"**{warm_speedup:.1f}x**" if warm_speedup >= 1.1 else f"{warm_speedup:.1f}x"
                )
                return f"{warm_str} ({warm_speedup_str})"

            pandas_cold_str = fmt_cell_cold("pandas")
            polars_cold_str = fmt_cell_cold("polars")
            booster_cold_str = fmt_cell_cold("booster")

            pandas_warm_str = fmt_cell_warm("pandas")
            polars_warm_str = fmt_cell_warm("polars")
            booster_warm_str = fmt_cell_warm("booster")

            # First row (Cold)
            display_label = label if label != prev_label else ""
            display_groups = groups if groups != prev_groups else ""
            display_sort = sort_str if sort_str != prev_sort_str else ""

            lines.append(
                f"| {display_label} | {display_groups} | {display_sort} | Cold | "
                f"{pandas_cold_str} | {polars_cold_str} | {booster_cold_str} |"
            )

            # Second row (Warm) - always hide label, groups, sort
            lines.append(
                f"|  |  |  | Warm | {pandas_warm_str} | {polars_warm_str} | {booster_warm_str} |"
            )

            prev_label = label
            prev_groups = groups
            prev_sort_str = sort_str

    return "\n".join(lines)


def render_threshold_table(results: list[dict]) -> str:
    """Render threshold-neighborhood table with cold/warm stats.

    Args:
        results: List of benchmark results for threshold presets (can include both sorted
            and unsorted).

    Returns:
        Markdown table string.
    """
    if not results:
        return ""

    preset_order = ["threshold_180k", "threshold_200k", "threshold_220k"]
    preset_labels = {
        "threshold_180k": "2-key (~180k elems)",
        "threshold_200k": "2-key (~200k elems)",
        "threshold_220k": "2-key (~220k elems)",
    }

    results_by_preset_sort = {}
    for r in results:
        key = (r["preset"], r["sort"])
        results_by_preset_sort[key] = r

    lines = []
    lines.append("| Operation | Groups | Sort | Type | Pandas | Polars | Booster |")
    lines.append("|-----------|--------|------|------|--------|--------|---------|")

    prev_label = None
    prev_groups = None
    prev_sort_str = None

    for preset in preset_order:
        for sort_val in [True, False]:
            key = (preset, sort_val)
            if key not in results_by_preset_sort:
                continue

            r = results_by_preset_sort[key]
            label = preset_labels[preset]
            groups = f"{r['combo_cardinality']:,}"
            sort_str = "True" if sort_val else "False"
            backends = r["backends"]

            if "pandas" not in backends:
                continue

            pandas_cold: BenchmarkStats = backends["pandas"]["cold_stats"]
            pandas_warm: BenchmarkStats = backends["pandas"]["warm_stats"]
            pandas_cold_mean = pandas_cold.mean
            pandas_warm_mean = pandas_warm.mean

            def fmt_cell_cold(name, *, backends=backends, pandas_cold_mean=pandas_cold_mean):
                if name not in backends:
                    return "-"
                cold_stats: BenchmarkStats = backends[name]["cold_stats"]

                cold_mean_ms = cold_stats.mean * 1000
                cold_std_ms = cold_stats.std * 1000
                cold_speedup = pandas_cold_mean / cold_stats.mean if cold_stats.mean > 0 else 0

                cold_str = f"{cold_mean_ms:.1f}±{cold_std_ms:.1f}ms"

                if name == "pandas":
                    return f"{cold_str} (1.0x)"

                cold_speedup_str = (
                    f"**{cold_speedup:.1f}x**" if cold_speedup >= 1.1 else f"{cold_speedup:.1f}x"
                )
                return f"{cold_str} ({cold_speedup_str})"

            def fmt_cell_warm(name, *, backends=backends, pandas_warm_mean=pandas_warm_mean):
                if name not in backends:
                    return "-"
                warm_stats: BenchmarkStats = backends[name]["warm_stats"]

                warm_mean_ms = warm_stats.mean * 1000
                warm_std_ms = warm_stats.std * 1000
                warm_speedup = pandas_warm_mean / warm_stats.mean if warm_stats.mean > 0 else 0

                warm_str = f"{warm_mean_ms:.1f}±{warm_std_ms:.1f}ms"

                if name == "pandas":
                    return f"{warm_str} (1.0x)"

                warm_speedup_str = (
                    f"**{warm_speedup:.1f}x**" if warm_speedup >= 1.1 else f"{warm_speedup:.1f}x"
                )
                return f"{warm_str} ({warm_speedup_str})"

            pandas_cold_str = fmt_cell_cold("pandas")
            polars_cold_str = fmt_cell_cold("polars")
            booster_cold_str = fmt_cell_cold("booster")

            pandas_warm_str = fmt_cell_warm("pandas")
            polars_warm_str = fmt_cell_warm("polars")
            booster_warm_str = fmt_cell_warm("booster")

            display_label = label if label != prev_label else ""
            display_groups = groups if groups != prev_groups else ""
            display_sort = sort_str if sort_str != prev_sort_str else ""

            lines.append(
                f"| {display_label} | {display_groups} | {display_sort} | Cold | "
                f"{pandas_cold_str} | {polars_cold_str} | {booster_cold_str} |"
            )
            lines.append(
                f"|  |  |  | Warm | {pandas_warm_str} | {polars_warm_str} | {booster_warm_str} |"
            )

            prev_label = label
            prev_groups = groups
            prev_sort_str = sort_str

    return "\n".join(lines)


def format_performance_section(
    results: list[dict],
    sort_mode: str,
    cardinality: str,
    diagnostic: str,
) -> str:
    """Format benchmark results as README Performance section with cold/warm stats.

    Args:
        results: List of all benchmark result dictionaries.
        sort_mode: "all", "sorted", or "unsorted".
        cardinality: "all", "standard", or "high".
        diagnostic: "none" or "threshold".

    Returns:
        Markdown string with Performance section structure.
    """
    standard_presets = {"1key", "2key", "3key", "4key", "5key"}
    high_presets = {"high_cardinality_1key", "high_cardinality_2key", "high_cardinality_3key"}
    threshold_presets = {"threshold_180k", "threshold_200k", "threshold_220k"}

    filtered_results = []
    for r in results:
        if (
            sort_mode == "all"
            or (sort_mode == "sorted" and r["sort"])
            or (sort_mode == "unsorted" and not r["sort"])
        ):
            filtered_results.append(r)

    sections = ["## Performance", ""]
    aggs = []
    for result in filtered_results:
        agg = result["agg"]
        if agg not in aggs:
            aggs.append(agg)

    multiple_aggs = len(aggs) > 1

    for agg in aggs:
        agg_results = [r for r in filtered_results if r["agg"] == agg]
        standard_results = [r for r in agg_results if r["preset"] in standard_presets]
        high_results = [r for r in agg_results if r["preset"] in high_presets]
        threshold_results = [r for r in agg_results if r["preset"] in threshold_presets]

        if multiple_aggs:
            sections.append(f"### Aggregation: `{agg}`")
            sections.append("")

        if cardinality in ["all", "standard"] and standard_results:
            heading = (
                "### Standard Cardinality (5M rows)"
                if not multiple_aggs
                else "#### Standard Cardinality (5M rows)"
            )
            sections.append(heading)
            sections.append("")
            sections.append(render_standard_table(standard_results))
            sections.append("")

        if cardinality in ["all", "high"] and high_results:
            heading = (
                "### High Cardinality (5M rows, ~5M unique groups)"
                if not multiple_aggs
                else "#### High Cardinality (5M rows, ~5M unique groups)"
            )
            sections.append(heading)
            sections.append("")
            sections.append(render_high_table(high_results))
            sections.append("")

        if diagnostic == "threshold" and threshold_results:
            diag_heading = "### Diagnostics" if not multiple_aggs else "#### Diagnostics"
            threshold_heading = (
                "#### Threshold Neighborhood (2-key, n_groups * n_keys near 200k)"
                if not multiple_aggs
                else "##### Threshold Neighborhood (2-key, n_groups * n_keys near 200k)"
            )
            sections.append(diag_heading)
            sections.append("")
            sections.append(threshold_heading)
            sections.append("")
            sections.append(render_threshold_table(threshold_results))
            sections.append("")

    return "\n".join(sections)


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
                            correctness_str = (
                                f" | Correctness: cold={cold_corr}, warm={warm_corr}"
                            )

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


def save_results_md(
    results: list[dict],
    stats_evidence: list[dict[str, Any]],
    output_path: str,
    sort_mode: str,
    cardinality: str,
    diagnostic: str,
) -> None:
    """Save benchmark results to Markdown file.

    Args:
        results: List of benchmark result dictionaries.
        output_path: Output file path (should end with .md).
        sort_mode: Sort mode used in benchmark.
        cardinality: Cardinality mode used in benchmark.
        diagnostic: Diagnostic mode used in benchmark.
    """
    path = Path(output_path)

    if not path.suffix:
        path = path.with_suffix(".md")
    elif path.suffix != ".md":
        print(f"Warning: Output path should have .md extension, got {path.suffix}")

    path.parent.mkdir(parents=True, exist_ok=True)

    performance_section = format_performance_section(results, sort_mode, cardinality, diagnostic)
    stats_evidence_section = render_stats_evidence_section(stats_evidence)

    def correctness_section() -> str:
        lines: list[str] = []
        lines.append("### Correctness")
        lines.append("")

        def summarize_backend(name: str) -> str:
            failures: list[str] = []
            checked = 0
            passed = 0

            for r in results:
                backend_data = r.get("backends", {}).get(name)
                if not backend_data:
                    continue

                for mode_label in ("cold", "warm"):
                    status = backend_data.get(f"{mode_label}_correctness", "not_checked")
                    if status == "not_checked":
                        continue
                    checked += 1
                    if status.startswith("fail:"):
                        failures.append(f"{r['preset']} (sort={r['sort']}, {mode_label}): {status}")
                    else:
                        passed += 1

            if checked == 0:
                return "not_checked"
            if failures:
                first = failures[0]
                more = f" (+{len(failures) - 1} more)" if len(failures) > 1 else ""
                return f"fail ({passed}/{checked} passed): {first}{more}"
            return f"pass ({passed}/{checked})"

        for backend in BACKEND_DISPLAY_ORDER:
            if backend == "pandas":
                continue
            if backend == "polars" and not any("polars" in r.get("backends", {}) for r in results):
                continue
            label = backend.capitalize()
            lines.append(f"- {label}: {summarize_backend(backend)}")

        lines.append("")
        return "\n".join(lines)

    with open(path, "w") as f:
        f.write(performance_section)
        f.write("\n")
        f.write("\n")
        f.write(correctness_section())
        f.write("\n")
        f.write("\n")
        f.write(stats_evidence_section)
        f.write("\n")

    print(f"\nResults saved to: {path}")


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
  python benches/benchmark.py
      # Run default benchmarks (cardinality=all, diagnostic=none)
  python benches/benchmark.py --cardinality all
      # Run core benchmarks only (standard + high)
  python benches/benchmark.py --cardinality standard             # Standard only
  python benches/benchmark.py --cardinality high                 # High only
  python benches/benchmark.py --agg std --agg var                # Run only std/var benchmarks
  python benches/benchmark.py --agg prod                         # Run only product benchmarks
  python benches/benchmark.py --agg min --agg max               # Run only min/max benchmarks
  python benches/benchmark.py --diagnostic threshold --sort-mode unsorted
      # Add threshold diagnostics
  python benches/benchmark.py --cardinality all --diagnostic threshold --sort-mode unsorted
      # Core + diagnostics
  python benches/benchmark.py --sort-mode sorted                 # Sorted only
  python benches/benchmark.py --cardinality high --sort-mode unsorted  # Combine
  python benches/benchmark.py --output results.md                # Save results
  python benches/benchmark.py --samples 10                       # Adjust sample count

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
            "Defaults to current behavior: core=sum, evidence=std/var."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path for Markdown results (e.g., results.md)",
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

    stats_evidence: list[dict[str, Any]] = []
    if args.output or args.profile_json:
        stats_evidence = collect_stats_evidence(
            args.samples,
            args.cardinality,
            args.sort_mode,
            args.aggs,
        )

    if args.output:
        save_results_md(
            all_results,
            stats_evidence,
            args.output,
            args.sort_mode,
            args.cardinality,
            args.diagnostic,
        )

    if args.profile_json:
        save_profile_json(
            stats_evidence,
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
