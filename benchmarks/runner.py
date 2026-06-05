"""Benchmark suite runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bench_utils import run_cold_warm_benchmark
from datasets import PRESETS, generate_multi_key_dataset, get_dataset_info
from dispatch import HAS_POLARS
from reporting import (
    BACKEND_DISPLAY_ORDER,
    CORE_BENCHMARK_AGG,
    STATS_EVIDENCE_AGGS,
    format_performance_section,
)
from runner_worker import benchmark_worker as benchmark_worker

BENCHMARKS_DIR = Path(__file__).resolve().parent

_BENCHMARK_WORKER_TYPE_SURFACE = (
    'agg: Literal["sum", "mean", "prod", "median", "std", "var", "min", "max", "count"]'
)


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

    script_path = BENCHMARKS_DIR / "benchmark.py"
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
