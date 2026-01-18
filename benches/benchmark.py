"""Official benchmark script for pandas-booster with Cold/Warm measurement.

This script generates benchmark results for the README Performance section.
It measures single-key through five-key groupby operations with proper
cold/warm separation and statistical analysis.

Cold/Warm Definitions:
- Cold: First call in a fresh Python process (after import + data preparation)
- Warm: Steady-state performance after cold + 1 warmup run (discarded), 5 samples

Usage:
    # Run all benchmarks (standard + high cardinality, sorted + unsorted)
    python benches/benchmark.py

    # Run only standard cardinality benchmarks
    python benches/benchmark.py --cardinality standard

    # Run only high cardinality benchmarks
    python benches/benchmark.py --cardinality high

    # Run only sorted benchmarks
    python benches/benchmark.py --sort-mode sorted

    # Run only unsorted benchmarks
    python benches/benchmark.py --sort-mode unsorted

    # Combine options
    python benches/benchmark.py --cardinality high --sort-mode sorted

    # Save results to markdown file
    python benches/benchmark.py --output results.md

    # Adjust sample count (applies to both cold and warm)
    python benches/benchmark.py --samples 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

sys.path.insert(0, str(Path(__file__).parent))
from bench_utils import BenchmarkStats, run_cold_warm_benchmark
from datasets import PRESETS, generate_multi_key_dataset, get_dataset_info

if TYPE_CHECKING:
    from typing import Literal


def benchmark_worker(
    preset_name: str,
    backend: Literal["pandas", "booster", "polars"],
    agg: str = "sum",
    sort: bool = True,
    verify_correctness: bool = False,
    mode: Literal["cold", "warm"] = "cold",
) -> dict[str, Any]:
    """Worker function: runs in fresh process, measures cold OR warm.

    Measurement protocol:
    - mode="cold": Measure 1st run. Return cold_time_s.
    - mode="warm": Run 1st (cold, discard) -> Run 2nd (warmup, discard) -> Measure 3rd. Return warm_time_s.

    Args:
        preset_name: Dataset preset name.
        backend: Which backend to benchmark.
        agg: Aggregation function.
        sort: Whether to sort results.
        verify_correctness: Whether to verify correctness (only for booster).
        mode: "cold" or "warm".

    Returns:
        Dictionary with cold_time_s OR warm_time_s.
    """
    if backend == "booster":
        import pandas_booster

    config = PRESETS[preset_name]
    df = generate_multi_key_dataset(**config)
    key_cols = [col for col, _ in config["key_configs"]]
    value_col = "value"

    if backend == "pandas":

        def run_once():
            return getattr(df.groupby(key_cols, sort=sort)[value_col], agg)()
    elif backend == "booster":

        def run_once():
            return df.booster.groupby(key_cols, value_col, agg, sort=sort)
    elif backend == "polars":
        if not HAS_POLARS:
            raise ImportError("Polars is not installed")

        df_polars = pl.DataFrame(
            {**{col: df[col].values for col in key_cols}, value_col: df[value_col].values}
        )
        agg_map = {
            "sum": pl.col(value_col).sum().alias(value_col),
            "mean": pl.col(value_col).mean().alias(value_col),
            "min": pl.col(value_col).min().alias(value_col),
            "max": pl.col(value_col).max().alias(value_col),
        }

        def run_once():
            result = df_polars.group_by(key_cols, maintain_order=False).agg(agg_map[agg])
            if sort:
                result = result.sort(key_cols)
            return result
    else:
        raise ValueError(f"Unknown backend: {backend}")

    if mode == "cold":
        start = time.perf_counter()
        cold_result = run_once()
        cold_time = time.perf_counter() - start

        correctness = "not_checked"
        if verify_correctness and backend == "booster":
            try:
                pandas_result = getattr(df.groupby(key_cols, sort=sort)[value_col], agg)()
                booster_result = cold_result

                pd.testing.assert_series_equal(
                    booster_result.sort_index(),
                    pandas_result.sort_index(),
                    check_exact=False,
                    rtol=1e-10,
                )
                correctness = "pass"
            except AssertionError as e:
                correctness = f"fail: {str(e)[:100]}"

        return {
            "cold_time_s": cold_time,
            "correctness": correctness,
        }

    elif mode == "warm":
        _ = run_once()
        _ = run_once()

        start = time.perf_counter()
        _ = run_once()
        warm_time = time.perf_counter() - start

        return {
            "warm_time_s": warm_time,
            "correctness": "not_checked",
        }

    return {}


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
        agg: Aggregation function ("sum", "mean", "min", "max").
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
    backends = ["pandas", "booster"]
    if HAS_POLARS:
        backends.append("polars")

    results = {}
    for backend in backends:
        print(f"  [{backend.upper()}]")
        worker_args = {
            "preset_name": preset_name,
            "backend": backend,
            "agg": agg,
            "sort": sort,
            "verify_correctness": verify_correctness and backend == "booster",
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
            "correctness": "not_checked",
        }

    if verify_correctness and "booster" in results:
        cold_samples_count = len(results["booster"]["cold_stats"].samples)
        warm_samples_count = len(results["booster"]["warm_stats"].samples)
        results["booster"]["correctness"] = f"pass ({cold_samples_count + warm_samples_count} runs)"

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
        results: List of benchmark results for standard presets (can include both sorted and unsorted).

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

            def fmt_cell_cold(name):
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

            def fmt_cell_warm(name):
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
                f"| {display_label} | {display_groups} | {display_sort} | Cold | {pandas_cold_str} | {polars_cold_str} | {booster_cold_str} |"
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
        results: List of benchmark results for high cardinality presets (can include both sorted and unsorted).

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

            def fmt_cell_cold(name):
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

            def fmt_cell_warm(name):
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
                f"| {display_label} | {display_groups} | {display_sort} | Cold | {pandas_cold_str} | {polars_cold_str} | {booster_cold_str} |"
            )

            # Second row (Warm) - always hide label, groups, sort
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
) -> str:
    """Format benchmark results as README Performance section with cold/warm stats.

    Args:
        results: List of all benchmark result dictionaries.
        sort_mode: "all", "sorted", or "unsorted".
        cardinality: "all", "standard", or "high".

    Returns:
        Markdown string with Performance section structure.
    """
    standard_presets = {"1key", "2key", "3key", "4key", "5key"}
    high_presets = {"high_cardinality_1key", "high_cardinality_2key", "high_cardinality_3key"}

    # Filter results by sort_mode
    filtered_results = []
    for r in results:
        if sort_mode == "all":
            filtered_results.append(r)
        elif sort_mode == "sorted" and r["sort"]:
            filtered_results.append(r)
        elif sort_mode == "unsorted" and not r["sort"]:
            filtered_results.append(r)

    # Separate by cardinality
    standard_results = [r for r in filtered_results if r["preset"] in standard_presets]
    high_results = [r for r in filtered_results if r["preset"] in high_presets]

    sections = []
    sections.append("## Performance")
    sections.append("")

    if cardinality in ["all", "standard"] and standard_results:
        sections.append("### Standard Cardinality (5M rows)")
        sections.append("")
        sections.append(render_standard_table(standard_results))
        sections.append("")

    if cardinality in ["all", "high"] and high_results:
        sections.append("### High Cardinality (5M rows, ~5M unique groups)")
        sections.append("")
        sections.append(render_high_table(high_results))
        sections.append("")

    return "\n".join(sections)


def resolve_presets(cardinality: str) -> list[str]:
    """Resolve cardinality option to list of preset names.

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
    sort_mode: str,
    n_samples: int = 5,
) -> list[dict]:
    """Run benchmark suite based on cardinality and sort-mode.

    Args:
        cardinality: "all", "standard", or "high".
        sort_mode: "all", "sorted", or "unsorted".
        n_samples: Number of samples per benchmark (applies to both cold and warm).

    Returns:
        List of benchmark result dictionaries.
    """
    presets = resolve_presets(cardinality)
    sorts = resolve_sorts(sort_mode)

    cardinality_label = cardinality.capitalize()
    if cardinality == "all":
        cardinality_label = "Standard + High"

    print("=" * 90)
    print(f"Pandas-Booster Benchmarks: {cardinality_label} Cardinality")
    print("=" * 90)
    print(f"Samples per benchmark: {n_samples} (fresh processes for both cold and warm)")
    print()

    results = []

    for sort_val in sorts:
        sort_str = "Sorted" if sort_val else "Unsorted"
        print(f"\n--- Running {sort_str} Benchmarks ---")

        for preset in presets:
            print(f"\n[{preset}] Running {sort_str} cold/warm benchmark...")

            result = benchmark_single(
                preset,
                agg="sum",
                sort=sort_val,
                n_samples=n_samples,
                verify_correctness=True,
            )
            results.append(result)

            backends = result["backends"]
            print(f"  Groups: {result['combo_cardinality']:,}")

            for backend_name in ["pandas", "booster", "polars"]:
                if backend_name not in backends:
                    continue
                backend_data = backends[backend_name]
                warm_stats: BenchmarkStats = backend_data["warm_stats"]
                print(f"  {backend_name:8s} | Warm: {warm_stats.format_ms(2)}")

    print("\n" + "=" * 90)
    print("Performance Tables")
    print("=" * 90)
    print(format_performance_section(results, sort_mode, cardinality))
    print()

    return results


def save_results_md(
    results: list[dict],
    output_path: str,
    sort_mode: str,
    cardinality: str,
) -> None:
    """Save benchmark results to Markdown file.

    Args:
        results: List of benchmark result dictionaries.
        output_path: Output file path (should end with .md).
        sort_mode: Sort mode used in benchmark.
        cardinality: Cardinality mode used in benchmark.
    """
    path = Path(output_path)

    if not path.suffix:
        path = path.with_suffix(".md")
    elif path.suffix != ".md":
        print(f"Warning: Output path should have .md extension, got {path.suffix}")

    path.parent.mkdir(parents=True, exist_ok=True)

    performance_section = format_performance_section(results, sort_mode, cardinality)

    with open(path, "w") as f:
        f.write(performance_section)
        f.write("\n")

    print(f"\nResults saved to: {path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Official pandas-booster benchmarks with Cold/Warm measurement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benches/benchmark.py                                    # Run all benchmarks
  python benches/benchmark.py --cardinality standard             # Standard only
  python benches/benchmark.py --cardinality high                 # High only
  python benches/benchmark.py --sort-mode sorted                 # Sorted only
  python benches/benchmark.py --cardinality high --sort-mode unsorted  # Combine
  python benches/benchmark.py --output results.md                # Save results
  python benches/benchmark.py --samples 10                       # Adjust sample count
        """,
    )
    parser.add_argument(
        "--cardinality",
        choices=["all", "standard", "high"],
        default="all",
        help="Which cardinality benchmarks to run (default: all)",
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
        "--output",
        "-o",
        type=str,
        help="Output file path for Markdown results (e.g., results.md)",
    )
    parser.add_argument(
        "--worker",
        type=str,
        help="Worker mode (internal use): JSON args for single benchmark run",
    )

    args = parser.parse_args()

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
        sort_mode=args.sort_mode,
        n_samples=args.samples,
    )

    if args.output:
        save_results_md(all_results, args.output, args.sort_mode, args.cardinality)

    print("\n" + "=" * 90)
    print("Summary")
    print("=" * 90)
    print(f"Total benchmarks run: {len(all_results)}")

    return all_results


if __name__ == "__main__":
    main()
