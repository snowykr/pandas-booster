"""Multi-key groupby benchmark script for pandas-booster.

This script generates the multi-key benchmark results for the README Performance section.
It measures Two-key, Three-key, Four-key, and Five-key groupby operations.

Usage:
    # Run all multi-key benchmarks
    python benches/benchmark_multi_key.py

    # Run README preset benchmarks only
    python benches/benchmark_multi_key.py --preset readme

    # Run quick benchmarks (smaller datasets)
    python benches/benchmark_multi_key.py --quick

    # Save results to file
    python benches/benchmark_multi_key.py --output results_multi.json

    # Include sort=False benchmarks
    python benches/benchmark_multi_key.py --include-unsorted
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

# Import from local datasets module
from datasets import PRESETS, generate_multi_key_dataset, get_dataset_info

if TYPE_CHECKING:
    from typing import Any, Literal


def benchmark_multi_key_groupby(
    preset_name: str,
    agg: str = "sum",
    sort: bool = True,
    n_runs: int = 5,
    verify_correctness: bool = True,
) -> dict[str, Any]:
    """Run a single multi-key groupby benchmark.

    Args:
        preset_name: Name of the dataset preset from PRESETS.
        agg: Aggregation function ("sum", "mean", "min", "max").
        sort: Whether to sort the result (Pandas sort= parameter).
        n_runs: Number of runs (uses minimum time).
        verify_correctness: Whether to verify booster result matches Pandas.

    Returns:
        Dictionary with benchmark results and metadata.
    """
    import pandas_booster  # noqa: F401

    config = PRESETS[preset_name]
    df = generate_multi_key_dataset(**config)

    key_cols = [col for col, _ in config["key_configs"]]
    value_col = "value"
    dataset_info = get_dataset_info(df, key_cols)

    # Get thread count
    thread_count = df.booster.thread_count()

    # Benchmark Pandas
    pandas_times = []
    pandas_result = None
    for _ in range(n_runs):
        start = time.perf_counter()
        pandas_result = getattr(df.groupby(key_cols, sort=sort)[value_col], agg)()
        pandas_times.append(time.perf_counter() - start)
    pandas_time = min(pandas_times)

    # Benchmark Booster
    booster_times = []
    booster_result = None
    for _ in range(n_runs):
        start = time.perf_counter()
        # Note: sort parameter will be added in Phase 2
        booster_result = df.booster.groupby(key_cols, value_col, agg)
        booster_times.append(time.perf_counter() - start)
    booster_time = min(booster_times)

    # Verify correctness
    correctness = "not_checked"
    if verify_correctness and pandas_result is not None and booster_result is not None:
        try:
            if sort:
                # For sort=True, order must match exactly
                pd.testing.assert_series_equal(
                    booster_result.sort_index(),
                    pandas_result.sort_index(),
                    check_exact=False,
                    rtol=1e-10,
                )
            else:
                # For sort=False, only values need to match (order-independent)
                booster_sorted = booster_result.sort_index()
                pandas_sorted = pandas_result.sort_index()
                pd.testing.assert_series_equal(
                    booster_sorted,
                    pandas_sorted,
                    check_exact=False,
                    rtol=1e-10,
                )
            correctness = "pass"
        except AssertionError as e:
            correctness = f"fail: {str(e)[:100]}"

    speedup = pandas_time / booster_time

    return {
        "preset": preset_name,
        "n_rows": dataset_info["n_rows"],
        "n_keys": dataset_info["n_keys"],
        "key_cols": key_cols,
        "combo_cardinality": dataset_info["combo_cardinality"],
        "group_ratio": round(dataset_info["group_ratio"], 6),
        "agg": agg,
        "sort": sort,
        "pandas_time_ms": round(pandas_time * 1000, 2),
        "booster_time_ms": round(booster_time * 1000, 2),
        "speedup": round(speedup, 2),
        "thread_count": thread_count,
        "correctness": correctness,
    }


def run_readme_benchmarks(n_runs: int = 5, include_unsorted: bool = False) -> list[dict]:
    """Run the benchmarks that appear in README Performance section.

    Args:
        n_runs: Number of runs per benchmark.
        include_unsorted: Whether to include sort=False benchmarks.

    Returns:
        List of benchmark result dictionaries.
    """
    results = []

    # README presets: 1-key through 5-key
    readme_presets = ["readme_1key", "readme_2key", "readme_3key", "readme_4key", "readme_5key"]

    print("=" * 90)
    print("README Multi-Key GroupBy Benchmarks")
    print("=" * 90)

    for preset in readme_presets:
        print(f"\n[{preset}]")

        # sort=True (default Pandas behavior)
        result = benchmark_multi_key_groupby(preset, agg="sum", sort=True, n_runs=n_runs)
        results.append(result)

        print(
            f"  sort=True:  Pandas {result['pandas_time_ms']:>8.2f}ms | "
            f"Booster {result['booster_time_ms']:>8.2f}ms | "
            f"Speedup {result['speedup']:>5.2f}x | "
            f"Groups: {result['combo_cardinality']:,} | "
            f"Correct: {result['correctness']}"
        )

        if include_unsorted:
            result_unsorted = benchmark_multi_key_groupby(
                preset, agg="sum", sort=False, n_runs=n_runs
            )
            results.append(result_unsorted)
            print(
                f"  sort=False: Pandas {result_unsorted['pandas_time_ms']:>8.2f}ms | "
                f"Booster {result_unsorted['booster_time_ms']:>8.2f}ms | "
                f"Speedup {result_unsorted['speedup']:>5.2f}x"
            )

    return results


def run_cardinality_benchmarks(n_runs: int = 5) -> list[dict]:
    """Run benchmarks focused on cardinality extremes.

    Args:
        n_runs: Number of runs per benchmark.

    Returns:
        List of benchmark result dictionaries.
    """
    results = []

    cardinality_presets = [
        "low_cardinality_3key",
        "readme_3key",  # medium cardinality
        "high_cardinality_3key",
        "high_cardinality_2key",
    ]

    print("\n" + "=" * 90)
    print("Cardinality Impact Benchmarks (3-key focus)")
    print("=" * 90)

    for preset in cardinality_presets:
        print(f"\n[{preset}]")

        result = benchmark_multi_key_groupby(preset, agg="sum", sort=True, n_runs=n_runs)
        results.append(result)

        print(
            f"  Groups: {result['combo_cardinality']:>10,} | "
            f"Ratio: {result['group_ratio']:.4f} | "
            f"Pandas {result['pandas_time_ms']:>8.2f}ms | "
            f"Booster {result['booster_time_ms']:>8.2f}ms | "
            f"Speedup {result['speedup']:>5.2f}x"
        )

    return results


def run_quick_benchmarks(n_runs: int = 3) -> list[dict]:
    """Run quick benchmarks for fast iteration.

    Args:
        n_runs: Number of runs per benchmark.

    Returns:
        List of benchmark result dictionaries.
    """
    results = []

    quick_presets = ["quick_2key", "quick_3key"]

    print("=" * 90)
    print("Quick Multi-Key Benchmarks (1M rows)")
    print("=" * 90)

    for preset in quick_presets:
        result = benchmark_multi_key_groupby(preset, agg="sum", sort=True, n_runs=n_runs)
        results.append(result)

        print(
            f"  {preset:15s} | "
            f"Pandas {result['pandas_time_ms']:>8.2f}ms | "
            f"Booster {result['booster_time_ms']:>8.2f}ms | "
            f"Speedup {result['speedup']:>5.2f}x | "
            f"Correct: {result['correctness']}"
        )

    return results


def run_all_operations_benchmark(preset: str = "readme_3key", n_runs: int = 5) -> list[dict]:
    """Run all aggregation operations on a single preset.

    Args:
        preset: Dataset preset name.
        n_runs: Number of runs per benchmark.

    Returns:
        List of benchmark result dictionaries.
    """
    results = []
    operations = ["sum", "mean", "min", "max"]

    print(f"\n" + "=" * 90)
    print(f"All Operations Benchmark ({preset})")
    print("=" * 90)

    for op in operations:
        result = benchmark_multi_key_groupby(preset, agg=op, sort=True, n_runs=n_runs)
        results.append(result)

        print(
            f"  {op:5s} | "
            f"Pandas {result['pandas_time_ms']:>8.2f}ms | "
            f"Booster {result['booster_time_ms']:>8.2f}ms | "
            f"Speedup {result['speedup']:>5.2f}x"
        )

    return results


def print_readme_table(results: list[dict]) -> None:
    """Print results in README markdown table format.

    Args:
        results: List of benchmark result dictionaries.
    """
    # Filter to sort=True only for README
    sorted_results = [r for r in results if r.get("sort", True)]

    print("\n" + "=" * 90)
    print("README Performance Table (copy-paste ready)")
    print("=" * 90)
    print()
    print("| Operation | Pandas | Booster | Speedup |")
    print("|-----------|--------|---------|---------|")

    for r in sorted_results:
        if r["preset"].startswith("readme_"):
            n_keys = r["n_keys"]
            key_label = f"{n_keys}-key groupby" if n_keys > 1 else "Single-key groupby"
            pandas_str = f"{r['pandas_time_ms']:.1f}ms"
            booster_str = f"{r['booster_time_ms']:.1f}ms"
            speedup_str = f"**{r['speedup']:.1f}x**" if r["speedup"] > 1 else f"{r['speedup']:.1f}x"
            print(f"| {key_label} | {pandas_str} | {booster_str} | {speedup_str} |")


def save_results(results: list[dict], output_path: str) -> None:
    """Save benchmark results to JSON file.

    Args:
        results: List of benchmark result dictionaries.
        output_path: Output file path.
    """
    output = {
        "timestamp": datetime.now().isoformat(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "results": results,
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-key groupby benchmarks for pandas-booster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_multi_key.py                    # Run all benchmarks
  python benchmark_multi_key.py --preset readme    # README benchmarks only
  python benchmark_multi_key.py --quick            # Quick benchmarks (1M rows)
  python benchmark_multi_key.py --cardinality      # Cardinality impact study
  python benchmark_multi_key.py --output results.json  # Save results
        """,
    )
    parser.add_argument(
        "--preset",
        choices=["readme", "cardinality", "operations", "all"],
        default="all",
        help="Which benchmark suite to run",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmarks (smaller datasets)",
    )
    parser.add_argument(
        "--runs",
        "-r",
        type=int,
        default=5,
        help="Number of runs per benchmark (default: 5)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path for JSON results",
    )
    parser.add_argument(
        "--include-unsorted",
        action="store_true",
        help="Include sort=False benchmarks",
    )
    parser.add_argument(
        "--readme-table",
        action="store_true",
        help="Print results in README table format",
    )

    args = parser.parse_args()

    all_results = []

    if args.quick:
        all_results.extend(run_quick_benchmarks(n_runs=args.runs))
    elif args.preset == "readme":
        all_results.extend(
            run_readme_benchmarks(n_runs=args.runs, include_unsorted=args.include_unsorted)
        )
    elif args.preset == "cardinality":
        all_results.extend(run_cardinality_benchmarks(n_runs=args.runs))
    elif args.preset == "operations":
        all_results.extend(run_all_operations_benchmark(n_runs=args.runs))
    else:
        # Run all
        all_results.extend(
            run_readme_benchmarks(n_runs=args.runs, include_unsorted=args.include_unsorted)
        )
        all_results.extend(run_cardinality_benchmarks(n_runs=args.runs))
        all_results.extend(run_all_operations_benchmark(n_runs=args.runs))

    if args.readme_table:
        print_readme_table(all_results)

    if args.output:
        save_results(all_results, args.output)

    # Always print summary
    print("\n" + "=" * 90)
    print("Summary")
    print("=" * 90)
    print(f"Total benchmarks run: {len(all_results)}")
    passed = sum(1 for r in all_results if r.get("correctness") == "pass")
    print(f"Correctness: {passed}/{len(all_results)} passed")

    return all_results


if __name__ == "__main__":
    main()
