"""Benchmark script for pandas-booster with report saving capabilities.

Usage:
    python benches/benchmark.py                    # Run and print to console
    python benches/benchmark.py --output results   # Save to results.csv and results.json
    python benches/benchmark.py --output results --format csv  # Save only CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from typing import Literal


def benchmark_groupby(
    sizes: list[int] | None = None,
    n_groups_list: list[int] | None = None,
    n_runs: int = 3,
) -> list[dict]:
    """Run groupby benchmarks and return results.

    Args:
        sizes: List of row counts to test. Defaults to [1M, 5M, 10M].
        n_groups_list: List of group counts to test. Defaults to [100, 1000, 10000].
        n_runs: Number of runs per configuration (uses minimum time).

    Returns:
        List of result dictionaries with benchmark metrics.
    """
    if sizes is None:
        sizes = [1_000_000, 5_000_000, 10_000_000]
    if n_groups_list is None:
        n_groups_list = [100, 1000, 10000]

    import pandas_booster  # noqa: F401

    results = []
    thread_count = pd.DataFrame({"a": [1]}).booster.thread_count()

    print("=" * 80)
    print("Pandas Booster Benchmark: GroupBy Sum")
    print("=" * 80)
    print(f"Thread count: {thread_count}")
    print()

    for n_rows in sizes:
        for n_groups in n_groups_list:
            np.random.seed(42)
            df = pd.DataFrame(
                {
                    "key": np.random.randint(0, n_groups, size=n_rows),
                    "value": np.random.random(size=n_rows),
                }
            )

            # Benchmark Pandas
            pandas_times = []
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = df.groupby("key")["value"].sum()
                pandas_times.append(time.perf_counter() - start)
            pandas_time = min(pandas_times)

            # Benchmark Booster
            booster_times = []
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = df.booster.groupby("key", "value", "sum")
                booster_times.append(time.perf_counter() - start)
            booster_time = min(booster_times)

            speedup = pandas_time / booster_time

            result = {
                "rows": n_rows,
                "groups": n_groups,
                "pandas_time_s": round(pandas_time, 6),
                "booster_time_s": round(booster_time, 6),
                "speedup": round(speedup, 2),
                "thread_count": thread_count,
                "operation": "sum",
                "dtype": "f64",
            }
            results.append(result)

            print(
                f"Rows: {n_rows:>12,} | Groups: {n_groups:>6,} | "
                f"Pandas: {pandas_time:.4f}s | Booster: {booster_time:.4f}s | "
                f"Speedup: {speedup:.2f}x"
            )

    print()
    return results


def benchmark_vs_polars(n_runs: int = 3) -> list[dict]:
    """Run comparative benchmarks against Polars.

    Args:
        n_runs: Number of runs per configuration (uses minimum time).

    Returns:
        List of result dictionaries with comparative metrics.
    """
    try:
        import polars as pl
    except ImportError:
        print("Polars not installed, skipping comparison")
        return []

    import pandas_booster  # noqa: F401

    print("=" * 80)
    print("Comparison with Polars")
    print("=" * 80)

    n_rows = 10_000_000
    n_groups = 1000

    np.random.seed(42)
    keys = np.random.randint(0, n_groups, size=n_rows)
    values = np.random.random(size=n_rows)

    df_pandas = pd.DataFrame({"key": keys, "value": values})
    df_polars = pl.DataFrame({"key": keys, "value": values})

    # Benchmark Pandas
    pandas_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = df_pandas.groupby("key")["value"].sum()
        pandas_times.append(time.perf_counter() - start)
    pandas_time = min(pandas_times)

    # Benchmark Booster
    booster_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = df_pandas.booster.groupby("key", "value", "sum")
        booster_times.append(time.perf_counter() - start)
    booster_time = min(booster_times)

    # Benchmark Polars
    polars_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = df_polars.group_by("key").agg(pl.col("value").sum())
        polars_times.append(time.perf_counter() - start)
    polars_time = min(polars_times)

    print(f"Dataset: {n_rows:,} rows, {n_groups:,} groups")
    print(f"Pandas:         {pandas_time:.4f}s (baseline)")
    print(
        f"Pandas Booster: {booster_time:.4f}s ({pandas_time / booster_time:.2f}x faster than Pandas)"
    )
    print(
        f"Polars:         {polars_time:.4f}s ({pandas_time / polars_time:.2f}x faster than Pandas)"
    )
    print()

    return [
        {
            "rows": n_rows,
            "groups": n_groups,
            "library": "pandas",
            "time_s": round(pandas_time, 6),
            "speedup_vs_pandas": 1.0,
        },
        {
            "rows": n_rows,
            "groups": n_groups,
            "library": "pandas_booster",
            "time_s": round(booster_time, 6),
            "speedup_vs_pandas": round(pandas_time / booster_time, 2),
        },
        {
            "rows": n_rows,
            "groups": n_groups,
            "library": "polars",
            "time_s": round(polars_time, 6),
            "speedup_vs_pandas": round(pandas_time / polars_time, 2),
        },
    ]


def benchmark_all_operations(
    n_rows: int = 5_000_000, n_groups: int = 1000, n_runs: int = 3
) -> list[dict]:
    """Benchmark all supported aggregation operations.

    Args:
        n_rows: Number of rows in test dataset.
        n_groups: Number of unique groups.
        n_runs: Number of runs per configuration.

    Returns:
        List of result dictionaries for each operation.
    """
    import pandas_booster  # noqa: F401

    print("=" * 80)
    print("Benchmark: All Operations")
    print("=" * 80)

    np.random.seed(42)
    df = pd.DataFrame(
        {
            "key": np.random.randint(0, n_groups, size=n_rows),
            "val_float": np.random.random(size=n_rows) * 100,
            "val_int": np.random.randint(0, 1000, size=n_rows),
        }
    )

    operations = ["sum", "mean", "min", "max"]
    dtypes = [("val_float", "f64"), ("val_int", "i64")]
    results = []

    for target, dtype in dtypes:
        for op in operations:
            # Pandas
            pandas_times = []
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = getattr(df.groupby("key")[target], op)()
                pandas_times.append(time.perf_counter() - start)
            pandas_time = min(pandas_times)

            # Booster
            booster_times = []
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = df.booster.groupby("key", target, op)
                booster_times.append(time.perf_counter() - start)
            booster_time = min(booster_times)

            speedup = pandas_time / booster_time

            result = {
                "rows": n_rows,
                "groups": n_groups,
                "operation": op,
                "dtype": dtype,
                "pandas_time_s": round(pandas_time, 6),
                "booster_time_s": round(booster_time, 6),
                "speedup": round(speedup, 2),
            }
            results.append(result)

            print(
                f"{op:>5} ({dtype}): Pandas {pandas_time:.4f}s | Booster {booster_time:.4f}s | {speedup:.2f}x"
            )

    print()
    return results


def benchmark_sort_options(
    n_rows: int = 5_000_000, n_groups: int = 1000, n_runs: int = 3
) -> list[dict]:
    """Benchmark sort=True vs sort=False.

    Args:
        n_rows: Number of rows in test dataset.
        n_groups: Number of unique groups.
        n_runs: Number of runs per configuration.

    Returns:
        List of result dictionaries.
    """
    import pandas_booster  # noqa: F401

    print("=" * 80)
    print("Benchmark: Sort vs Unsorted (Pandas Booster)")
    print("=" * 80)

    np.random.seed(42)
    # Single key setup
    df = pd.DataFrame(
        {"key": np.random.randint(0, n_groups, size=n_rows), "val": np.random.random(size=n_rows)}
    )

    results = []

    # 1. Single Key
    print(f"Single Key (Rows: {n_rows:,}, Groups: {n_groups:,})")

    # Baseline: Pandas (always sorted by default, though sort=False is possible)
    pandas_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = df.groupby("key")["val"].sum()
        pandas_times.append(time.perf_counter() - start)
    pandas_time = min(pandas_times)
    print(f"  Pandas (default): {pandas_time:.4f}s")

    for sort_option in [True, False]:
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = df.booster.groupby("key", "val", "sum", sort=sort_option)
            times.append(time.perf_counter() - start)
        min_time = min(times)

        print(f"  Booster sort={str(sort_option):<5}: {min_time:.4f}s")
        results.append(
            {
                "type": "single_key",
                "rows": n_rows,
                "groups": n_groups,
                "sort": sort_option,
                "time_s": round(min_time, 6),
                "pandas_time_s": round(pandas_time, 6),
            }
        )

    # 2. Multi Key
    print(f"\nMulti Key (2 keys) (Rows: {n_rows:,}, Groups: {n_groups:,})")
    df["key2"] = np.random.randint(0, n_groups, size=n_rows)

    # Baseline: Pandas
    pandas_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = df.groupby(["key", "key2"])["val"].sum()
        pandas_times.append(time.perf_counter() - start)
    pandas_time = min(pandas_times)
    print(f"  Pandas (default): {pandas_time:.4f}s")

    for sort_option in [True, False]:
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = df.booster.groupby(["key", "key2"], "val", "sum", sort=sort_option)
            times.append(time.perf_counter() - start)
        min_time = min(times)

        print(f"  Booster sort={str(sort_option):<5}: {min_time:.4f}s")
        results.append(
            {
                "type": "multi_key",
                "rows": n_rows,
                "groups": n_groups,
                "sort": sort_option,
                "time_s": round(min_time, 6),
                "pandas_time_s": round(pandas_time, 6),
            }
        )

    print()
    return results


def save_results(
    results: dict,
    output_path: str | Path,
    formats: list[Literal["csv", "json"]] | None = None,
) -> list[str]:
    """Save benchmark results to files.

    Args:
        results: Dictionary containing benchmark results by category.
        output_path: Base path for output files (without extension).
        formats: List of formats to save. Defaults to ["csv", "json"].

    Returns:
        List of saved file paths.
    """
    if formats is None:
        formats = ["csv", "json"]

    output_base = Path(output_path)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    saved_files = []

    # Add metadata
    results["metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }

    if "json" in formats:
        json_path = output_base.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        saved_files.append(str(json_path))
        print(f"Saved JSON report: {json_path}")

    if "csv" in formats:
        # Save main groupby results as CSV
        if results.get("groupby"):
            csv_path = output_base.with_suffix(".csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results["groupby"][0].keys())
                writer.writeheader()
                writer.writerows(results["groupby"])
            saved_files.append(str(csv_path))
            print(f"Saved CSV report: {csv_path}")

        # Save operations benchmark
        if results.get("operations"):
            ops_csv_path = output_base.parent / f"{output_base.stem}_operations.csv"
            with open(ops_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results["operations"][0].keys())
                writer.writeheader()
                writer.writerows(results["operations"])
            saved_files.append(str(ops_csv_path))
            print(f"Saved operations CSV: {ops_csv_path}")

        # Save polars comparison
        if results.get("polars_comparison"):
            polars_csv_path = output_base.parent / f"{output_base.stem}_polars.csv"
            with open(polars_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results["polars_comparison"][0].keys())
                writer.writeheader()
                writer.writerows(results["polars_comparison"])
            saved_files.append(str(polars_csv_path))
            print(f"Saved Polars comparison CSV: {polars_csv_path}")

        # Save sort options benchmark
        if results.get("sort_options"):
            sort_csv_path = output_base.parent / f"{output_base.stem}_sort.csv"
            with open(sort_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results["sort_options"][0].keys())
                writer.writeheader()
                writer.writerows(results["sort_options"])
            saved_files.append(str(sort_csv_path))
            print(f"Saved sort options CSV: {sort_csv_path}")

    return saved_files


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run pandas-booster benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py                           # Run and print to console
  python benchmark.py --output results          # Save to results.csv and results.json
  python benchmark.py --output results --format csv  # Save only CSV
  python benchmark.py --quick                   # Run quick benchmark (smaller datasets)
        """,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Base path for output files (without extension)",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        nargs="+",
        choices=["csv", "json"],
        default=["csv", "json"],
        help="Output formats (default: csv json)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark with smaller datasets",
    )
    parser.add_argument(
        "--runs",
        "-r",
        type=int,
        default=3,
        help="Number of runs per configuration (default: 3)",
    )
    parser.add_argument(
        "--skip-polars",
        action="store_true",
        help="Skip Polars comparison benchmark",
    )
    parser.add_argument(
        "--skip-operations",
        action="store_true",
        help="Skip all-operations benchmark",
    )
    parser.add_argument(
        "--skip-sort",
        action="store_true",
        help="Skip sort options benchmark",
    )

    args = parser.parse_args()

    # Configure sizes based on --quick flag
    if args.quick:
        sizes = [500_000, 1_000_000]
        n_groups_list = [100, 1000]
        ops_rows = 1_000_000
    else:
        sizes = [1_000_000, 5_000_000, 10_000_000]
        n_groups_list = [100, 1000, 10000]
        ops_rows = 5_000_000

    # Run benchmarks
    results = {}

    results["groupby"] = benchmark_groupby(
        sizes=sizes,
        n_groups_list=n_groups_list,
        n_runs=args.runs,
    )

    if not args.skip_operations:
        results["operations"] = benchmark_all_operations(
            n_rows=ops_rows,
            n_groups=1000,
            n_runs=args.runs,
        )

    if not args.skip_sort:
        results["sort_options"] = benchmark_sort_options(
            n_rows=ops_rows,
            n_groups=1000,
            n_runs=args.runs,
        )

    if not args.skip_polars:
        results["polars_comparison"] = benchmark_vs_polars(n_runs=args.runs)

    # Save results if output path specified
    if args.output:
        save_results(results, args.output, formats=args.format)

    return results


if __name__ == "__main__":
    main()
