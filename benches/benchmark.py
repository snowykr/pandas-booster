import time

import numpy as np
import pandas as pd


def benchmark_groupby():
    sizes = [1_000_000, 5_000_000, 10_000_000]
    n_groups_list = [100, 1000, 10000]

    print("=" * 80)
    print("Pandas Booster Benchmark: GroupBy Sum")
    print("=" * 80)

    import pandas_booster

    print(f"Thread count: {pd.DataFrame({'a': [1]}).booster.thread_count()}")
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

            pandas_times = []
            for _ in range(3):
                start = time.perf_counter()
                _ = df.groupby("key")["value"].sum()
                pandas_times.append(time.perf_counter() - start)
            pandas_time = min(pandas_times)

            booster_times = []
            for _ in range(3):
                start = time.perf_counter()
                _ = df.booster.groupby("key", "value", "sum")
                booster_times.append(time.perf_counter() - start)
            booster_time = min(booster_times)

            speedup = pandas_time / booster_time

            print(
                f"Rows: {n_rows:>12,} | Groups: {n_groups:>6,} | "
                f"Pandas: {pandas_time:.4f}s | Booster: {booster_time:.4f}s | "
                f"Speedup: {speedup:.2f}x"
            )

    print()


def benchmark_vs_polars():
    try:
        import polars as pl
    except ImportError:
        print("Polars not installed, skipping comparison")
        return

    import pandas_booster

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

    pandas_times = []
    for _ in range(3):
        start = time.perf_counter()
        _ = df_pandas.groupby("key")["value"].sum()
        pandas_times.append(time.perf_counter() - start)
    pandas_time = min(pandas_times)

    booster_times = []
    for _ in range(3):
        start = time.perf_counter()
        _ = df_pandas.booster.groupby("key", "value", "sum")
        booster_times.append(time.perf_counter() - start)
    booster_time = min(booster_times)

    polars_times = []
    for _ in range(3):
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


if __name__ == "__main__":
    benchmark_groupby()
    benchmark_vs_polars()
