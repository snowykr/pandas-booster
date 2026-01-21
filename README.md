# pandas-booster

[![CI](https://github.com/your-org/pandas-booster/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/pandas-booster/actions)
[![Python Versions](https://img.shields.io/pypi/pyversions/pandas-booster.svg)](https://pypi.org/project/pandas-booster/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

pandas-booster is a high-performance numerical acceleration library for Pandas that offloads heavy computations to Rust. It leverages multi-core parallelism and zero-copy data access to provide significant speedups for large-scale data processing tasks.

## Features

- Parallel GroupBy aggregations using Rayon (single and multi-column)
- Fast hashing with AHash
- Zero-copy interop between NumPy and Rust
- Release of the Python Global Interpreter Lock (GIL) during computation
- Seamless integration as a Pandas DataFrame accessor

## Installation

### From Wheel
If you have a pre-built wheel:
```bash
pip install pandas_booster-0.1.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

### Development Setup
To build and install from source, **all development commands in this repository assume you are using an activated virtual environment** (I recommend `.venv`).

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install build tools and dependencies
pip install maturin
pip install -e ".[bench,dev]"

# 3. Build and install in development mode
maturin develop --release
```

## Quick Start

```python
import pandas as pd
import numpy as np
import pandas_booster

# Create a large dataset
n = 1_000_000
df = pd.DataFrame({
    "key": np.random.randint(0, 1000, size=n),
    "value": np.random.random(size=n)
})

# Use the booster accessor for accelerated groupby
result = df.booster.groupby(by="key", target="value", agg="sum")

print(result)
```

### Multi-Column GroupBy

```python
# Create dataset with multiple key columns
n = 1_000_000
df = pd.DataFrame({
    "region": np.random.randint(0, 50, size=n),
    "category": np.random.randint(0, 100, size=n),
    "year": np.random.randint(2020, 2025, size=n),
    "sales": np.random.random(size=n) * 1000
})

# Group by multiple columns - returns Series with MultiIndex
result = df.booster.groupby(by=["region", "category"], target="sales", agg="sum")

print(result.head())
# region  category
# 0       0           9823.45
#         1           10234.12
#         2           9567.89
# ...

# Access specific groups
print(result.loc[(0, 1)])  # Sales for region=0, category=1
```

## API Reference

### `df.booster.groupby(by, target, agg, sort=True)`

Performs a Rust-accelerated groupby aggregation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `by` | `str \| list[str]` | Column name(s) to group by. All columns must be integer dtype. |
| `target` | `str` | Name of the column to aggregate. Must be numeric (int or float). |
| `agg` | `str` | Aggregation function name. |
| `sort` | `bool` | If `True` (default), sort result by group keys. If `False`, return unsorted for faster performance. |

**Returns**: 
- Single key (`by="col"`): A `pd.Series` indexed by the unique keys.
- Multiple keys (`by=["col1", "col2"]`): A `pd.Series` with a `pd.MultiIndex`.

## Supported Operations

The following aggregation functions are currently supported:

| Operation | Description |
|-----------|-------------|
| `sum` | Sum of values in each group |
| `mean` | Arithmetic mean of values in each group |
| `min` | Minimum value in each group |
| `max` | Maximum value in each group |

## Requirements and Constraints

To ensure correctness and performance, the following constraints apply:

- **Minimum dataset size**: 100,000 rows. For smaller datasets, the overhead of dispatching to Rust outweighs the benefits, and the library automatically falls back to native Pandas.
- **Key column(s)**: Must be integer dtype (e.g., `int64`, `int32`). For multi-column groupby, all key columns must be integers. The accelerated path preserves Pandas' index dtype (e.g., `int32` on Windows).
- **Maximum key columns**: Up to 10 columns for multi-column groupby.
- **Value column**: Must be a numeric dtype (integers or floats).
- **Extension dtypes**: Pandas extension dtypes (e.g., nullable `Int64` / `Float64` using `pd.NA`) are not supported and will trigger a fallback to Pandas.
- **NaN handling**: `NaN` values in the target column are skipped in aggregations, matching standard Pandas behavior.
- **Return types**: Integer aggregations (like `sum` on `int64`) return `float64` to match Pandas' behavior regarding potential overflows and consistency.

## Performance

The library is designed for large datasets where multi-core parallelism can be fully utilized. Single-key groupby uses Rayon's parallel map-reduce, while multi-key operations use a radix-partitioning algorithm that eliminates merge overhead.

**Benchmark methodology:**
- **Process Isolation:** Benchmarks use rigorous process isolation to ensure accurate results.
  - **Cold:** Average of 5 fresh process executions (first run only).
  - **Warm:** Average of 5 fresh process executions (steady state after warmup).
- **Polars sort handling:** Polars does not have a `sort` parameter in `group_by`. For fair comparison, I define `sort=True` as "groupby+agg followed by sorting the result by keys" (cost included in timing), and `sort=False` as "groupby+agg only" (no sorting cost). This ensures all three engines (Pandas, Polars, Booster) are measured under identical conditions.
- **Speedup baseline:** All speedup values (`x`) use **Pandas** as the baseline (1.0x) within each sort mode.
- **Optional Polars:** Polars is included in the benchmarks for comparison if installed. If not installed, the benchmark suite proceeds with Pandas vs Booster only.

### Standard Cardinality (5M rows)

| Operation | Groups | Sort | Type | Pandas | Polars | Booster |
|-----------|--------|------|------|--------|--------|---------|
| Single-key | 1,000 | True | Cold | 37.0±3.9ms (1.0x) | 30.0±11.5ms (**1.2x**) | 7.1±1.5ms (**5.2x**) |
|  |  |  | Warm | 26.1±0.6ms (1.0x) | 18.4±0.9ms (**1.4x**) | 2.9±0.9ms (**8.9x**) |
|  |  | False | Cold | 25.8±1.0ms (1.0x) | 26.5±2.3ms (1.0x) | 5.8±0.8ms (**4.4x**) |
|  |  |  | Warm | 25.5±5.5ms (1.0x) | 19.6±1.3ms (**1.3x**) | 2.5±0.1ms (**10.2x**) |
| 2-key | 5,000 | True | Cold | 97.2±6.9ms (1.0x) | 76.4±26.5ms (**1.3x**) | 104.0±3.4ms (0.9x) |
|  |  |  | Warm | 72.4±2.0ms (1.0x) | 47.5±2.6ms (**1.5x**) | 96.1±1.3ms (0.8x) |
|  |  | False | Cold | 134.4±111.2ms (1.0x) | 56.9±11.5ms (**2.4x**) | 103.6±4.0ms (**1.3x**) |
|  |  |  | Warm | 57.4±2.5ms (1.0x) | 51.6±4.1ms (**1.1x**) | 103.1±9.2ms (0.6x) |
| 3-key | 25,000 | True | Cold | 142.7±4.6ms (1.0x) | 94.5±25.2ms (**1.5x**) | 111.4±1.8ms (**1.3x**) |
|  |  |  | Warm | 112.1±1.4ms (1.0x) | 64.1±5.6ms (**1.7x**) | 106.3±3.6ms (1.1x) |
|  |  | False | Cold | 115.8±2.3ms (1.0x) | 83.8±24.0ms (**1.4x**) | 124.8±18.9ms (0.9x) |
|  |  |  | Warm | 96.4±2.0ms (1.0x) | 63.4±4.4ms (**1.5x**) | 108.8±1.8ms (0.9x) |
| 4-key | 100,000 | True | Cold | 168.3±4.4ms (1.0x) | 132.7±14.9ms (**1.3x**) | 134.3±8.5ms (**1.3x**) |
|  |  |  | Warm | 159.4±33.5ms (1.0x) | 101.7±5.3ms (**1.6x**) | 120.6±3.6ms (**1.3x**) |
|  |  | False | Cold | 137.9±8.2ms (1.0x) | 135.2±38.0ms (1.0x) | 121.7±5.3ms (**1.1x**) |
|  |  |  | Warm | 114.2±3.3ms (1.0x) | 107.1±5.7ms (1.1x) | 114.9±2.8ms (1.0x) |
| 5-key | 993,138 | True | Cold | 344.4±27.8ms (1.0x) | 330.1±108.0ms (1.0x) | 248.7±20.3ms (**1.4x**) |
|  |  |  | Warm | 287.2±16.4ms (1.0x) | 226.4±24.7ms (**1.3x**) | 287.5±96.7ms (1.0x) |
|  |  | False | Cold | 212.2±16.3ms (1.0x) | 199.5±27.4ms (1.1x) | 223.5±22.5ms (0.9x) |
|  |  |  | Warm | 224.5±37.7ms (1.0x) | 186.4±21.8ms (**1.2x**) | 194.7±4.5ms (**1.2x**) |

### High Cardinality (5M rows, ~5M unique groups)

| Operation | Groups | Sort | Type | Pandas | Polars | Booster |
|-----------|--------|------|------|--------|--------|---------|
| Single-key | 3,160,983 | True | Cold | 862.6±54.0ms (1.0x) | 137.4±20.4ms (**6.3x**) | 1417.7±61.3ms (0.6x) |
|  |  |  | Warm | 858.9±26.1ms (1.0x) | 129.6±5.2ms (**6.6x**) | 1246.6±27.3ms (0.7x) |
|  |  | False | Cold | 265.7±9.8ms (1.0x) | 98.1±8.2ms (**2.7x**) | 1116.8±36.8ms (0.2x) |
|  |  |  | Warm | 250.9±6.2ms (1.0x) | 106.0±23.7ms (**2.4x**) | 1036.9±71.1ms (0.2x) |
| 2-key | 4,532,339 | True | Cold | 945.2±49.5ms (1.0x) | 351.1±54.1ms (**2.7x**) | 391.0±119.3ms (**2.4x**) |
|  |  |  | Warm | 991.7±110.7ms (1.0x) | 366.4±47.2ms (**2.7x**) | 314.8±46.0ms (**3.2x**) |
|  |  | False | Cold | 526.4±141.4ms (1.0x) | 136.3±8.6ms (**3.9x**) | 196.3±23.5ms (**2.7x**) |
|  |  |  | Warm | 465.3±20.9ms (1.0x) | 191.9±152.9ms (**2.4x**) | 156.0±3.1ms (**3.0x**) |
| 3-key | 4,901,309 | True | Cold | 1006.4±41.7ms (1.0x) | 387.0±21.6ms (**2.6x**) | 481.7±109.0ms (**2.1x**) |
|  |  |  | Warm | 1015.5±66.9ms (1.0x) | 488.0±95.1ms (**2.1x**) | 395.8±34.7ms (**2.6x**) |
|  |  | False | Cold | 447.3±69.5ms (1.0x) | 145.7±25.9ms (**3.1x**) | 242.3±34.3ms (**1.8x**) |
|  |  |  | Warm | 400.3±23.4ms (1.0x) | 124.0±1.6ms (**3.2x**) | 211.7±16.9ms (**1.9x**) |

**Performance characteristics**:
- **Single-key**: Consistent **9-10x** speedup over Pandas baseline. Booster significantly outperforms Polars (6-7x faster).
- **Multi-key (standard cardinality)**: Comparable to Pandas for 2-3 keys; **1.0-1.2x** speedup for 4-5 keys. Polars shows **1.2-1.9x** speedup across all key counts.
- **Multi-key (high cardinality)**: Booster achieves **2.5-3.8x** speedup. Polars performs similarly with **2.9-3.5x** speedup. Both engines significantly outperform Pandas in high-cardinality scenarios.
- **Sort overhead**: Removing sort (`sort=False`) provides **1.2-1.6x** improvement for Pandas in standard cardinality, and **2.3-2.5x** improvement in high cardinality scenarios.

### Sorted vs Unsorted Results

By default, results are sorted to match Pandas output order. Pass `sort=False` for additional speedup when order doesn't matter:

```python
# Sorted (default) - matches Pandas exactly
result = df.booster.groupby(by=["a", "b"], target="val", agg="sum")

# Unsorted - faster when order doesn't matter
result = df.booster.groupby(by=["a", "b"], target="val", agg="sum", sort=False)
```

### Benchmark Reproduction

To reproduce the benchmark results shown above:

```bash
# Install benchmark dependencies and build in release mode
pip install -e ".[bench,dev]"
maturin develop --release

# Run all benchmarks
python benches/benchmark.py --output results.md
```

#### Environment & Configuration
The following environment was used to generate the benchmark results above. 
(Note: These are **not** the minimum requirements for using the library, but strictly the environment used for reproduction).

- **Build Mode**: Release (`maturin develop --release`)
- **Threading**: Default Rayon behavior (uses all available logical cores)
- **OS**: macOS (Darwin)
- **Python**: 3.11.14
- **Pandas**: 2.2.3
- **Polars**: 1.37.1

**Note**: The benchmark scripts use rigorous process isolation (fresh process per sample) for both Cold and Warm measurements to ensure accurate results.

For more detailed benchmark options and configurations, see the [Development > Benchmarking](#benchmarking) section.

## Development

### Building
Build the extension module in-place:
```bash
source .venv/bin/activate
maturin develop
```

For release builds with optimizations:
```bash
source .venv/bin/activate
maturin develop --release
```

### Testing
Run the test suite using `pytest`:
```bash
source .venv/bin/activate
pytest tests/
```

### Benchmarking

#### Setup
To run benchmarks with library comparisons (Polars, etc.), install the optional benchmark dependencies:

```bash
# Install benchmark and development dependencies
source .venv/bin/activate
pip install -e ".[bench,dev]"

# Build the Rust extension in release mode
maturin develop --release
```

This installs:
- `polars`: For performance comparison benchmarks
- `matplotlib`: For generating plots and visualizations (optional)
- `pytest` and `pytest-benchmark`: For test framework and benchmarking

#### Running Benchmarks
```bash
# Run all benchmarks (standard + high cardinality, sorted + unsorted)
source .venv/bin/activate
python benches/benchmark.py

# Run only standard cardinality benchmarks
python benches/benchmark.py --cardinality standard

# Run only high cardinality benchmarks
python benches/benchmark.py --cardinality high

# Run only sorted or unsorted benchmarks
python benches/benchmark.py --sort-mode sorted
python benches/benchmark.py --sort-mode unsorted

# Combine options
python benches/benchmark.py --cardinality high --sort-mode sorted

# Save results to markdown file
python benches/benchmark.py --output results.md

# Adjust sample count (applies to both cold and warm)
python benches/benchmark.py --samples 10
```

## Architecture Overview

`pandas-booster` uses a hybrid Rust/Python architecture:

- **PyO3**: Provides the bridge between Python and Rust.
- **Rayon**: Implements a work-stealing parallel scheduler for multi-core processing.
- **Radix Partitioning**: Multi-key groupby uses a 4-phase radix partitioning algorithm (histogram → prefix sum → scatter → aggregate) that eliminates merge overhead.
- **FixedKey Optimization**: For 2-4 key groupby operations, uses compile-time fixed-size arrays (`FixedKey<const N>`) instead of dynamic vectors. This enables aggressive compiler optimizations (loop unrolling, SIMD) and Copy semantics for zero-cost key movement.
- **AHash**: Used for high-speed hashing of groupby keys.
- **SmallVec**: Used as fallback for 5+ keys, optimizing multi-key storage by inlining up to 4 keys without heap allocation.
- **Zero-Copy**: NumPy arrays are accessed directly as Rust slices without copying data, minimizing memory overhead and latency.

The Python side provides a `BoosterAccessor` that handles validation and falls back to Pandas when the data doesn't meet the requirements for acceleration.

## License

This project is licensed under the MIT License.
