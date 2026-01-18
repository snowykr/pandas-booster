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
To build and install from source, it is highly recommended to use a virtual environment:

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
- **Key column(s)**: Must be integer dtype (e.g., `int64`, `int32`). For multi-column groupby, all key columns must be integers.
- **Maximum key columns**: Up to 10 columns for multi-column groupby.
- **Value column**: Must be a numeric dtype (integers or floats).
- **Nullable types**: Nullable extension arrays (e.g., `Int64`, `Float64` using `pd.NA`) are not supported and will trigger a fallback to Pandas.
- **NaN handling**: `NaN` values in the target column are skipped in aggregations, matching standard Pandas behavior.
- **Return types**: Integer aggregations (like `sum` on `int64`) return `float64` to match Pandas' behavior regarding potential overflows and consistency.

## Performance

The library is designed for large datasets where multi-core parallelism can be fully utilized. Single-key groupby uses Rayon's parallel map-reduce, while multi-key operations use a radix-partitioning algorithm that eliminates merge overhead.

**Benchmark methodology:**
- **Process Isolation:** Benchmarks use rigorous process isolation to ensure accurate results.
  - **Cold:** Average of 5 fresh process executions (first run only).
  - **Warm:** Average of 5 fresh process executions (steady state after warmup).
- **Polars sort handling:** Polars does not have a `sort` parameter in `group_by`. For fair comparison, we define `sort=True` as "groupby+agg followed by sorting the result by keys" (cost included in timing), and `sort=False` as "groupby+agg only" (no sorting cost). This ensures all three engines (Pandas, Polars, Booster) are measured under identical conditions.
- **Speedup baseline:** All speedup values (`x`) use **Pandas** as the baseline (1.0x) within each sort mode.
- **Optional Polars:** Polars is included in the benchmarks for comparison if installed. If not installed, the benchmark suite proceeds with Pandas vs Booster only.

### Standard Cardinality (5M rows)

| Operation | Groups | Sort | Type | Pandas | Polars | Booster |
|-----------|--------|------|------|--------|--------|---------|
| Single-key | 1,000 | True | Cold | 39.0±12.4ms (1.0x) | 35.2±12.9ms (**1.1x**) | 9.2±1.4ms (**4.3x**) |
|  |  |  | Warm | 32.5±8.5ms (1.0x) | 21.3±3.3ms (**1.5x**) | 2.9±0.3ms (**11.1x**) |
|  |  | False | Cold | 27.3±1.1ms (1.0x) | 22.3±1.6ms (**1.2x**) | 5.5±1.0ms (**4.9x**) |
|  |  |  | Warm | 23.6±0.7ms (1.0x) | 19.4±2.1ms (**1.2x**) | 2.5±0.2ms (**9.6x**) |
| 2-key | 5,000 | True | Cold | 99.3±7.0ms (1.0x) | 54.2±13.5ms (**1.8x**) | 97.2±2.9ms (1.0x) |
|  |  |  | Warm | 71.0±1.2ms (1.0x) | 44.1±6.0ms (**1.6x**) | 97.4±7.3ms (0.7x) |
|  |  | False | Cold | 69.8±4.4ms (1.0x) | 60.5±18.4ms (**1.2x**) | 100.0±2.5ms (0.7x) |
|  |  |  | Warm | 56.6±3.0ms (1.0x) | 46.6±4.1ms (**1.2x**) | 98.2±3.5ms (0.6x) |
| 3-key | 25,000 | True | Cold | 130.4±10.1ms (1.0x) | 70.8±19.3ms (**1.8x**) | 108.3±4.1ms (**1.2x**) |
|  |  |  | Warm | 112.0±2.6ms (1.0x) | 57.8±2.8ms (**1.9x**) | 103.8±5.1ms (1.1x) |
|  |  | False | Cold | 113.6±8.7ms (1.0x) | 78.4±18.7ms (**1.4x**) | 124.2±27.4ms (0.9x) |
|  |  |  | Warm | 99.8±9.2ms (1.0x) | 65.0±13.0ms (**1.5x**) | 113.4±15.6ms (0.9x) |
| 4-key | 100,000 | True | Cold | 163.4±17.7ms (1.0x) | 118.5±11.4ms (**1.4x**) | 129.2±11.7ms (**1.3x**) |
|  |  |  | Warm | 144.6±12.6ms (1.0x) | 107.4±10.6ms (**1.3x**) | 119.0±8.4ms (**1.2x**) |
|  |  | False | Cold | 137.3±7.6ms (1.0x) | 90.4±6.9ms (**1.5x**) | 117.3±6.4ms (**1.2x**) |
|  |  |  | Warm | 130.4±34.5ms (1.0x) | 80.6±1.6ms (**1.6x**) | 107.3±2.1ms (**1.2x**) |
| 5-key | 993,138 | True | Cold | 298.9±16.7ms (1.0x) | 324.4±59.0ms (0.9x) | 290.9±49.2ms (1.0x) |
|  |  |  | Warm | 277.3±10.5ms (1.0x) | 206.5±31.8ms (**1.3x**) | 292.8±32.9ms (0.9x) |
|  |  | False | Cold | 191.4±21.4ms (1.0x) | 165.8±53.4ms (**1.2x**) | 222.7±2.2ms (0.9x) |
|  |  |  | Warm | 168.4±16.3ms (1.0x) | 137.8±10.0ms (**1.2x**) | 209.9±2.7ms (0.8x) |

### High Cardinality (5M rows, ~5M unique groups)

| Operation | Groups | Sort | Type | Pandas | Polars | Booster |
|-----------|--------|------|------|--------|--------|---------|
| Single-key | 3,160,983 | True | Cold | 769.3±7.0ms (1.0x) | 136.2±2.6ms (**5.6x**) | 1441.2±212.6ms (0.5x) |
|  |  |  | Warm | 778.4±38.2ms (1.0x) | 136.9±9.9ms (**5.7x**) | 1281.9±14.6ms (0.6x) |
|  |  | False | Cold | 225.2±3.6ms (1.0x) | 80.7±2.1ms (**2.8x**) | 989.8±30.1ms (0.2x) |
|  |  |  | Warm | 224.1±7.1ms (1.0x) | 87.2±13.8ms (**2.6x**) | 933.0±32.1ms (0.2x) |
| 2-key | 4,532,339 | True | Cold | 954.1±26.2ms (1.0x) | 270.7±19.9ms (**3.5x**) | 279.3±5.5ms (**3.4x**) |
|  |  |  | Warm | 897.0±4.7ms (1.0x) | 265.7±16.0ms (**3.4x**) | 244.1±6.0ms (**3.7x**) |
|  |  | False | Cold | 391.9±20.3ms (1.0x) | 120.0±4.7ms (**3.3x**) | 172.7±14.5ms (**2.3x**) |
|  |  |  | Warm | 357.2±13.5ms (1.0x) | 121.1±2.6ms (**2.9x**) | 187.7±41.6ms (**1.9x**) |
| 3-key | 4,901,309 | True | Cold | 1013.3±23.4ms (1.0x) | 358.4±8.0ms (**2.8x**) | 398.5±12.8ms (**2.5x**) |
|  |  |  | Warm | 998.9±32.4ms (1.0x) | 371.8±31.2ms (**2.7x**) | 370.9±31.1ms (**2.7x**) |
|  |  | False | Cold | 487.2±98.5ms (1.0x) | 133.7±32.5ms (**3.6x**) | 216.6±18.7ms (**2.2x**) |
|  |  |  | Warm | 391.5±18.0ms (1.0x) | 128.7±14.8ms (**3.0x**) | 184.7±8.5ms (**2.1x**) |

**Performance characteristics:
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

### Running Benchmarks

```bash
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
```

## Development

### Building
Build the extension module in-place:
```bash
maturin develop
```

For release builds with optimizations:
```bash
maturin develop --release
```

### Testing
Run the test suite using `pytest`:
```bash
pytest tests/
```

### Benchmarking

#### Setup for Full Benchmarks (including Polars comparison)
To run benchmarks with library comparisons (Polars, etc.), install the optional benchmark dependencies:

```bash
# Install benchmark and development dependencies
pip install -e ".[bench,dev]"

# Build the Rust extension
maturin develop --release
```

This installs:
- `polars`: For performance comparison benchmarks
- `matplotlib`: For generating plots and visualizations (optional)
- `pytest` and `pytest-benchmark`: For test framework and benchmarking

#### Running Benchmarks
```bash
# Run all benchmarks (standard + high cardinality, sorted + unsorted)
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

**Note**: The benchmark scripts use rigorous process isolation (fresh process per sample) for both Cold and Warm measurements to ensure accurate results. Polars is required for benchmarks and will be used for comparison.

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
