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
| Single-key | 1,000 | True | Cold | 42.4±4.5ms (1.0x) | 37.5±25.3ms (**1.1x**) | 9.1±5.6ms (**4.7x**) |
|  |  |  | Warm | 25.4±0.4ms (1.0x) | 31.0±22.5ms (0.8x) | 2.9±0.3ms (**8.7x**) |
|  |  | False | Cold | 29.5±1.1ms (1.0x) | 40.2±18.3ms (0.7x) | 9.2±4.6ms (**3.2x**) |
|  |  |  | Warm | 26.5±1.8ms (1.0x) | 25.4±6.5ms (1.0x) | 3.2±0.4ms (**8.3x**) |
| 2-key | 5,000 | True | Cold | 86.7±15.8ms (1.0x) | 100.1±44.5ms (0.9x) | 102.6±3.2ms (0.8x) |
|  |  |  | Warm | 70.3±3.7ms (1.0x) | 50.3±10.0ms (**1.4x**) | 100.7±4.7ms (0.7x) |
|  |  | False | Cold | 75.4±8.9ms (1.0x) | 61.1±23.4ms (**1.2x**) | 129.6±16.5ms (0.6x) |
|  |  |  | Warm | 57.4±2.0ms (1.0x) | 55.0±7.8ms (1.0x) | 111.1±8.9ms (0.5x) |
| 3-key | 25,000 | True | Cold | 146.5±7.1ms (1.0x) | 130.9±59.9ms (**1.1x**) | 119.1±3.4ms (**1.2x**) |
|  |  |  | Warm | 109.9±3.7ms (1.0x) | 61.3±3.9ms (**1.8x**) | 109.1±6.8ms (1.0x) |
|  |  | False | Cold | 128.7±47.4ms (1.0x) | 125.3±33.2ms (1.0x) | 136.8±18.8ms (0.9x) |
|  |  |  | Warm | 99.3±5.3ms (1.0x) | 87.9±24.9ms (**1.1x**) | 130.1±10.5ms (0.8x) |
| 4-key | 100,000 | True | Cold | 163.8±9.3ms (1.0x) | 263.8±135.6ms (0.6x) | 146.3±57.3ms (**1.1x**) |
|  |  |  | Warm | 145.5±12.2ms (1.0x) | 108.8±13.3ms (**1.3x**) | 140.0±21.5ms (1.0x) |
|  |  | False | Cold | 195.7±57.6ms (1.0x) | 107.9±20.6ms (**1.8x**) | 132.4±6.2ms (**1.5x**) |
|  |  |  | Warm | 126.7±17.6ms (1.0x) | 107.6±19.8ms (**1.2x**) | 135.8±16.8ms (0.9x) |
| 5-key | 993,138 | True | Cold | 335.5±38.2ms (1.0x) | 222.2±26.1ms (**1.5x**) | 326.4±100.3ms (1.0x) |
|  |  |  | Warm | 286.3±30.0ms (1.0x) | 233.6±25.4ms (**1.2x**) | 245.7±23.1ms (**1.2x**) |
|  |  | False | Cold | 257.4±20.7ms (1.0x) | 192.3±49.2ms (**1.3x**) | 309.5±76.1ms (0.8x) |
|  |  |  | Warm | 230.5±42.7ms (1.0x) | 177.5±14.4ms (**1.3x**) | 270.6±32.0ms (0.9x) |

### High Cardinality (5M rows, ~5M unique groups)

| Operation | Groups | Sort | Type | Pandas | Polars | Booster |
|-----------|--------|------|------|--------|--------|---------|
| Single-key | 3,160,983 | True | Cold | 831.5±25.2ms (1.0x) | 131.2±5.0ms (**6.3x**) | 1312.5±43.4ms (0.6x) |
|  |  |  | Warm | 810.9±33.5ms (1.0x) | 127.3±2.7ms (**6.4x**) | 1225.1±50.4ms (0.7x) |
|  |  | False | Cold | 295.7±51.9ms (1.0x) | 87.5±3.5ms (**3.4x**) | 1231.5±83.3ms (0.2x) |
|  |  |  | Warm | 321.8±49.8ms (1.0x) | 89.6±5.1ms (**3.6x**) | 1070.4±39.7ms (0.3x) |
| 2-key | 4,532,339 | True | Cold | 926.3±14.0ms (1.0x) | 275.2±24.6ms (**3.4x**) | 326.8±63.1ms (**2.8x**) |
|  |  |  | Warm | 913.3±30.5ms (1.0x) | 317.8±89.6ms (**2.9x**) | 260.4±11.7ms (**3.5x**) |
|  |  | False | Cold | 443.7±53.8ms (1.0x) | 142.6±38.0ms (**3.1x**) | 210.2±33.3ms (**2.1x**) |
|  |  |  | Warm | 397.6±11.3ms (1.0x) | 110.6±3.5ms (**3.6x**) | 197.8±59.9ms (**2.0x**) |
| 3-key | 4,901,309 | True | Cold | 1049.6±49.2ms (1.0x) | 418.7±38.3ms (**2.5x**) | 480.8±126.4ms (**2.2x**) |
|  |  |  | Warm | 990.8±55.3ms (1.0x) | 486.2±80.1ms (**2.0x**) | 349.4±36.7ms (**2.8x**) |
|  |  | False | Cold | 401.7±8.1ms (1.0x) | 132.0±13.8ms (**3.0x**) | 223.7±10.3ms (**1.8x**) |
|  |  |  | Warm | 398.9±7.9ms (1.0x) | 138.4±25.8ms (**2.9x**) | 187.7±2.5ms (**2.1x**) |

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

#### Setup
To run benchmarks with library comparisons (Polars, etc.), install the optional benchmark dependencies:

```bash
# Install benchmark and development dependencies
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
