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
| Single-key | 1,000 | True | Cold | 32.3±2.6ms (1.0x) | 23.4±2.4ms (**1.4x**) | 5.6±0.6ms (**5.8x**) |
|  |  |  | Warm | 25.2±0.9ms (1.0x) | 22.0±6.9ms (**1.1x**) | 2.3±0.0ms (**11.1x**) |
|  |  | False | Cold | 26.4±0.8ms (1.0x) | 25.7±6.6ms (1.0x) | 7.8±7.9ms (**3.4x**) |
|  |  |  | Warm | 23.0±0.7ms (1.0x) | 19.1±1.0ms (**1.2x**) | 2.2±0.1ms (**10.4x**) |
| 2-key | 5,000 | True | Cold | 89.9±6.7ms (1.0x) | 59.4±28.4ms (**1.5x**) | 100.5±4.9ms (0.9x) |
|  |  |  | Warm | 69.3±1.9ms (1.0x) | 41.0±2.2ms (**1.7x**) | 92.6±1.3ms (0.7x) |
|  |  | False | Cold | 71.2±3.2ms (1.0x) | 61.1±20.2ms (**1.2x**) | 97.3±2.9ms (0.7x) |
|  |  |  | Warm | 54.1±1.4ms (1.0x) | 43.1±2.5ms (**1.3x**) | 92.7±1.3ms (0.6x) |
| 3-key | 25,000 | True | Cold | 128.9±9.0ms (1.0x) | 67.6±9.5ms (**1.9x**) | 100.5±1.3ms (**1.3x**) |
|  |  |  | Warm | 109.8±1.3ms (1.0x) | 52.1±3.7ms (**2.1x**) | 98.2±1.3ms (**1.1x**) |
|  |  | False | Cold | 109.7±5.8ms (1.0x) | 72.5±14.3ms (**1.5x**) | 104.8±2.9ms (1.0x) |
|  |  |  | Warm | 91.3±1.9ms (1.0x) | 57.0±4.1ms (**1.6x**) | 101.1±2.7ms (0.9x) |
| 4-key | 100,000 | True | Cold | 156.4±5.3ms (1.0x) | 93.7±11.5ms (**1.7x**) | 112.3±2.3ms (**1.4x**) |
|  |  |  | Warm | 135.1±2.9ms (1.0x) | 79.3±3.6ms (**1.7x**) | 108.5±2.6ms (**1.2x**) |
|  |  | False | Cold | 147.8±61.5ms (1.0x) | 97.3±10.9ms (**1.5x**) | 112.5±1.7ms (**1.3x**) |
|  |  |  | Warm | 111.4±4.2ms (1.0x) | 88.0±3.1ms (**1.3x**) | 111.1±4.4ms (1.0x) |
| 5-key | 993,138 | True | Cold | 280.0±7.8ms (1.0x) | 198.5±10.9ms (**1.4x**) | 207.0±4.4ms (**1.4x**) |
|  |  |  | Warm | 255.2±3.4ms (1.0x) | 206.2±29.7ms (**1.2x**) | 192.3±4.5ms (**1.3x**) |
|  |  | False | Cold | 197.9±14.8ms (1.0x) | 161.5±21.7ms (**1.2x**) | 199.1±16.0ms (1.0x) |
|  |  |  | Warm | 173.7±3.4ms (1.0x) | 141.3±2.7ms (**1.2x**) | 175.4±8.8ms (1.0x) |

### High Cardinality (5M rows, ~5M unique groups)

| Operation | Groups | Sort | Type | Pandas | Polars | Booster |
|-----------|--------|------|------|--------|--------|---------|
| Single-key | 3,160,983 | True | Cold | 823.8±52.5ms (1.0x) | 152.9±27.5ms (**5.4x**) | 458.6±13.8ms (**1.8x**) |
|  |  |  | Warm | 819.3±81.5ms (1.0x) | 134.5±19.5ms (**6.1x**) | 483.3±71.2ms (**1.7x**) |
|  |  | False | Cold | 237.3±4.2ms (1.0x) | 92.1±21.7ms (**2.6x**) | 152.6±6.2ms (**1.6x**) |
|  |  |  | Warm | 231.0±6.0ms (1.0x) | 83.6±2.5ms (**2.8x**) | 139.3±7.9ms (**1.7x**) |
| 2-key | 4,532,339 | True | Cold | 911.6±25.8ms (1.0x) | 263.7±23.6ms (**3.5x**) | 270.1±11.8ms (**3.4x**) |
|  |  |  | Warm | 910.8±65.5ms (1.0x) | 275.5±48.0ms (**3.3x**) | 238.8±7.9ms (**3.8x**) |
|  |  | False | Cold | 390.0±15.5ms (1.0x) | 213.6±110.8ms (**1.8x**) | 196.5±45.0ms (**2.0x**) |
|  |  |  | Warm | 359.8±5.2ms (1.0x) | 126.1±8.8ms (**2.9x**) | 199.1±105.4ms (**1.8x**) |
| 3-key | 4,901,309 | True | Cold | 955.5±16.8ms (1.0x) | 360.7±20.7ms (**2.6x**) | 404.1±25.5ms (**2.4x**) |
|  |  |  | Warm | 997.6±83.4ms (1.0x) | 373.6±36.9ms (**2.7x**) | 347.5±18.5ms (**2.9x**) |
|  |  | False | Cold | 417.9±19.5ms (1.0x) | 136.4±18.6ms (**3.1x**) | 251.0±35.8ms (**1.7x**) |
|  |  |  | Warm | 408.9±62.2ms (1.0x) | 137.3±14.3ms (**3.0x**) | 197.7±21.2ms (**2.1x**) |

**Performance characteristics**:
- **Single-key (standard cardinality)**: Warm state shows **10.3-10.9x** speedup over Pandas baseline. Booster significantly outperforms Polars in warm state.
- **Multi-key (standard cardinality)**: Performance varies by key count. 2-3 keys show **0.6-1.1x** (comparable to Pandas), while 4-5 keys achieve **1.0-1.6x** speedup. Polars consistently shows **1.2-2.0x** speedup across all key counts.
- **Multi-key (high cardinality)**: Booster achieves **1.9-3.8x** speedup in warm state. Polars performs similarly with **2.7-6.4x** speedup. Both engines significantly outperform Pandas in high-cardinality scenarios.
- **Sort overhead**: Removing sort (`sort=False`) provides minimal improvement for single-key operations, but offers **1.2-1.8x** improvement for multi-key operations in standard cardinality, and **2.0-3.9x** improvement in high cardinality scenarios.

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
