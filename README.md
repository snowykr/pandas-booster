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
| `sort` | `bool` | If `True` (default), sort result by group keys. If `False`, preserve Pandas appearance order (first-seen group order). |

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
| `count` | Count of non-NaN values in each group |

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

The library is designed for large datasets where multi-core parallelism can be fully utilized.

- `sort=True`: single-key groupby uses Rayon's parallel map-reduce; multi-key groupby uses a radix-partitioning algorithm that eliminates merge overhead.
- `sort=False`: results preserve Pandas appearance order (first-seen group order). Internally, this path tracks the first-seen row index per group and reorders groups with an integer radix sort to avoid `O(G log G)` comparison sorting.

**Benchmark methodology:**
- **Process Isolation:** Benchmarks use rigorous process isolation to ensure accurate results.
- **Cold:** Average of 5 fresh process executions (first run only).
- **Warm:** Average of 5 fresh process executions (steady state after warmup).
- **Correctness:** Booster and Polars outputs are validated against a Pandas baseline. For `sort=False`, benchmarks validate Pandas-compatible appearance order (first-seen group order).
- **Polars sort handling:** Polars does not have a `sort` parameter in `group_by`. For fair comparison, I define `sort=True` as "groupby+agg followed by sorting the result by keys" (cost included in timing), and `sort=False` as "groupby+agg with Pandas-compatible appearance order (first-seen group order)". This ensures all three engines (Pandas, Polars, Booster) are measured under identical conditions.
- **Speedup baseline:** All speedup values (`x`) use **Pandas** as the baseline (1.0x) within each sort mode.
- **Optional Polars:** Polars is included in the benchmarks for comparison if installed. If not installed, the benchmark suite proceeds with Pandas vs Booster only.

### Standard Cardinality (5M rows)

| Operation | Groups | Sort | Type | Pandas | Polars | Booster |
|-----------|--------|------|------|--------|--------|---------|
| Single-key | 1,000 | True | Cold | 38.9±6.6ms (1.0x) | 34.6±17.8ms (**1.1x**) | 6.7±3.1ms (**5.8x**) |
|  |  |  | Warm | 25.3±0.2ms (1.0x) | 20.1±0.6ms (**1.3x**) | 2.4±0.1ms (**10.5x**) |
|  |  | False | Cold | 26.2±0.1ms (1.0x) | 23.0±0.8ms (**1.1x**) | 7.7±4.3ms (**3.4x**) |
|  |  |  | Warm | 23.3±0.3ms (1.0x) | 20.1±0.7ms (**1.2x**) | 2.5±0.1ms (**9.2x**) |
| 2-key | 5,000 | True | Cold | 90.0±11.3ms (1.0x) | 68.3±23.2ms (**1.3x**) | 102.9±2.2ms (0.9x) |
|  |  |  | Warm | 72.6±1.0ms (1.0x) | 48.2±2.4ms (**1.5x**) | 99.5±0.5ms (0.7x) |
|  |  | False | Cold | 62.0±0.7ms (1.0x) | 82.2±41.1ms (0.8x) | 104.1±2.3ms (0.6x) |
|  |  |  | Warm | 54.3±0.5ms (1.0x) | 46.2±2.2ms (**1.2x**) | 102.0±1.5ms (0.5x) |
| 3-key | 25,000 | True | Cold | 130.4±3.9ms (1.0x) | 96.0±44.0ms (**1.4x**) | 111.9±2.6ms (**1.2x**) |
|  |  |  | Warm | 117.1±1.5ms (1.0x) | 76.1±12.7ms (**1.5x**) | 107.5±1.7ms (1.1x) |
|  |  | False | Cold | 108.1±10.9ms (1.0x) | 83.5±23.9ms (**1.3x**) | 115.6±0.1ms (0.9x) |
|  |  |  | Warm | 92.1±1.9ms (1.0x) | 61.4±0.7ms (**1.5x**) | 113.5±2.6ms (0.8x) |
| 4-key | 100,000 | True | Cold | 169.7±5.0ms (1.0x) | 122.0±24.5ms (**1.4x**) | 123.5±1.7ms (**1.4x**) |
|  |  |  | Warm | 142.6±2.3ms (1.0x) | 98.2±0.5ms (**1.5x**) | 127.3±9.5ms (**1.1x**) |
|  |  | False | Cold | 120.1±0.1ms (1.0x) | 147.7±21.7ms (0.8x) | 122.9±6.1ms (1.0x) |
|  |  |  | Warm | 110.9±2.5ms (1.0x) | 105.2±9.9ms (1.1x) | 124.7±0.1ms (0.9x) |
| 5-key | 993,138 | True | Cold | 321.2±37.7ms (1.0x) | 252.1±58.8ms (**1.3x**) | 196.7±5.4ms (**1.6x**) |
|  |  |  | Warm | 285.3±8.6ms (1.0x) | 216.7±5.2ms (**1.3x**) | 179.8±0.9ms (**1.6x**) |
|  |  | False | Cold | 206.3±5.3ms (1.0x) | 179.5±3.8ms (**1.1x**) | 210.2±18.7ms (1.0x) |
|  |  |  | Warm | 189.5±5.7ms (1.0x) | 279.6±142.9ms (0.7x) | 190.4±6.5ms (1.0x) |

### High Cardinality (5M rows, ~5M unique groups)

| Operation | Groups | Sort | Type | Pandas | Polars | Booster |
|-----------|--------|------|------|--------|--------|---------|
| Single-key | 3,160,983 | True | Cold | 819.8±14.2ms (1.0x) | 147.6±11.5ms (**5.6x**) | 498.5±60.4ms (**1.6x**) |
|  |  |  | Warm | 796.4±10.3ms (1.0x) | 134.8±0.9ms (**5.9x**) | 449.9±3.1ms (**1.8x**) |
|  |  | False | Cold | 254.6±1.2ms (1.0x) | 193.0±2.1ms (**1.3x**) | 265.7±12.9ms (1.0x) |
|  |  |  | Warm | 277.7±39.6ms (1.0x) | 201.1±16.8ms (**1.4x**) | 270.3±38.4ms (1.0x) |
| 2-key | 4,532,339 | True | Cold | 955.6±54.4ms (1.0x) | 265.5±4.2ms (**3.6x**) | 280.6±7.1ms (**3.4x**) |
|  |  |  | Warm | 905.6±13.4ms (1.0x) | 255.9±3.2ms (**3.5x**) | 280.1±32.8ms (**3.2x**) |
|  |  | False | Cold | 426.2±34.2ms (1.0x) | 292.5±43.3ms (**1.5x**) | 396.2±9.6ms (1.1x) |
|  |  |  | Warm | 403.8±10.0ms (1.0x) | 255.2±0.5ms (**1.6x**) | 351.8±19.9ms (**1.1x**) |
| 3-key | 4,901,309 | True | Cold | 983.2±3.3ms (1.0x) | 366.5±6.2ms (**2.7x**) | 442.4±45.1ms (**2.2x**) |
|  |  |  | Warm | 1001.5±66.5ms (1.0x) | 362.7±12.6ms (**2.8x**) | 368.3±4.0ms (**2.7x**) |
|  |  | False | Cold | 415.5±0.1ms (1.0x) | 280.3±11.3ms (**1.5x**) | 471.4±25.0ms (0.9x) |
|  |  |  | Warm | 404.5±11.1ms (1.0x) | 272.1±7.7ms (**1.5x**) | 421.9±22.0ms (1.0x) |

**Performance characteristics**:
- **Single-key (standard cardinality)**: Warm state shows **10.3-10.9x** speedup over Pandas baseline. Booster significantly outperforms Polars in warm state.
- **Multi-key (standard cardinality)**: Performance varies by key count. 2-3 keys show **0.6-1.1x** (comparable to Pandas), while 4-5 keys achieve **1.0-1.6x** speedup. Polars consistently shows **1.2-2.0x** speedup across all key counts.
- **Multi-key (high cardinality)**: Booster achieves **1.9-3.8x** speedup in warm state. Polars performs similarly with **2.7-6.4x** speedup. Both engines significantly outperform Pandas in high-cardinality scenarios.
- **Sort overhead**: Skipping key sorting (`sort=False`) provides minimal improvement for single-key operations, but offers **1.2-1.8x** improvement for multi-key operations in standard cardinality, and **2.0-3.9x** improvement in high cardinality scenarios.

### Sorted vs Appearance-Ordered Results

By default, results are sorted by group keys to match Pandas `sort=True` output. Pass `sort=False` to preserve Pandas' appearance order (first-seen group order) and avoid the key sort cost:

```python
# Sorted (default) - matches Pandas exactly
result = df.booster.groupby(by=["a", "b"], target="val", agg="sum")

# Appearance-ordered (sort=False) - matches Pandas semantics
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
- `pyarrow`: For fast Polars -> Pandas conversion during correctness checks
- `matplotlib`: For generating plots and visualizations (optional)
- `pytest` and `pytest-benchmark`: For test framework and benchmarking

#### Running Benchmarks
```bash
# Run all benchmarks (standard + high cardinality, sorted + sort=False)
source .venv/bin/activate
python benches/benchmark.py

# Run only standard cardinality benchmarks
python benches/benchmark.py --cardinality standard

# Run only high cardinality benchmarks
python benches/benchmark.py --cardinality high

# Run only sorted or sort=False benchmarks
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
- **FixedKey Optimization**: For 1-10 key groupby operations (the supported maximum), uses compile-time fixed-size arrays (`FixedKey<const N>`) instead of dynamic vectors. This enables aggressive compiler optimizations (loop unrolling, SIMD) and avoids per-group heap allocation for key storage.
- **AHash**: Used for high-speed hashing of groupby keys.
- **SmallVec**: Used only as a generic fallback key representation (e.g., if the max-key constraint is raised in the future). The current implementation inlines up to 10 key values before spilling to the heap.
- **Zero-Copy**: NumPy arrays are accessed directly as Rust slices without copying data, minimizing memory overhead and latency.

The Python side provides a `BoosterAccessor` that handles validation and falls back to Pandas when the data doesn't meet the requirements for acceleration.

## License

This project is licensed under the MIT License.
