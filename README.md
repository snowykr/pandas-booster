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

### Configuration

`pandas-booster` defaults to Rust-side sorting kernels for `sort=True`.

Emergency toggle (panic button):

- `PANDAS_BOOSTER_FORCE_PANDAS_SORT=1`: force Python `Series.sort_index()` after Rust aggregation.
- `PANDAS_BOOSTER_FORCE_PANDAS_SORT=0` (default when unset): keep Rust-side sorting.

This toggle is intended for quick rollback if a Rust sorting bug is discovered. Forcing Python sort moves the `sort=True` cost to Pandas and is slower. If the Rust wheel is missing `*_sorted` kernels, pandas-booster also falls back to Python `sort_index()` automatically.

Note: the Rust-side `sort=True` kernels allocate a permutation vector and perform an `O(G log G)` comparison sort over groups (G = number of groups). This can increase memory usage at very high cardinality.

Note: the benchmark runner defaults `PANDAS_BOOSTER_FORCE_PANDAS_SORT=0`.

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
- **Samples:** The tables below were generated with `--samples 20` (default: 5). Each sample runs in a fresh Python process.
- **Cold:** Average of 20 fresh process executions (1st run measured immediately).
- **Warm:** Average of 20 fresh process executions. Each process runs Cold once and a Warmup once (both discarded), then measures the next run (steady state).
- **Correctness:** Booster and Polars outputs are validated against a Pandas baseline. For `sort=False`, benchmarks validate Pandas-compatible appearance order (first-seen group order).
- **Polars sort handling:** Polars does not have a `sort` parameter in `group_by`. For fair comparison, I define `sort=True` as "groupby+agg followed by sorting the result by keys" (cost included in timing), and `sort=False` as "groupby+agg with Pandas-compatible appearance order (first-seen group order)". This ensures all three engines (Pandas, Polars, Booster) are measured under identical conditions.
- **Speedup baseline:** All speedup values (`x`) use **Pandas** as the baseline (1.0x) within each sort mode.
- **Optional Polars:** Polars is included in the benchmarks for comparison if installed. If not installed, the benchmark suite proceeds with Pandas vs Booster only.

### Standard Cardinality (5M rows)

| Operation | Groups | Sort | Type | Pandas | Polars | Booster |
|-----------|--------|------|------|--------|--------|---------|
| Single-key | 1,000 | True | Cold | 29.1±0.8ms (1.0x) | 23.0±9.0ms (**1.3x**) | 5.1±1.5ms (**5.7x**) |
|  |  |  | Warm | 27.1±3.3ms (1.0x) | 17.7±2.2ms (**1.5x**) | 2.3±0.1ms (**11.9x**) |
|  |  | False | Cold | 26.5±0.8ms (1.0x) | 20.2±2.4ms (**1.3x**) | 5.5±0.9ms (**4.8x**) |
|  |  |  | Warm | 22.6±0.2ms (1.0x) | 18.1±1.2ms (**1.2x**) | 2.3±0.1ms (**9.8x**) |
| 2-key | 5,000 | True | Cold | 83.5±10.0ms (1.0x) | 54.3±20.3ms (**1.5x**) | 94.4±1.5ms (0.9x) |
|  |  |  | Warm | 68.3±1.6ms (1.0x) | 40.4±2.7ms (**1.7x**) | 91.0±1.4ms (0.8x) |
|  |  | False | Cold | 61.8±2.9ms (1.0x) | 57.9±24.7ms (1.1x) | 97.0±2.6ms (0.6x) |
|  |  |  | Warm | 53.1±0.3ms (1.0x) | 43.8±4.6ms (**1.2x**) | 95.0±4.5ms (0.6x) |
| 3-key | 25,000 | True | Cold | 123.5±9.9ms (1.0x) | 70.2±22.5ms (**1.8x**) | 102.3±3.3ms (**1.2x**) |
|  |  |  | Warm | 107.7±1.3ms (1.0x) | 53.4±2.7ms (**2.0x**) | 98.7±0.6ms (1.1x) |
|  |  | False | Cold | 99.7±3.1ms (1.0x) | 66.9±14.2ms (**1.5x**) | 106.0±1.4ms (0.9x) |
|  |  |  | Warm | 92.0±3.4ms (1.0x) | 52.6±2.0ms (**1.7x**) | 104.6±2.7ms (0.9x) |
| 4-key | 100,000 | True | Cold | 143.3±2.2ms (1.0x) | 97.6±9.0ms (**1.5x**) | 113.5±3.5ms (**1.3x**) |
|  |  |  | Warm | 131.7±2.5ms (1.0x) | 79.3±0.9ms (**1.7x**) | 112.8±1.7ms (**1.2x**) |
|  |  | False | Cold | 119.7±5.9ms (1.0x) | 95.3±6.8ms (**1.3x**) | 114.9±1.9ms (1.0x) |
|  |  |  | Warm | 108.8±3.1ms (1.0x) | 82.7±8.6ms (**1.3x**) | 111.6±2.0ms (1.0x) |
| 5-key | 993,138 | True | Cold | 279.7±11.5ms (1.0x) | 199.7±19.6ms (**1.4x**) | 218.2±22.0ms (**1.3x**) |
|  |  |  | Warm | 265.0±5.9ms (1.0x) | 180.6±4.8ms (**1.5x**) | 193.1±1.8ms (**1.4x**) |
|  |  | False | Cold | 191.9±12.8ms (1.0x) | 178.1±18.9ms (1.1x) | 190.3±11.8ms (1.0x) |
|  |  |  | Warm | 172.7±12.6ms (1.0x) | 159.6±7.0ms (1.1x) | 174.8±8.4ms (1.0x) |

### High Cardinality (5M rows, ~5M unique groups)

| Operation | Groups | Sort | Type | Pandas | Polars | Booster |
|-----------|--------|------|------|--------|--------|---------|
| Single-key | 3,160,983 | True | Cold | 741.1±3.5ms (1.0x) | 122.0±3.1ms (**6.1x**) | 235.3±5.7ms (**3.1x**) |
|  |  |  | Warm | 731.3±4.5ms (1.0x) | 114.3±2.4ms (**6.4x**) | 216.9±3.9ms (**3.4x**) |
|  |  | False | Cold | 246.1±2.4ms (1.0x) | 174.0±9.3ms (**1.4x**) | 235.3±14.3ms (1.0x) |
|  |  |  | Warm | 223.1±4.6ms (1.0x) | 171.6±5.0ms (**1.3x**) | 229.8±12.2ms (1.0x) |
| 2-key | 4,532,339 | True | Cold | 870.3±14.5ms (1.0x) | 234.8±2.0ms (**3.7x**) | 423.0±6.5ms (**2.1x**) |
|  |  |  | Warm | 870.5±54.6ms (1.0x) | 233.2±3.9ms (**3.7x**) | 422.1±46.4ms (**2.1x**) |
|  |  | False | Cold | 365.8±8.4ms (1.0x) | 244.8±14.1ms (**1.5x**) | 354.0±23.8ms (1.0x) |
|  |  |  | Warm | 365.4±15.2ms (1.0x) | 246.8±6.1ms (**1.5x**) | 313.8±8.8ms (**1.2x**) |
| 3-key | 4,901,309 | True | Cold | 944.3±33.2ms (1.0x) | 341.1±21.9ms (**2.8x**) | 604.4±21.7ms (**1.6x**) |
|  |  |  | Warm | 911.4±27.7ms (1.0x) | 329.9±14.2ms (**2.8x**) | 576.7±27.5ms (**1.6x**) |
|  |  | False | Cold | 469.1±54.6ms (1.0x) | 260.4±13.0ms (**1.8x**) | 404.0±17.7ms (**1.2x**) |
|  |  |  | Warm | 403.1±15.3ms (1.0x) | 267.5±20.8ms (**1.5x**) | 356.2±16.8ms (**1.1x**) |

**Performance characteristics**:
- **Single-key (standard cardinality)**: Warm state shows **9.8-11.9x** speedup over Pandas baseline (cold: **4.8-5.7x**). Booster outperforms Polars in warm state.
- **Multi-key (standard cardinality)**: Performance depends on key count and `sort`. 2-key is slower than Pandas (**0.6-0.8x** warm); 3-key ranges **0.9-1.1x** warm; 4-key ranges **1.0-1.2x** warm; 5-key ranges **1.0-1.4x** warm. Polars is typically faster here (**1.1-2.0x** warm).
- **Multi-key (high cardinality)**: With `sort=True`, Booster achieves **1.6-3.4x** speedup in warm state; with `sort=False`, it is near parity (**1.0-1.2x**). Polars is faster than Booster on these workloads (sorted: **2.8-6.4x**, unsorted: **1.3-1.5x**).
- **Sort overhead**: For Pandas, `sort=False` is often a meaningful win (about **1.2-1.4x** on standard multi-key, and ~**3.0x** on high-cardinality single-key). For Booster, the difference between `sort=True/False` is smaller and can go either direction (roughly **0.8-1.1x** in these tables), so choose based on desired output ordering first.

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
python benches/benchmark.py --samples 20 --output results.md
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

# Adjust sample count (applies to both cold and warm; default: 5)
python benches/benchmark.py --samples 20
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
