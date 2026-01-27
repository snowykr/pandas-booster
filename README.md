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
| Single-key | 1,000 | True | Cold | 30.5±4.5ms (1.0x) | 21.7±7.0ms (**1.4x**) | 4.4±0.7ms (**6.9x**) |
|  |  |  | Warm | 25.8±3.6ms (1.0x) | 17.1±1.0ms (**1.5x**) | 2.2±0.1ms (**12.0x**) |
|  |  | False | Cold | 26.1±0.9ms (1.0x) | 23.1±6.9ms (**1.1x**) | 4.8±0.8ms (**5.5x**) |
|  |  |  | Warm | 23.7±0.9ms (1.0x) | 19.1±2.1ms (**1.2x**) | 2.3±0.1ms (**10.2x**) |
| 2-key | 5,000 | True | Cold | 87.2±9.2ms (1.0x) | 46.0±9.9ms (**1.9x**) | 94.0±5.7ms (0.9x) |
|  |  |  | Warm | 69.6±3.2ms (1.0x) | 40.9±6.1ms (**1.7x**) | 89.5±4.1ms (0.8x) |
|  |  | False | Cold | 85.6±38.8ms (1.0x) | 60.9±26.6ms (**1.4x**) | 105.6±10.5ms (0.8x) |
|  |  |  | Warm | 56.1±5.0ms (1.0x) | 43.4±7.6ms (**1.3x**) | 97.8±7.1ms (0.6x) |
| 3-key | 25,000 | True | Cold | 126.9±7.6ms (1.0x) | 62.5±9.1ms (**2.0x**) | 99.0±3.5ms (**1.3x**) |
|  |  |  | Warm | 107.3±1.8ms (1.0x) | 51.9±4.7ms (**2.1x**) | 96.4±3.3ms (**1.1x**) |
|  |  | False | Cold | 106.7±8.6ms (1.0x) | 72.0±17.0ms (**1.5x**) | 112.1±5.6ms (1.0x) |
|  |  |  | Warm | 92.8±3.0ms (1.0x) | 63.1±10.1ms (**1.5x**) | 119.1±36.9ms (0.8x) |
| 4-key | 100,000 | True | Cold | 167.9±40.3ms (1.0x) | 95.5±16.5ms (**1.8x**) | 110.5±3.0ms (**1.5x**) |
|  |  |  | Warm | 139.5±15.5ms (1.0x) | 78.3±5.5ms (**1.8x**) | 106.9±1.8ms (**1.3x**) |
|  |  | False | Cold | 129.1±9.6ms (1.0x) | 124.1±31.1ms (1.0x) | 126.6±17.1ms (1.0x) |
|  |  |  | Warm | 113.2±9.2ms (1.0x) | 93.6±10.7ms (**1.2x**) | 120.3±9.1ms (0.9x) |
| 5-key | 993,138 | True | Cold | 272.0±11.3ms (1.0x) | 185.1±7.7ms (**1.5x**) | 191.6±11.5ms (**1.4x**) |
|  |  |  | Warm | 252.2±3.8ms (1.0x) | 179.1±6.2ms (**1.4x**) | 184.8±14.2ms (**1.4x**) |
|  |  | False | Cold | 209.2±34.6ms (1.0x) | 186.3±35.8ms (**1.1x**) | 194.7±27.5ms (1.1x) |
|  |  |  | Warm | 176.3±17.3ms (1.0x) | 175.2±16.1ms (1.0x) | 178.8±8.1ms (1.0x) |

### High Cardinality (5M rows, ~5M unique groups)

| Operation | Groups | Sort | Type | Pandas | Polars | Booster |
|-----------|--------|------|------|--------|--------|---------|
| Single-key | 3,160,983 | True | Cold | 749.8±20.3ms (1.0x) | 127.5±16.4ms (**5.9x**) | 218.0±29.3ms (**3.4x**) |
|  |  |  | Warm | 736.3±37.0ms (1.0x) | 122.3±11.0ms (**6.0x**) | 202.6±25.7ms (**3.6x**) |
|  |  | False | Cold | 264.6±45.4ms (1.0x) | 180.0±10.2ms (**1.5x**) | 252.6±24.3ms (1.0x) |
|  |  |  | Warm | 243.7±28.8ms (1.0x) | 178.8±15.3ms (**1.4x**) | 229.4±23.8ms (1.1x) |
| 2-key | 4,532,339 | True | Cold | 843.1±18.4ms (1.0x) | 228.7±3.6ms (**3.7x**) | 326.1±27.0ms (**2.6x**) |
|  |  |  | Warm | 827.1±29.6ms (1.0x) | 223.6±4.1ms (**3.7x**) | 301.2±11.5ms (**2.7x**) |
|  |  | False | Cold | 411.7±36.1ms (1.0x) | 252.1±10.7ms (**1.6x**) | 355.9±35.2ms (**1.2x**) |
|  |  |  | Warm | 361.8±10.5ms (1.0x) | 258.4±28.1ms (**1.4x**) | 339.1±63.2ms (1.1x) |
| 3-key | 4,901,309 | True | Cold | 925.7±35.7ms (1.0x) | 326.9±42.1ms (**2.8x**) | 442.3±50.6ms (**2.1x**) |
|  |  |  | Warm | 900.5±33.7ms (1.0x) | 331.2±52.8ms (**2.7x**) | 400.5±22.5ms (**2.2x**) |
|  |  | False | Cold | 459.1±104.0ms (1.0x) | 293.6±84.2ms (**1.6x**) | 406.7±12.1ms (**1.1x**) |
|  |  |  | Warm | 391.2±29.1ms (1.0x) | 264.3±17.0ms (**1.5x**) | 383.2±32.2ms (1.0x) |

**Performance characteristics**:
- **Single-key (standard cardinality)**: Warm state shows **10.2-12.0x** speedup over Pandas baseline (cold: **5.5-6.9x**). Booster outperforms Polars in warm state.
- **Multi-key (standard cardinality)**: Performance depends on key count and `sort`. 2-key is slower than Pandas (**0.6-0.8x** warm); 3-key ranges **0.8-1.1x** warm; 4-key ranges **0.9-1.3x** warm; 5-key ranges **1.0-1.4x** warm. Polars is typically faster here (**1.0-2.1x** warm).
- **Multi-key (high cardinality)**: With `sort=True`, Booster achieves **2.2-3.6x** speedup in warm state; with `sort=False`, it is near parity (**1.0-1.1x**). Polars is faster than Booster on these workloads (sorted: **2.7-6.0x**, unsorted: **1.4-1.6x**).
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
