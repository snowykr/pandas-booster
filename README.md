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
To build and install from source, you need a Rust toolchain and `maturin`:
```bash
pip install maturin
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

### Standard Cardinality (5M rows)

| Operation | Pandas | Booster (sort=True) | Booster (sort=False) | Speedup (sort=True) | Speedup (sort=False) |
|-----------|--------|---------------------|----------------------|---------------------|----------------------|
| Single-key groupby | 25.0ms | 3.1ms | 2.4ms | **8.1x** | **10.4x** |
| 2-key groupby | 67.0ms | 106.0ms | 102.2ms | 0.6x | 0.7x |
| 3-key groupby | 104.2ms | 111.2ms | 133.1ms | 0.9x | 0.8x |
| 4-key groupby | 143.3ms | 126.9ms | 122.7ms | 1.1x | **1.2x** |
| 5-key groupby | 252.8ms | 222.5ms | 237.6ms | **1.1x** | **1.1x** |

### High Cardinality (5M rows, ~5M unique groups)

| Operation | Groups | Pandas | Booster (sort=True) | Booster (sort=False) | Speedup (sort=True) | Speedup (sort=False) |
|-----------|--------|--------|---------------------|----------------------|---------------------|----------------------|
| 2-key groupby | 4.5M | 823.6ms | 265.8ms | 248.8ms | **3.1x** | **3.3x** |
| 3-key groupby | 4.9M | 886.5ms | 384.4ms | 344.9ms | **2.3x** | **2.6x** |

**Performance characteristics:**
- **Single-key**: Consistent **8-10x** speedup.
- **Multi-key (standard cardinality)**: Comparable to Pandas; sort=False provides a small edge.
- **Multi-key (high cardinality)**: **2-3x** speedup. `sort=False` is fastest as it skips the expensive Python-side sorting of large results.

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
# Single-key benchmark
python benches/benchmark.py

# Multi-key benchmarks
python benches/benchmark_multi_key.py --preset readme
python benches/benchmark_multi_key.py --preset cardinality
```

## Development

### Building
Build the extension module in-place:
```bash
maturin develop
```

### Testing
Run the test suite using `pytest`:
```bash
pytest tests/
```

### Benchmarking
```bash
# Basic benchmark
python benches/benchmark.py

# Save results to file
python benches/benchmark.py --output results.csv --format csv

# Quick benchmark with fewer iterations
python benches/benchmark.py --quick
```

## Architecture Overview

`pandas-booster` uses a hybrid Rust/Python architecture:

- **PyO3**: Provides the bridge between Python and Rust.
- **Rayon**: Implements a work-stealing parallel scheduler for multi-core processing.
- **Radix Partitioning**: Multi-key groupby uses a 4-phase radix partitioning algorithm (histogram → prefix sum → scatter → aggregate) that eliminates merge overhead.
- **AHash**: Used for high-speed hashing of groupby keys.
- **SmallVec**: Optimizes multi-key storage by inlining up to 4 keys without heap allocation.
- **Zero-Copy**: NumPy arrays are accessed directly as Rust slices without copying data, minimizing memory overhead and latency.

The Python side provides a `BoosterAccessor` that handles validation and falls back to Pandas when the data doesn't meet the requirements for acceleration.

## License

This project is licensed under the MIT License.
