# pandas-booster

[![CI](https://github.com/snowykr/pandas-booster/actions/workflows/ci.yml/badge.svg)](https://github.com/snowykr/pandas-booster/actions)
[![Python Versions](https://img.shields.io/pypi/pyversions/pandas-booster.svg)](https://pypi.org/project/pandas-booster/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

pandas-booster is a high-performance numerical acceleration library for Pandas that offloads heavy computations to Rust. It leverages multi-core parallelism and zero-copy data access to provide significant speedups for large-scale data processing tasks.

This project is an independent third-party package and is not affiliated with, endorsed by, or sponsored by the pandas project or NumFOCUS.

## Features

- Parallel GroupBy aggregations using Rayon (single and multi-column)
- Fast hashing with AHash
- Zero-copy interop between NumPy and Rust
- Release of the Python Global Interpreter Lock (GIL) during computation
- Seamless integration as a Pandas DataFrame accessor

## Installation

### From PyPI
Install the latest published release from PyPI:
```bash
pip install pandas-booster
```

### Development Setup
To build and install from source, **all development commands in this repository assume you are using an activated virtual environment** (I recommend `.venv`).

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install build tools and dependencies
pip install "maturin>=1.13,<2.0"
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

- `PANDAS_BOOSTER_FORCE_PANDAS_SORT` (default: OFF when unset):
  - truthy (`1/true/yes/on`, case-insensitive): force Python `Series.sort_index()` after Rust aggregation for `sort=True`.
  - anything else (including `0/false/no/off`): keep Rust-side sorting.
- `PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY` (default: OFF when unset):
  - truthy (`1/true/yes/on`, case-insensitive): force pandas fallback for **single-key float** `sum`/`mean`/`std`/`var`.
  - anything else (including `0/false/no/off`): use Rust deterministic reduction kernels.

These toggles are intended for quick rollback if a Rust ordering or deterministic-float reduction issue is discovered. Forcing Python sort moves the `sort=True` cost to Pandas and is slower. Forcing pandas float groupby rolls single-key float `sum`/`mean`/`std`/`var` back to pandas semantics/performance.

Note: the Rust-side `sort=True` kernels allocate a permutation vector and perform an `O(G log G)` comparison sort over groups (G = number of groups). This can increase memory usage at very high cardinality.

Note: the benchmark runner defaults `PANDAS_BOOSTER_FORCE_PANDAS_SORT=0`.

ABI skew controls:

- `PANDAS_BOOSTER_STRICT_ABI` (default: OFF when unset):
  - truthy (`1/true/yes/on`, case-insensitive): treat detected ABI skew as a hard error (no fallback).
  - anything else (including `0/false/no/off`): fall back to pandas on detected ABI skew.
- `PANDAS_BOOSTER_ABI_SKEW_NOTICE` (default: ON when unset):
  - unset / truthy (`1/true/yes/on`, case-insensitive): enable ABI-skew warnings.
  - anything else (including `0/false/no/off`): disable ABI-skew warnings.

**Returns**: 
- Single key (`by="col"`): A `pd.Series` indexed by the unique keys.
- Multiple keys (`by=["col1", "col2"]`): A `pd.Series` with a `pd.MultiIndex`.

## Supported Operations

The following aggregation functions are currently supported:

| Operation | Description |
|-----------|-------------|
| `sum` | Sum of values in each group |
| `mean` | Arithmetic mean of values in each group |
| `std` | Sample standard deviation (ddof=1) |
| `var` | Sample variance (ddof=1) |
| `min` | Minimum value in each group |
| `max` | Maximum value in each group |
| `count` | Count of non-NaN values in each group |

### Acceleration Scope and Mandatory Fallback

To ensure predictable performance and correctness, the following rules define where Rust-first dispatch is certified.

#### Certified Rust Dispatch Domain (`std`/`var`)

| Feature | Supported (Rust-first) | Mandatory pandas Fallback |
|---------|-------------------------|--------------------------|
| Keys | Single or Multi-key (up to 10), integer dtypes | Non-integer keys, custom objects |
| Values | Numeric (`int64`, `float64`) | `uint64`, object, bool, datetime, category |
| Semantics | `ddof=1` (default) | Custom `ddof`, `numeric_only`, `skipna=False` |
| Dtypes | Primitive NumPy arrays | Extension dtypes (nullable `Int64`, `Float64`, `pd.NA`) |

#### Determinism and Precision

- **Determinism**: Accelerated float aggregations (`sum`, `mean`, `std`, `var`) are bitwise-identical across thread counts for identical inputs in the same environment.
- **Precision**: Results are semantically pandas-compatible but not guaranteed to be bit-for-bit identical to pandas. `std` and `var` follow this same policy.
- **Escape Hatch**: `PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY=1` forces pandas execution for single-key float-input `sum`/`mean`/`std`/`var` if bit-for-bit identity is required.

## Requirements and Constraints

To ensure correctness and performance, the following constraints apply:

- **Minimum dataset size**: 100,000 rows for legacy aggregations (`sum`, `mean`, `min`, `max`, `count`). For smaller datasets on these operations, the library automatically falls back to native Pandas. **Note**: Supported `std` and `var` operations are Rust-first by default within their certified dispatch domain regardless of dataset size.
- **Key column(s)**: Must be integer dtype (e.g., `int64`, `int32`). For multi-column groupby, all key columns must be integers. The accelerated path preserves Pandas' index dtype (e.g., `int32` on numpy-backend pandas).
- **Maximum key columns**: Up to 10 columns for multi-column groupby.
- **Value column**: Must be a numeric dtype (integers or floats).
- **Extension dtypes**: Pandas extension dtypes (e.g., nullable `Int64` / `Float64` using `pd.NA`) are not supported and will trigger a fallback to Pandas.
- **NaN handling**: `NaN` values in the target column are skipped in aggregations, matching standard Pandas behavior.
- **Determinism policy (single-key float `sum`/`mean`/`std`/`var`)**: For identical inputs in the same runtime environment, pandas-booster returns bitwise-identical results across thread counts. `NaN` inputs are skipped; all-`NaN` groups follow existing semantics (`sum -> +0.0`, `mean -> NaN`, `std/var -> NaN`). Compared with pandas, outputs may differ at the last-bit level (including `+0.0` vs `-0.0`) because pandas-booster uses an implementation-defined deterministic reduction order.
- **Return types**: Integer aggregations follow Pandas-style dtypes: `sum/min/max/count` return integer results; `mean/std/var` return `float64`. Standard deviation and variance always use `ddof=1`.

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
- **Profile evidence:** `--profile-json` writes internal single-key `std`/`var` phase timings for the Rust path (`local_build`, `merge`, `reorder`, `materialize`, and Python post-processing) so benchmark reports can separate kernel time from conversion and Series construction overhead.
- **Speedup baseline:** All speedup values (`x`) use **Pandas** as the baseline (1.0x) within each sort mode.
- **Optional Polars:** Polars is included in the benchmarks for comparison if installed. If not installed, the benchmark suite proceeds with Pandas vs Booster only.

### Standard Cardinality (5M rows)

| Operation | Groups | Sort | Type | Pandas | Polars | Booster |
|-----------|--------|------|------|--------|--------|---------|
| Single-key | 1,000 | True | Cold | 31.3±4.5ms (1.0x) | 22.2±5.5ms (**1.4x**) | 4.7±0.8ms (**6.6x**) |
|  |  |  | Warm | 23.9±0.9ms (1.0x) | 18.0±1.4ms (**1.3x**) | 2.4±0.5ms (**10.1x**) |
|  |  | False | Cold | 30.4±5.2ms (1.0x) | 23.7±8.3ms (**1.3x**) | 5.3±0.8ms (**5.7x**) |
|  |  |  | Warm | 22.6±1.0ms (1.0x) | 20.3±7.2ms (**1.1x**) | 2.7±0.4ms (**8.3x**) |
| 2-key | 5,000 | True | Cold | 84.4±6.1ms (1.0x) | 46.9±5.4ms (**1.8x**) | 31.0±3.2ms (**2.7x**) |
|  |  |  | Warm | 68.7±2.2ms (1.0x) | 38.8±2.6ms (**1.8x**) | 29.6±2.7ms (**2.3x**) |
|  |  | False | Cold | 70.7±7.4ms (1.0x) | 49.0±7.6ms (**1.4x**) | 32.0±4.3ms (**2.2x**) |
|  |  |  | Warm | 55.4±4.6ms (1.0x) | 56.0±41.4ms (1.0x) | 29.5±5.4ms (**1.9x**) |
| 3-key | 25,000 | True | Cold | 123.8±7.9ms (1.0x) | 59.2±3.7ms (**2.1x**) | 44.9±4.9ms (**2.8x**) |
|  |  |  | Warm | 108.6±3.6ms (1.0x) | 55.5±4.5ms (**2.0x**) | 39.5±2.4ms (**2.7x**) |
|  |  | False | Cold | 104.2±10.5ms (1.0x) | 67.5±14.3ms (**1.5x**) | 45.8±2.3ms (**2.3x**) |
|  |  |  | Warm | 89.9±3.7ms (1.0x) | 62.2±9.0ms (**1.4x**) | 43.3±1.9ms (**2.1x**) |
| 4-key | 100,000 | True | Cold | 151.7±15.7ms (1.0x) | 104.5±18.2ms (**1.5x**) | 56.3±3.1ms (**2.7x**) |
|  |  |  | Warm | 132.8±4.7ms (1.0x) | 86.9±6.4ms (**1.5x**) | 51.4±4.0ms (**2.6x**) |
|  |  | False | Cold | 123.1±12.6ms (1.0x) | 100.1±20.8ms (**1.2x**) | 55.4±2.1ms (**2.2x**) |
|  |  |  | Warm | 106.6±1.6ms (1.0x) | 89.0±5.6ms (**1.2x**) | 52.4±2.8ms (**2.0x**) |
| 5-key | 993,138 | True | Cold | 302.5±23.4ms (1.0x) | 241.1±36.1ms (**1.3x**) | 175.5±4.7ms (**1.7x**) |
|  |  |  | Warm | 305.1±14.7ms (1.0x) | 221.0±26.1ms (**1.4x**) | 167.2±4.5ms (**1.8x**) |
|  |  | False | Cold | 206.7±24.2ms (1.0x) | 173.2±15.6ms (**1.2x**) | 102.0±5.2ms (**2.0x**) |
|  |  |  | Warm | 169.4±5.2ms (1.0x) | 165.4±7.4ms (1.0x) | 93.0±2.6ms (**1.8x**) |

### High Cardinality (5M rows, ~5M unique groups)

| Operation | Groups | Sort | Type | Pandas | Polars | Booster |
|-----------|--------|------|------|--------|--------|---------|
| Single-key | 3,160,983 | True | Cold | 855.9±86.2ms (1.0x) | 135.6±11.6ms (**6.3x**) | 246.7±22.9ms (**3.5x**) |
|  |  |  | Warm | 785.4±70.2ms (1.0x) | 130.1±24.0ms (**6.0x**) | 227.7±14.3ms (**3.5x**) |
|  |  | False | Cold | 231.9±8.0ms (1.0x) | 173.4±3.1ms (**1.3x**) | 225.2±14.4ms (1.0x) |
|  |  |  | Warm | 224.8±3.7ms (1.0x) | 176.8±8.6ms (**1.3x**) | 199.9±4.1ms (**1.1x**) |
| 2-key | 4,532,339 | True | Cold | 879.0±23.3ms (1.0x) | 261.6±21.2ms (**3.4x**) | 369.4±25.4ms (**2.4x**) |
|  |  |  | Warm | 904.2±64.9ms (1.0x) | 241.7±24.6ms (**3.7x**) | 355.3±38.5ms (**2.5x**) |
|  |  | False | Cold | 375.9±8.7ms (1.0x) | 248.5±6.3ms (**1.5x**) | 170.1±7.3ms (**2.2x**) |
|  |  |  | Warm | 354.1±4.0ms (1.0x) | 245.1±11.0ms (**1.4x**) | 151.4±5.3ms (**2.3x**) |
| 3-key | 4,901,309 | True | Cold | 1004.5±95.8ms (1.0x) | 355.6±26.5ms (**2.8x**) | 541.5±16.1ms (**1.9x**) |
|  |  |  | Warm | 911.4±22.5ms (1.0x) | 334.1±18.6ms (**2.7x**) | 513.6±44.1ms (**1.8x**) |
|  |  | False | Cold | 399.6±16.6ms (1.0x) | 259.9±15.6ms (**1.5x**) | 220.7±11.8ms (**1.8x**) |
|  |  |  | Warm | 388.4±40.0ms (1.0x) | 252.4±5.3ms (**1.5x**) | 194.2±4.5ms (**2.0x**) |

**Performance characteristics**:
- **Single-key (standard cardinality)**: Warm state shows **8.3-10.1x** speedup over Pandas baseline (cold: **5.7-6.6x**). Booster also outperforms Polars on these single-key standard-cardinality runs.
- **Multi-key (standard cardinality)**: Booster is consistently faster than Pandas across 2-5 keys for both `sort=True` and `sort=False` (warm: **1.8-2.7x**, cold: **1.7-2.8x**), and also faster than Polars in these standard-cardinality multi-key runs.
- **High cardinality (~5M unique groups)**: With `sort=True`, Booster remains faster than Pandas (warm: **1.8-3.5x**), but Polars is faster on the sorted path. With `sort=False`, Booster ranges from near-parity on single-key (**1.1x**) to clear gains on multi-key (**2.0-2.3x** warm).
- **Sort overhead**: For Pandas, `sort=False` is a strong win on high-cardinality workloads (about **2.3-3.5x** warm) and still beneficial on standard multi-key (**1.2-1.8x** warm). For Booster, `sort=False` impact is workload-dependent: near parity (or slightly slower) on most standard cases, but much faster on high-cardinality multi-key (**~2.3-2.6x** warm) and on standard 5-key (**~1.8x** warm).

### Sorted vs Appearance-Ordered Results

By default, results are sorted by group keys to match Pandas `sort=True` output. Pass `sort=False` to preserve Pandas' appearance order (first-seen group order). Performance impact depends on workload (see tables above):

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

# Run default benchmarks (standard + high)
python benches/benchmark.py --samples 20 --output results.md

# Run only selected aggregation functions
python benches/benchmark.py --agg std --agg var --samples 20 --output results.md

# Save single-key std/var phase-profile evidence as JSON
python benches/benchmark.py --agg std --agg var --samples 20 --profile-json profile.json

# Include threshold diagnostics as well
python benches/benchmark.py --cardinality all --diagnostic threshold --sort-mode unsorted --samples 20 --output results.md
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

### Release

The release process is automated via the `publish.yml` workflow triggered by version tags.

1. **Preconditions**:
   - PyPI project exists.
   - [Trusted Publisher](https://docs.pypi.org/trusted-publishers/) is configured for `publish.yml`.
   - GitHub environment `pypi` is configured (if repository is protected).

2. **Operator Flow**:
   Ensure the environment is ready and push a new tag:
   ```bash
   python -m pip install --upgrade pip
   pip install "maturin>=1.13,<2.0"
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

The workflow builds cross-platform wheels and handles the PyPI upload automatically.

### Testing
Run the same local checks that CI runs:
```bash
source .venv/bin/activate

# Rust static validation
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test

# Python quality and release contract checks
basedpyright --project pyrightconfig.json
ruff check python tests scripts
python scripts/check_release_contract.py metadata
python scripts/check_release_contract.py workflow --file .github/workflows/publish.yml

# Build the extension and run the Python suite
maturin develop --release
pytest tests/ -v --strict-markers -m "not stress"

# Optional longer determinism lane, matching the CI stress job
pytest tests/test_sort_false_determinism.py -v --strict-markers -m stress
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
# Run default benchmarks (cardinality=all, diagnostic=none)
source .venv/bin/activate
python benches/benchmark.py

# Run full suite (core + diagnostics)
python benches/benchmark.py --cardinality all --diagnostic threshold --sort-mode unsorted

# Run only standard cardinality benchmarks
python benches/benchmark.py --cardinality standard

# Run only high cardinality benchmarks
python benches/benchmark.py --cardinality high

# Run only selected aggregation functions
python benches/benchmark.py --agg std --agg var
python benches/benchmark.py --agg min --agg max --cardinality high --sort-mode sorted

# Add threshold-neighborhood diagnostics (opt-in)
python benches/benchmark.py --diagnostic threshold --sort-mode unsorted

# Run only sorted or sort=False benchmarks
python benches/benchmark.py --sort-mode sorted
python benches/benchmark.py --sort-mode unsorted

# Combine options
python benches/benchmark.py --cardinality high --sort-mode sorted

# Save results to markdown file
python benches/benchmark.py --output results.md

# Save internal single-key std/var profile evidence to JSON
python benches/benchmark.py --agg std --agg var --profile-json profile.json

# Adjust sample count (applies to both cold and warm; default: 5)
python benches/benchmark.py --samples 20
```

Note: `--agg` is repeatable and filters the benchmark to only the selected aggregation functions.
If omitted, the current default behavior is preserved: the core performance tables benchmark `sum`,
while the extra single-key evidence section benchmarks `std` and `var`.

Note: `--profile-json` is an internal benchmark diagnostics output. It includes the same
single-key `std`/`var` evidence cases used by the Markdown report, plus phase breakdowns when a
Rust-only Booster profile hook is available. Cases that fall back to pandas or require Python
sorting remain in the JSON with `breakdown: null`.

Note: `--cardinality` is for workload classes (`standard`, `high`, `all`), while
`--diagnostic` is for internal boundary checks (`none`, `threshold`).

Note: `--diagnostic threshold` is only valid with `--sort-mode unsorted` because it targets
`sort=False` multi-key boundary behavior around `n_groups * n_keys ~= 200k`.

Breaking change (no compatibility mode): `--cardinality default` and
`--cardinality threshold` were removed in this version.

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
