"""Dataset generation module for reproducible benchmarks.

This module provides consistent, reproducible dataset generation for benchmarking
pandas-booster's multi-key groupby operations.

Usage:
    from datasets import generate_multi_key_dataset, PRESETS

    # Use a preset
    df = generate_multi_key_dataset(**PRESETS["3key"])

    # Or customize
    df = generate_multi_key_dataset(
        n_rows=5_000_000,
        key_configs=[("k1", 100), ("k2", 50), ("k3", 20)],
        seed=42,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from typing import Literal


# =============================================================================
# Dataset Presets for README Performance Table
# =============================================================================

PRESETS: dict[str, dict] = {
    # Single-key baseline (for comparison)
    "1key": {
        "n_rows": 5_000_000,
        "key_configs": [("key", 1000)],
        "value_dtype": "float64",
        "seed": 42,
    },
    # Two-key groupby
    "2key": {
        "n_rows": 5_000_000,
        "key_configs": [("region", 50), ("category", 100)],
        "value_dtype": "float64",
        "seed": 42,
    },
    # Three-key groupby
    "3key": {
        "n_rows": 5_000_000,
        "key_configs": [("region", 50), ("category", 100), ("year", 5)],
        "value_dtype": "float64",
        "seed": 42,
    },
    # Four-key groupby
    "4key": {
        "n_rows": 5_000_000,
        "key_configs": [("region", 50), ("category", 100), ("year", 5), ("quarter", 4)],
        "value_dtype": "float64",
        "seed": 42,
    },
    # Five-key groupby (beyond SmallVec inline limit)
    "5key": {
        "n_rows": 5_000_000,
        "key_configs": [
            ("region", 50),
            ("category", 100),
            ("year", 5),
            ("quarter", 4),
            ("channel", 10),
        ],
        "value_dtype": "float64",
        "seed": 42,
    },
    # High cardinality test (worst case for merge-based approach)
    "high_cardinality_1key": {
        "n_rows": 5_000_000,
        "key_configs": [("key", 5_000_000)],
        "value_dtype": "float64",
        "seed": 42,
    },
    "high_cardinality_2key": {
        "n_rows": 5_000_000,
        "key_configs": [("k1", 5000), ("k2", 5000)],
        "value_dtype": "float64",
        "seed": 42,
    },
    "high_cardinality_3key": {
        "n_rows": 5_000_000,
        "key_configs": [("k1", 500), ("k2", 500), ("k3", 500)],
        "value_dtype": "float64",
        "seed": 42,
    },
    # Low cardinality test (best case for merge-based approach)
    "low_cardinality_3key": {
        "n_rows": 5_000_000,
        "key_configs": [("k1", 10), ("k2", 10), ("k3", 10)],
        "value_dtype": "float64",
        "seed": 42,
    },
    # Quick benchmarks (smaller datasets for fast iteration)
    "quick_2key": {
        "n_rows": 1_000_000,
        "key_configs": [("k1", 100), ("k2", 50)],
        "value_dtype": "float64",
        "seed": 42,
    },
    "quick_3key": {
        "n_rows": 1_000_000,
        "key_configs": [("k1", 100), ("k2", 50), ("k3", 20)],
        "value_dtype": "float64",
        "seed": 42,
    },
    # Threshold-neighborhood cases for sort=False multi-key path.
    # Target total output elements ~= n_groups * n_keys around 200k.
    "threshold_180k": {
        "n_rows": 5_000_000,
        "key_configs": [("k1", 1000), ("k2", 90)],
        "value_dtype": "float64",
        "seed": 42,
    },
    "threshold_200k": {
        "n_rows": 5_000_000,
        "key_configs": [("k1", 1000), ("k2", 100)],
        "value_dtype": "float64",
        "seed": 42,
    },
    "threshold_220k": {
        "n_rows": 5_000_000,
        "key_configs": [("k1", 1000), ("k2", 110)],
        "value_dtype": "float64",
        "seed": 42,
    },
    # Median route diagnostics: keep dense, sparse, skewed, NaN-heavy, and
    # adversarial shapes separate so sorted single-key median reports cannot
    # collapse dense-coded wins into a generic median claim.
    "median_dense_1key_5m_1k": {
        "n_rows": 5_000_000,
        "key_configs": [("key", 1000)],
        "value_dtype": "float64",
        "seed": 42,
        "key_generation": "dense",
    },
    "median_sparse_gap_1key_5m_1k": {
        "n_rows": 5_000_000,
        "key_configs": [("key", 1000)],
        "value_dtype": "float64",
        "seed": 42,
        "key_generation": "sparse_gap",
        "key_gap": 1_000_000_000,
    },
    "median_sparse_gap_1key_5m_10k": {
        "n_rows": 5_000_000,
        "key_configs": [("key", 10_000)],
        "value_dtype": "float64",
        "seed": 42,
        "key_generation": "sparse_gap",
        "key_gap": 1_000_000_000,
    },
    "median_sparse_gap_1key_5m_50k": {
        "n_rows": 5_000_000,
        "key_configs": [("key", 50_000)],
        "value_dtype": "float64",
        "seed": 42,
        "key_generation": "sparse_gap",
        "key_gap": 1_000_000_000,
    },
    "median_near_unique_1key_5m": {
        "n_rows": 5_000_000,
        "key_configs": [("key", 5_000_000)],
        "value_dtype": "float64",
        "seed": 42,
        "key_generation": "dense",
    },
    "median_skewed_zipf_1key_5m": {
        "n_rows": 5_000_000,
        "key_configs": [("key", 10_000)],
        "value_dtype": "float64",
        "seed": 42,
        "key_generation": "zipf",
    },
    "median_skewed_dominant_1key_5m": {
        "n_rows": 5_000_000,
        "key_configs": [("key", 10_000)],
        "value_dtype": "float64",
        "seed": 42,
        "key_generation": "dominant_tail",
    },
    "median_nan_dense_1key_5m_1k_p0": {
        "n_rows": 5_000_000,
        "key_configs": [("key", 1000)],
        "value_dtype": "float64",
        "seed": 42,
        "nan_rate": 0.0,
        "key_generation": "dense",
    },
    "median_nan_dense_1key_5m_1k_p50": {
        "n_rows": 5_000_000,
        "key_configs": [("key", 1000)],
        "value_dtype": "float64",
        "seed": 42,
        "nan_rate": 0.50,
        "key_generation": "dense",
    },
    "median_nan_dense_1key_5m_1k_p95": {
        "n_rows": 5_000_000,
        "key_configs": [("key", 1000)],
        "value_dtype": "float64",
        "seed": 42,
        "nan_rate": 0.95,
        "key_generation": "dense",
    },
    "median_nan_dense_1key_5m_1k_p100": {
        "n_rows": 5_000_000,
        "key_configs": [("key", 1000)],
        "value_dtype": "float64",
        "seed": 42,
        "nan_rate": 1.0,
        "key_generation": "dense",
    },
    "median_boundary_rows_100k_1k": {
        "n_rows": 100_000,
        "key_configs": [("key", 1000)],
        "value_dtype": "float64",
        "seed": 42,
    },
    "median_boundary_rows_300k_1k": {
        "n_rows": 300_000,
        "key_configs": [("key", 1000)],
        "value_dtype": "float64",
        "seed": 42,
    },
    "median_boundary_rows_1m_1k": {
        "n_rows": 1_000_000,
        "key_configs": [("key", 1000)],
        "value_dtype": "float64",
        "seed": 42,
    },
    "median_negative_huge_sparse_1key": {
        "n_rows": 5_000_000,
        "key_configs": [("key", 1000)],
        "value_dtype": "float64",
        "seed": 42,
        "key_generation": "negative_huge_sparse",
        "key_gap": 1_000_000_000,
    },
    "median_false_low_sample_tail_unique": {
        "n_rows": 5_000_000,
        "key_configs": [("key", 500_000)],
        "value_dtype": "float64",
        "seed": 42,
        "key_generation": "false_low_sample_tail_unique",
    },
}


def generate_multi_key_dataset(
    n_rows: int,
    key_configs: list[tuple[str, int]],
    value_dtype: Literal["float64", "int64"] = "float64",
    seed: int = 42,
    value_col_name: str = "value",
    key_generation: Literal[
        "dense",
        "sparse_gap",
        "zipf",
        "dominant_tail",
        "negative_huge_sparse",
        "false_low_sample_tail_unique",
    ] = "dense",
    key_gap: int = 1,
    nan_rate: float = 0.0,
) -> pd.DataFrame:
    """Generate a reproducible dataset for multi-key groupby benchmarks.

    Args:
        n_rows: Number of rows in the dataset.
        key_configs: List of (column_name, n_unique) tuples defining key columns.
            Each key column will have values from 0 to n_unique-1.
        value_dtype: Data type for the value column ("float64" or "int64").
        seed: Random seed for reproducibility.
        value_col_name: Name of the value column.

    Returns:
        DataFrame with key columns and a value column.

    Example:
        >>> df = generate_multi_key_dataset(
        ...     n_rows=1_000_000,
        ...     key_configs=[("region", 50), ("category", 100)],
        ...     seed=42,
        ... )
        >>> df.shape
        (1000000, 3)
        >>> df.columns.tolist()
        ['region', 'category', 'value']
    """
    np.random.seed(seed)

    data = {}

    # Generate key columns
    for col_name, n_unique in key_configs:
        data[col_name] = _generate_key_column(
            n_rows=n_rows,
            n_unique=n_unique,
            key_generation=key_generation,
            key_gap=key_gap,
        )

    # Generate value column
    if value_dtype == "float64":
        values = np.random.random(size=n_rows) * 1000
        if nan_rate:
            nan_mask = np.random.random(size=n_rows) < nan_rate
            values[nan_mask] = np.nan
        data[value_col_name] = values
    else:
        data[value_col_name] = np.random.randint(0, 10000, size=n_rows, dtype=np.int64)

    return pd.DataFrame(data)


def _generate_key_column(
    *,
    n_rows: int,
    n_unique: int,
    key_generation: str,
    key_gap: int,
) -> np.ndarray:
    if key_generation == "dense":
        return np.random.randint(0, n_unique, size=n_rows, dtype=np.int64)

    if key_generation == "sparse_gap":
        base = np.random.randint(0, n_unique, size=n_rows, dtype=np.int64)
        return base * np.int64(key_gap)

    if key_generation == "negative_huge_sparse":
        base = np.random.randint(0, n_unique, size=n_rows, dtype=np.int64)
        return base * np.int64(key_gap) - np.int64(key_gap * (n_unique // 2))

    if key_generation == "zipf":
        raw = np.random.zipf(1.25, size=n_rows).astype(np.int64)
        return (raw - 1) % np.int64(n_unique)

    if key_generation == "dominant_tail":
        dominant_mask = np.random.random(size=n_rows) < 0.90
        tail = np.random.randint(1, n_unique, size=n_rows, dtype=np.int64)
        return np.where(dominant_mask, np.int64(0), tail)

    if key_generation == "false_low_sample_tail_unique":
        keys = np.random.randint(0, min(n_unique, 1024), size=n_rows, dtype=np.int64)
        tail_start = n_rows // 2
        tail_len = n_rows - tail_start
        if tail_len > 0:
            keys[tail_start:] = np.arange(tail_len, dtype=np.int64) % np.int64(n_unique)
        return keys

    raise ValueError(f"Unsupported key_generation: {key_generation!r}")


def get_dataset_info(df: pd.DataFrame, key_cols: list[str]) -> dict:
    """Get metadata about a dataset for benchmark reporting.

    Args:
        df: The DataFrame to analyze.
        key_cols: List of key column names.

    Returns:
        Dictionary with dataset metadata.
    """
    combo_cardinality = df.groupby(key_cols).ngroups

    return {
        "n_rows": len(df),
        "n_keys": len(key_cols),
        "key_cols": key_cols,
        "key_cardinalities": {col: df[col].nunique() for col in key_cols},
        "combo_cardinality": combo_cardinality,
        "group_ratio": combo_cardinality / len(df),
    }


def list_presets() -> None:
    """Print available dataset presets."""
    print("Available dataset presets:")
    print("=" * 60)
    for name, config in PRESETS.items():
        n_keys = len(config["key_configs"])
        key_ranges = [f"{k}:{n}" for k, n in config["key_configs"]]
        print(
            f"  {name:25s} | {config['n_rows']:>10,} rows | {n_keys} keys | {', '.join(key_ranges)}"
        )
    print("=" * 60)


if __name__ == "__main__":
    list_presets()
