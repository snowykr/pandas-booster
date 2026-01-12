"""Dataset generation module for reproducible benchmarks.

This module provides consistent, reproducible dataset generation for benchmarking
pandas-booster's multi-key groupby operations.

Usage:
    from datasets import generate_multi_key_dataset, PRESETS

    # Use a preset
    df = generate_multi_key_dataset(**PRESETS["readme_3key"])

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
    "readme_1key": {
        "n_rows": 5_000_000,
        "key_configs": [("key", 1000)],
        "value_dtype": "float64",
        "seed": 42,
    },
    # Two-key groupby
    "readme_2key": {
        "n_rows": 5_000_000,
        "key_configs": [("region", 50), ("category", 100)],
        "value_dtype": "float64",
        "seed": 42,
    },
    # Three-key groupby (the problematic case)
    "readme_3key": {
        "n_rows": 5_000_000,
        "key_configs": [("region", 50), ("category", 100), ("year", 5)],
        "value_dtype": "float64",
        "seed": 42,
    },
    # Four-key groupby
    "readme_4key": {
        "n_rows": 5_000_000,
        "key_configs": [("region", 50), ("category", 100), ("year", 5), ("quarter", 4)],
        "value_dtype": "float64",
        "seed": 42,
    },
    # Five-key groupby (beyond SmallVec inline limit)
    "readme_5key": {
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
}


def generate_multi_key_dataset(
    n_rows: int,
    key_configs: list[tuple[str, int]],
    value_dtype: Literal["float64", "int64"] = "float64",
    seed: int = 42,
    value_col_name: str = "value",
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
        data[col_name] = np.random.randint(0, n_unique, size=n_rows, dtype=np.int64)

    # Generate value column
    if value_dtype == "float64":
        data[value_col_name] = np.random.random(size=n_rows) * 1000
    else:
        data[value_col_name] = np.random.randint(0, 10000, size=n_rows, dtype=np.int64)

    return pd.DataFrame(data)


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
