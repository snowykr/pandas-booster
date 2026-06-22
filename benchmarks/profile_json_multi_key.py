from __future__ import annotations

import time
from typing import Any, cast

import numpy as np
import pandas as pd
from bench_utils import compute_stats
from dispatch import resolve_booster_benchmark_dispatch

REQUIRED_MULTI_KEY_SORTED_PHASES = (
    "hash_build_s",
    "partition_scatter_s",
    "partition_aggregation_s",
    "flatten_s",
    "sort_key_construction_s",
    "radix_sort_s",
    "sorted_materialization_s",
    "sort_first_permutation_s",
    "sort_first_segment_scan_s",
    "conversion_s",
    "pandas_index_construction_s",
)
MULTI_KEY_SORTED_HIGH_PRESET = "high_cardinality_3key"


def _profile_float_phase_samples() -> dict[str, list[float]]:
    return {phase_name: [] for phase_name in REQUIRED_MULTI_KEY_SORTED_PHASES} | {
        "rust_total_s": [],
        "python_total_s": [],
        "total_pipeline_s": [],
    }


def measure_booster_multi_key_sorted_breakdown(
    df: pd.DataFrame,
    key_cols: list[str],
    agg: str,
    sort: bool,
    n_samples: int,
) -> dict[str, Any] | None:
    import pandas_booster._abi_compat as abi_compat
    import pandas_booster._rust as rust
    from pandas_booster import _groupby_accel as groupby_accel

    if agg != "max" or not sort:
        return None

    val_col = cast(pd.Series, df["value"])
    key_series = [cast(pd.Series, df[col]) for col in key_cols]
    key_dtypes = [groupby_accel.capture_key_numpy_dtype(key_col) for key_col in key_series]
    value_dtype = groupby_accel.capture_value_numpy_dtype(val_col)
    is_val_int = pd.api.types.is_integer_dtype(val_col)
    dispatch = resolve_booster_benchmark_dispatch(df, key_cols, "value", agg, sort)
    rust_func = dispatch["rust_func"]
    if rust_func is None or bool(dispatch["needs_python_sort"]):
        return None

    profile_func_name = f"profile_{rust_func.__name__}"
    profile_func = getattr(rust, profile_func_name, None)
    if profile_func is None:
        return None

    phase_samples = _profile_float_phase_samples()
    partial_group_total = 0
    final_group_count = 0
    partial_to_final_ratio = 0.0
    route = "hash_first"
    sort_first_segment_scan_count = 0
    selected_sort_strategy = ""
    sort_key_bit_widths: list[int] = []

    for _ in range(n_samples):
        total_start = time.perf_counter()
        keys = [
            groupby_accel.to_i64_contiguous(key_col.to_numpy(copy=False))
            for key_col in key_series
        ]
        if is_val_int:
            values = np.ascontiguousarray(val_col.to_numpy(dtype=np.int64))
        else:
            values = np.ascontiguousarray(val_col.to_numpy(dtype=np.float64))

        result_keys, result_values, profile = profile_func(keys, values)
        conversion_start = time.perf_counter()
        result_values_arr = abi_compat.normalize_result_values(
            result_values,
            agg=agg,
            is_val_int=is_val_int,
            context="benchmark",
        )
        keys_cols = abi_compat.normalize_multi_keys_cols(
            result_keys,
            n_groups=result_values_arr.shape[0],
            n_keys=len(key_cols),
            context="benchmark",
        )
        conversion_s = float(profile["conversion_s"]) + (
            time.perf_counter() - conversion_start
        )

        index_start = time.perf_counter()
        _ = groupby_accel.build_series_from_multi_result(
            keys_cols,
            result_values_arr,
            by_cols=key_cols,
            key_dtypes=key_dtypes,
            name=val_col.name,
            value_dtype=value_dtype,
            agg=agg,
            is_val_int=is_val_int,
            sort=sort,
            needs_python_sort=False,
        )
        pandas_index_construction_s = time.perf_counter() - index_start
        for phase_name in REQUIRED_MULTI_KEY_SORTED_PHASES:
            if phase_name == "conversion_s":
                phase_samples[phase_name].append(conversion_s)
            elif phase_name == "pandas_index_construction_s":
                phase_samples[phase_name].append(pandas_index_construction_s)
            else:
                phase_samples[phase_name].append(float(profile[phase_name]))
        phase_samples["rust_total_s"].append(float(profile["rust_total_s"]))
        phase_samples["python_total_s"].append(conversion_s + pandas_index_construction_s)
        phase_samples["total_pipeline_s"].append(time.perf_counter() - total_start)
        partial_group_total = int(profile["partial_group_total"])
        final_group_count = int(profile["final_group_count"])
        partial_to_final_ratio = float(profile["partial_to_final_ratio"])
        route = str(profile["route"])
        sort_first_segment_scan_count = int(profile["sort_first_segment_scan_count"])
        selected_sort_strategy = str(profile["selected_sort_strategy"])
        sort_key_bit_widths = [int(width) for width in profile["sort_key_bit_widths"]]

    stats = {name: compute_stats(samples) for name, samples in phase_samples.items()}
    return {
        "profile_kind": "multi_key_sorted",
        "route": route,
        "execution": f"booster->rust.{profile_func_name}",
        "phases": {name: stats[name] for name in REQUIRED_MULTI_KEY_SORTED_PHASES},
        "rust_total_s": stats["rust_total_s"].mean,
        "python_total_s": stats["python_total_s"].mean,
        "total_pipeline_s": stats["total_pipeline_s"].mean,
        "partial_group_total": partial_group_total,
        "final_group_count": final_group_count,
        "partial_to_final_ratio": partial_to_final_ratio,
        "sort_first_segment_scan_count": sort_first_segment_scan_count,
        "selected_sort_strategy": selected_sort_strategy,
        "sort_key_bit_widths": sort_key_bit_widths,
    }
