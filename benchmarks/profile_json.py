"""Benchmark profile JSON serialization."""

from __future__ import annotations

import time
from typing import Any, cast

import numpy as np
import pandas as pd
from bench_utils import compute_stats
from datasets import PRESETS, generate_multi_key_dataset
from dispatch import (
    HAS_POLARS,
    describe_booster_execution,
    resolve_booster_benchmark_dispatch,
)
from profile_json_payload import (
    build_profile_json_payload as build_profile_json_payload,
)
from profile_json_payload import (
    save_profile_json as save_profile_json,
)
from profile_json_payload import (
    serialize_phase_stats as serialize_phase_stats,
)
from profile_json_payload import (
    serialize_stats as serialize_stats,
)
from profile_json_payload import (
    stats_mean_map as stats_mean_map,
)
from profile_json_payload import (
    summarize_profile_cases as summarize_profile_cases,
)
from reporting import STATS_EVIDENCE_PRESETS
from runner import benchmark_single, resolve_sorts, resolve_stats_evidence_aggs


def stats_evidence_workload_label(preset_name: str) -> str:
    if preset_name == STATS_EVIDENCE_PRESETS["standard"]:
        return "standard"
    if preset_name == STATS_EVIDENCE_PRESETS["high"]:
        return "high"
    return preset_name


def measure_booster_single_key_breakdown(
    preset_name: str,
    agg: str,
    sort: bool,
    n_samples: int,
    *,
    ignore_force_pandas_sort: bool = False,
) -> dict[str, Any] | None:
    import pandas_booster._abi_compat as abi_compat
    import pandas_booster._rust as rust
    from pandas_booster import _groupby_accel as groupby_accel

    config = PRESETS[preset_name]
    df = generate_multi_key_dataset(**config)
    key_cols = [col for col, _ in config["key_configs"]]
    if len(key_cols) != 1:
        raise ValueError("Breakdown evidence only supports single-key presets")

    key_col = cast(pd.Series, df[key_cols[0]])
    val_col = cast(pd.Series, df["value"])
    key_dtype = groupby_accel.capture_key_numpy_dtype(key_col)
    value_dtype = groupby_accel.capture_value_numpy_dtype(val_col)
    is_val_int = pd.api.types.is_integer_dtype(val_col)
    dispatch = resolve_booster_benchmark_dispatch(
        df,
        key_cols,
        "value",
        agg,
        sort,
        ignore_force_pandas_sort=ignore_force_pandas_sort,
    )
    rust_func = dispatch["rust_func"]
    if rust_func is None:
        return None

    needs_python_sort = bool(dispatch["needs_python_sort"])
    if needs_python_sort and sort:
        return None

    profile_func_name = f"profile_{rust_func.__name__}"
    profile_func = getattr(rust, profile_func_name, None)
    if profile_func is None:
        return None

    phase_samples: dict[str, list[float]] = {
        "prepare_inputs_s": [],
        "local_build_s": [],
        "merge_s": [],
        "reorder_s": [],
        "materialize_s": [],
        "python_normalize_s": [],
        "python_series_build_s": [],
        "rust_total_s": [],
        "python_total_s": [],
        "total_pipeline_s": [],
    }
    partial_group_total = 0
    final_group_count = 0
    partial_to_final_ratio = 0.0

    for _ in range(n_samples):
        total_start = time.perf_counter()

        prepare_start = time.perf_counter()
        keys = groupby_accel.to_i64_contiguous(key_col.to_numpy(copy=False))
        if is_val_int:
            values = np.ascontiguousarray(val_col.to_numpy(dtype=np.int64))
        else:
            values = np.ascontiguousarray(val_col.to_numpy(dtype=np.float64))
        phase_samples["prepare_inputs_s"].append(time.perf_counter() - prepare_start)

        result_keys, result_values, profile = profile_func(keys, values)
        phase_samples["local_build_s"].append(float(profile["local_build_s"]))
        phase_samples["merge_s"].append(float(profile["merge_s"]))
        phase_samples["reorder_s"].append(float(profile["reorder_s"]))
        phase_samples["materialize_s"].append(float(profile["materialize_s"]))
        phase_samples["rust_total_s"].append(float(profile["rust_total_s"]))
        partial_group_total = int(profile["partial_group_total"])
        final_group_count = int(profile["final_group_count"])
        partial_to_final_ratio = float(profile["partial_to_final_ratio"])

        normalize_start = time.perf_counter()
        result_values_arr = abi_compat.normalize_result_values(
            result_values,
            agg=agg,
            is_val_int=is_val_int,
            context="benchmark",
        )
        normalize_duration = time.perf_counter() - normalize_start
        phase_samples["python_normalize_s"].append(normalize_duration)

        build_start = time.perf_counter()
        _ = groupby_accel.build_series_from_single_result(
            np.asarray(result_keys),
            result_values_arr,
            name=val_col.name,
            index_name=key_col.name,
            index_dtype=key_dtype,
            value_dtype=value_dtype,
            agg=agg,
            is_val_int=is_val_int,
            sort=sort,
            needs_python_sort=needs_python_sort,
        )
        build_duration = time.perf_counter() - build_start
        phase_samples["python_series_build_s"].append(build_duration)
        phase_samples["python_total_s"].append(normalize_duration + build_duration)
        phase_samples["total_pipeline_s"].append(time.perf_counter() - total_start)

    stats = {name: compute_stats(samples) for name, samples in phase_samples.items()}
    return {
        "execution": dispatch["execution"],
        "phases": stats,
        "rust_total_s": stats["rust_total_s"].mean,
        "python_total_s": stats["python_total_s"].mean,
        "total_pipeline_s": stats["total_pipeline_s"].mean,
        "partial_group_total": partial_group_total,
        "final_group_count": final_group_count,
        "partial_to_final_ratio": partial_to_final_ratio,
    }


def collect_stats_evidence(
    n_samples: int,
    cardinality: str,
    sort_mode: str,
    selected_aggs: list[str] | None = None,
    *,
    benchmark_single_func=benchmark_single,
    generate_multi_key_dataset_func=generate_multi_key_dataset,
    describe_booster_execution_func=describe_booster_execution,
    measure_booster_single_key_breakdown_func=measure_booster_single_key_breakdown,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    evidence_aggs = resolve_stats_evidence_aggs(selected_aggs)
    if not evidence_aggs:
        return evidence

    preset_names: list[str] = []
    if cardinality in {"all", "standard"}:
        preset_names.append(STATS_EVIDENCE_PRESETS["standard"])
    if cardinality in {"all", "high"}:
        preset_names.append(STATS_EVIDENCE_PRESETS["high"])

    sorts = resolve_sorts(sort_mode)

    for preset_name in preset_names:
        config = PRESETS[preset_name]
        key_cols = [col for col, _ in config["key_configs"]]
        workload = stats_evidence_workload_label(preset_name)
        for agg in evidence_aggs:
            for sort in sorts:
                result = benchmark_single_func(
                    preset_name,
                    agg=agg,
                    sort=sort,
                    n_samples=n_samples,
                    verify_correctness=True,
                )
                df = generate_multi_key_dataset_func(**config)
                execution = {
                    "pandas": f"pandas.groupby.{agg}",
                    "booster": describe_booster_execution_func(df, key_cols, "value", agg, sort),
                }
                if HAS_POLARS:
                    execution["polars"] = f"polars.group_by.agg({agg})"
                evidence.append(
                    {
                        "preset": preset_name,
                        "workload": workload,
                        "agg": agg,
                        "sort": sort,
                        "result": result,
                        "execution": execution,
                        "breakdown": measure_booster_single_key_breakdown_func(
                            preset_name,
                            agg,
                            sort,
                            n_samples,
                        ),
                    }
                )
    return evidence
