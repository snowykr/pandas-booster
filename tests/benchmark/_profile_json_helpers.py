from __future__ import annotations

from ._report_output_helpers import _make_result

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


def make_multi_key_sorted_result(benchmark_module) -> dict:
    result = _make_result(
        benchmark_module,
        preset="high_cardinality_3key",
        agg="max",
        sort=True,
    )
    result.update(
        {
            "n_keys": 3,
            "key_cols": ["k1", "k2", "k3"],
            "combo_cardinality": 900,
            "group_ratio": 0.9,
        }
    )
    return result


def make_multi_key_sorted_breakdown(benchmark_module) -> dict:
    stats = benchmark_module.compute_stats([0.1])
    return {
        "profile_kind": "multi_key_sorted",
        "route": "hash_first",
        "execution": "booster->rust.profile_groupby_multi_max_f64_sorted",
        "phases": dict.fromkeys(REQUIRED_MULTI_KEY_SORTED_PHASES, stats),
        "rust_total_s": 0.7,
        "python_total_s": 0.2,
        "total_pipeline_s": 0.9,
        "partial_group_total": 900,
        "final_group_count": 900,
        "partial_to_final_ratio": 1.0,
        "sort_first_segment_scan_count": 0,
        "selected_sort_strategy": "packed_u64",
        "sort_key_bit_widths": [9, 9, 9],
    }
