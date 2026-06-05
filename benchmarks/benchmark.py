"""Official benchmark script for pandas-booster with Cold/Warm measurement.

This module is the executable compatibility facade. Runtime internals live in
focused benchmark modules while existing script execution and imports continue
to use benchmarks/benchmark.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

import numpy as np
import pandas as pd

BENCHMARKS_DIR = Path(__file__).resolve().parent
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

from bench_utils import BenchmarkStats, compute_stats, run_cold_warm_benchmark  # noqa: E402
from cli import main as _cli_main  # noqa: E402
from datasets import PRESETS, generate_multi_key_dataset, get_dataset_info  # noqa: E402
from dispatch import (  # noqa: E402
    HAS_POLARS,
    build_polars_agg_expr,
    describe_booster_execution,
    pl,
    resolve_booster_benchmark_dispatch,
)
from profile_json import (  # noqa: E402
    collect_stats_evidence as _collect_stats_evidence,
)
from profile_json import (  # noqa: E402
    measure_booster_single_key_breakdown,
    stats_evidence_workload_label,
)
from profile_json_payload import (  # noqa: E402
    build_profile_json_payload,
    serialize_phase_stats,
    serialize_stats,
    stats_mean_map,
    summarize_profile_cases,
)
from profile_json_payload import (  # noqa: E402
    save_profile_json as _save_profile_json,
)
from reporting import (  # noqa: E402
    BACKEND_DISPLAY_ORDER,
    BENCHMARK_INDEX_FILENAME,
    BENCHMARK_REPORT_GENERATED_MARKER,
    CORE_BENCHMARK_AGG,
    STATS_EVIDENCE_AGGS,
    STATS_EVIDENCE_PRESETS,
    STATS_EVIDENCE_SORTS,
    SUPPORTED_AGGS,
    benchmark_report_filename,
    format_benchmark_document,
    format_benchmark_index,
    format_correctness_section,
    format_performance_section,
    is_generated_benchmark_report,
    ordered_result_aggs,
    render_generated_markdown,
    render_high_table,
    render_standard_table,
    render_stats_evidence_section,
    render_threshold_table,
    save_results_md,
    validate_report_output_conflicts,
)
from runner import (  # noqa: E402
    benchmark_single,
    benchmark_worker,
    resolve_core_aggs,
    resolve_core_presets,
    resolve_diagnostic_presets,
    resolve_selected_aggs,
    resolve_sorts,
    resolve_stats_evidence_aggs,
    run_benchmarks,
)

_COMPAT_PUBLIC_SURFACE = (
    argparse,
    json,
    os,
    time,
    TYPE_CHECKING,
    Callable,
    cast,
    np,
    pd,
    BenchmarkStats,
    compute_stats,
    run_cold_warm_benchmark,
    PRESETS,
    get_dataset_info,
    HAS_POLARS,
    build_polars_agg_expr,
    pl,
    resolve_booster_benchmark_dispatch,
    stats_evidence_workload_label,
    serialize_phase_stats,
    serialize_stats,
    stats_mean_map,
    summarize_profile_cases,
    resolve_core_aggs,
    resolve_core_presets,
    resolve_diagnostic_presets,
    resolve_selected_aggs,
    resolve_sorts,
    resolve_stats_evidence_aggs,
)

__all__ = [
    "BACKEND_DISPLAY_ORDER",
    "BENCHMARK_INDEX_FILENAME",
    "BENCHMARK_REPORT_GENERATED_MARKER",
    "CORE_BENCHMARK_AGG",
    "STATS_EVIDENCE_AGGS",
    "STATS_EVIDENCE_PRESETS",
    "STATS_EVIDENCE_SORTS",
    "SUPPORTED_AGGS",
    "benchmark_report_filename",
    "format_benchmark_document",
    "format_benchmark_index",
    "format_correctness_section",
    "format_performance_section",
    "is_generated_benchmark_report",
    "ordered_result_aggs",
    "render_generated_markdown",
    "render_high_table",
    "render_standard_table",
    "render_stats_evidence_section",
    "render_threshold_table",
    "save_results_md",
    "validate_report_output_conflicts",
]


def collect_stats_evidence(
    n_samples: int,
    cardinality: str,
    sort_mode: str,
    selected_aggs: list[str] | None = None,
):
    return _collect_stats_evidence(
        n_samples,
        cardinality,
        sort_mode,
        selected_aggs,
        benchmark_single_func=benchmark_single,
        generate_multi_key_dataset_func=generate_multi_key_dataset,
        describe_booster_execution_func=describe_booster_execution,
        measure_booster_single_key_breakdown_func=measure_booster_single_key_breakdown,
    )


def save_profile_json(
    evidence: list[dict[str, Any]],
    output_path: str,
    *,
    cardinality: str,
    sort_mode: str,
    n_samples: int,
    selected_aggs: list[str] | None,
) -> None:
    _save_profile_json(
        evidence,
        output_path,
        cardinality=cardinality,
        sort_mode=sort_mode,
        n_samples=n_samples,
        selected_aggs=selected_aggs,
        build_profile_json_payload_func=build_profile_json_payload,
    )


def main():
    return _cli_main(
        run_benchmarks_func=run_benchmarks,
        benchmark_worker_func=benchmark_worker,
        collect_stats_evidence_func=collect_stats_evidence,
        save_results_md_func=save_results_md,
        save_profile_json_func=save_profile_json,
    )


if __name__ == "__main__":
    main()
