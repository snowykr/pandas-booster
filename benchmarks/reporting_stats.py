from __future__ import annotations

from typing import Any

from bench_utils import BenchmarkStats
from reporting_constants import BACKEND_DISPLAY_ORDER


def render_stats_evidence_section(evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return ""

    evidence_aggs = sorted({item["agg"] for item in evidence})
    if evidence_aggs == ["std", "var"]:
        heading = "## Single-Key `std`/`var` Evidence"
    elif len(evidence_aggs) == 1:
        heading = f"## Single-Key `{evidence_aggs[0]}` Evidence"
    else:
        agg_list = "/".join(f"`{agg}`" for agg in evidence_aggs)
        heading = f"## Single-Key {agg_list} Evidence"

    lines: list[str] = [
        heading,
        "",
        _stats_evidence_summary(evidence_aggs),
        _stats_overhead_summary(),
        "",
        "| Workload | Agg | Sort | Backend | Execution | Cold | Warm |",
        "|----------|-----|------|---------|-----------|------|------|",
    ]

    for item in evidence:
        _append_backend_evidence_rows(lines, item)

    lines.extend(["", "### Booster conversion vs compute breakdown", ""])
    lines.append(
        "The table below isolates the Rust-first Booster path for the same single-key datasets. "
        "`local_build`, `merge`, `reorder`, and `materialize` come from the internal Rust "
        "profiling hook, while the Python-side phases measure post-kernel normalization and "
        "final pandas Series assembly."
    )
    lines.append("")
    _append_breakdown_table(lines, evidence)
    lines.append("")
    return "\n".join(lines)


def _stats_evidence_summary(evidence_aggs: list[str]) -> str:
    agg_phrase = (
        "`std` and `var`"
        if evidence_aggs == ["std", "var"]
        else ", ".join(f"`{agg}`" for agg in evidence_aggs)
    )
    return (
        f"{agg_phrase} scale on the Rust-first path because each worker accumulates "
        "mergeable `(count, mean, m2)` state and Rayon only merges those partial states at the end."
    )


def _stats_overhead_summary() -> str:
    return (
        "When the dataset is smaller or the group count is low, the fixed overhead from NumPy "
        "contiguity/copy work, the Python↔Rust boundary, and Series materialization eats into "
        "that gain even though the kernel itself remains parallel."
    )


def _append_backend_evidence_rows(lines: list[str], item: dict[str, Any]) -> None:
    agg = item["agg"]
    sort = "True" if item["sort"] else "False"
    result = item["result"]
    execution = item["execution"]
    for backend_name in BACKEND_DISPLAY_ORDER:
        backend_data = result["backends"].get(backend_name)
        if backend_data is None:
            continue
        cold_stats: BenchmarkStats = backend_data["cold_stats"]
        warm_stats: BenchmarkStats = backend_data["warm_stats"]
        row = [
            item["workload"],
            f"`{agg}`",
            sort,
            backend_name,
            f"`{execution[backend_name]}`",
            cold_stats.format_ms(2),
            warm_stats.format_ms(2),
        ]
        lines.append("| " + " | ".join(row) + " |")


def _append_breakdown_table(lines: list[str], evidence: list[dict[str, Any]]) -> None:
    breakdown_columns = [
        "Workload",
        "Agg",
        "Sort",
        "Execution",
        "Prepare inputs",
        "Local build",
        "Merge",
        "Reorder",
        "Materialize",
        "Python normalize",
        "Series build",
        "Rust total",
        "Total pipeline",
        "Partial groups",
        "Final groups",
        "Partial/final",
    ]
    lines.append("| " + " | ".join(breakdown_columns) + " |")
    separator_row = "|" + "|".join(
        f"{'-' * len(column):-^{len(column) + 2}}" for column in breakdown_columns
    )
    lines.append(separator_row + "|")
    has_breakdown_rows = False
    for item in evidence:
        breakdown = item["breakdown"]
        if breakdown is None:
            continue
        has_breakdown_rows = True
        phases = breakdown["phases"]
        row = [
            item["workload"],
            f"`{item['agg']}`",
            "True" if item["sort"] else "False",
            f"`{breakdown['execution']}`",
            phases["prepare_inputs_s"].format_ms(2),
            phases["local_build_s"].format_ms(2),
            phases["merge_s"].format_ms(2),
            phases["reorder_s"].format_ms(2),
            phases["materialize_s"].format_ms(2),
            phases["python_normalize_s"].format_ms(2),
            phases["python_series_build_s"].format_ms(2),
            phases["rust_total_s"].format_ms(2),
            phases["total_pipeline_s"].format_ms(2),
            f"{breakdown['partial_group_total']:,}",
            f"{breakdown['final_group_count']:,}",
            f"{breakdown['partial_to_final_ratio']:.3f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    if not has_breakdown_rows:
        lines.append(
            "No Rust-only Booster breakdown rows were available for the selected evidence cases."
        )
