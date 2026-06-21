from __future__ import annotations

from typing import Any

from reporting_constants import BACKEND_DISPLAY_ORDER, BENCHMARK_REPORT_ENVIRONMENT_TEMPLATE_LINES
from reporting_stats import render_stats_evidence_section
from reporting_tables import render_high_table, render_standard_table, render_threshold_table


def format_performance_section(
    results: list[dict],
    sort_mode: str,
    cardinality: str,
    diagnostic: str,
) -> str:
    standard_presets = {"1key", "2key", "3key", "4key", "5key"}
    high_presets = {"high_cardinality_1key", "high_cardinality_2key", "high_cardinality_3key"}
    threshold_presets = {"threshold_180k", "threshold_200k", "threshold_220k"}
    filtered_results = [
        r
        for r in results
        if sort_mode == "all"
        or (sort_mode == "sorted" and r["sort"])
        or (sort_mode == "unsorted" and not r["sort"])
    ]
    sections = ["## Performance", ""]
    aggs = ordered_result_aggs(filtered_results)
    multiple_aggs = len(aggs) > 1

    for agg in aggs:
        agg_results = [r for r in filtered_results if r["agg"] == agg]
        standard_results = [r for r in agg_results if r["preset"] in standard_presets]
        high_results = [r for r in agg_results if r["preset"] in high_presets]
        threshold_results = [r for r in agg_results if r["preset"] in threshold_presets]
        if multiple_aggs:
            sections.extend([f"### Aggregation: `{agg}`", ""])
        _append_table_section(
            sections,
            cardinality in ["all", "standard"] and bool(standard_results),
            "### Standard Cardinality (5M rows)",
            "#### Standard Cardinality (5M rows)",
            multiple_aggs,
            render_standard_table(standard_results),
        )
        _append_table_section(
            sections,
            cardinality in ["all", "high"] and bool(high_results),
            "### High Cardinality (5M rows, ~5M unique groups)",
            "#### High Cardinality (5M rows, ~5M unique groups)",
            multiple_aggs,
            render_high_table(high_results),
        )
        if diagnostic == "threshold" and threshold_results:
            sections.extend(
                [
                    "### Diagnostics" if not multiple_aggs else "#### Diagnostics",
                    "",
                    (
                        "#### Threshold Neighborhood (2-key, n_groups * n_keys near 200k)"
                        if not multiple_aggs
                        else "##### Threshold Neighborhood (2-key, n_groups * n_keys near 200k)"
                    ),
                    "",
                    render_threshold_table(threshold_results),
                    "",
                ]
            )

    return "\n".join(sections)


def ordered_result_aggs(results: list[dict]) -> list[str]:
    aggs: list[str] = []
    for result in results:
        agg = result["agg"]
        if agg not in aggs:
            aggs.append(agg)
    return aggs


def benchmark_report_filename(agg: str) -> str:
    return f"{agg}.md"


def format_correctness_section(results: list[dict]) -> str:
    lines = ["### Correctness", ""]
    for backend in BACKEND_DISPLAY_ORDER:
        if backend == "pandas":
            continue
        if backend == "polars" and not any("polars" in r.get("backends", {}) for r in results):
            continue
        lines.append(f"- {backend.capitalize()}: {_summarize_backend(results, backend)}")
    lines.append("")
    return "\n".join(lines)


def format_benchmark_document(
    results: list[dict],
    stats_evidence: list[dict[str, Any]],
    sort_mode: str,
    cardinality: str,
    diagnostic: str,
) -> str:
    sections = [
        format_performance_section(results, sort_mode, cardinality, diagnostic),
        format_correctness_section(results),
    ]
    stats_evidence_section = render_stats_evidence_section(stats_evidence)
    if stats_evidence_section:
        sections.append(stats_evidence_section)
    return "\n\n".join(sections)


def format_benchmark_index(aggs: list[str]) -> str:
    lines = [
        "# Benchmark Reports",
        "",
        "Each report contains benchmark tables for one aggregation function.",
        "",
        "| Aggregation | Report |",
        "|-------------|--------|",
    ]
    for agg in aggs:
        filename = benchmark_report_filename(agg)
        lines.append(f"| `{agg}` | [{filename}]({filename}) |")
    lines.extend(["", *BENCHMARK_REPORT_ENVIRONMENT_TEMPLATE_LINES, ""])
    return "\n".join(lines)


def _append_table_section(
    sections: list[str],
    should_append: bool,
    single_heading: str,
    multi_heading: str,
    multiple_aggs: bool,
    table: str,
) -> None:
    if should_append:
        sections.extend([multi_heading if multiple_aggs else single_heading, "", table, ""])


def _summarize_backend(results: list[dict], name: str) -> str:
    failures: list[str] = []
    checked = 0
    passed = 0
    for r in results:
        backend_data = r.get("backends", {}).get(name)
        if not backend_data:
            continue
        for mode_label in ("cold", "warm"):
            status = backend_data.get(f"{mode_label}_correctness", "not_checked")
            if status == "not_checked":
                continue
            checked += 1
            if status.startswith("fail:"):
                failures.append(f"{r['preset']} (sort={r['sort']}, {mode_label}): {status}")
            else:
                passed += 1
    if checked == 0:
        return "not_checked"
    if not failures:
        return f"pass ({passed}/{checked})"
    first = failures[0]
    more = f" (+{len(failures) - 1} more)" if len(failures) > 1 else ""
    return f"fail ({passed}/{checked} passed): {first}{more}"
