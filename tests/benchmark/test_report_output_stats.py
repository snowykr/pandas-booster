"""Benchmark report stats-evidence rendering contract tests."""

from __future__ import annotations

from ._profile_json_helpers import (
    make_multi_key_sorted_breakdown,
    make_multi_key_sorted_result,
)
from ._report_output_helpers import _make_breakdown, _make_result


def test_render_stats_evidence_section_is_empty_when_no_selected_stats_aggs(benchmark_module):
    assert benchmark_module.render_stats_evidence_section([]) == ""


def test_render_stats_evidence_section_skips_unavailable_breakdowns(benchmark_module):
    rendered = benchmark_module.render_stats_evidence_section(
        [
            {
                "preset": "1key",
                "workload": "standard",
                "agg": "std",
                "sort": True,
                "execution": {
                    "pandas": "pandas.groupby.std",
                    "booster": "booster->pandas.groupby.std",
                },
                "result": _make_result(benchmark_module, preset="1key", agg="std", sort=True),
                "breakdown": None,
            }
        ]
    )

    assert "### Booster conversion vs compute breakdown" in rendered
    assert "booster->pandas.groupby.std" in rendered
    assert "No Rust-only Booster breakdown rows were available" in rendered


def test_render_stats_evidence_section_skips_multi_key_profile_breakdowns(benchmark_module):
    rendered = benchmark_module.render_stats_evidence_section(
        [
            {
                "preset": "high_cardinality_3key",
                "workload": "high_cardinality_3key",
                "agg": "max",
                "sort": True,
                "execution": {
                    "pandas": "pandas.groupby.max",
                    "booster": "booster->rust.profile_groupby_multi_max_f64_sorted",
                },
                "result": make_multi_key_sorted_result(benchmark_module),
                "breakdown": make_multi_key_sorted_breakdown(benchmark_module),
            }
        ]
    )

    assert rendered == ""


def test_render_stats_evidence_section_skips_unavailable_multi_key_profile_breakdowns(
    benchmark_module,
):
    rendered = benchmark_module.render_stats_evidence_section(
        [
            {
                "preset": "high_cardinality_3key",
                "workload": "high_cardinality_3key",
                "agg": "max",
                "sort": True,
                "execution": {
                    "pandas": "pandas.groupby.max",
                    "booster": "booster->pandas.groupby.max",
                },
                "result": make_multi_key_sorted_result(benchmark_module),
                "breakdown": None,
            }
        ]
    )

    assert rendered == ""


def test_render_stats_evidence_section_preserves_single_key_breakdown_with_multi_key_evidence(
    benchmark_module,
):
    rendered = benchmark_module.render_stats_evidence_section(
        [
            {
                "preset": "1key",
                "workload": "standard",
                "agg": "std",
                "sort": True,
                "execution": {
                    "pandas": "pandas.groupby.std",
                    "booster": "booster->rust.groupby_std_f64_sorted",
                },
                "result": _make_result(benchmark_module, preset="1key", agg="std", sort=True),
                "breakdown": _make_breakdown(
                    benchmark_module,
                    execution="booster->rust.groupby_std_f64_sorted",
                ),
            },
            {
                "preset": "high_cardinality_3key",
                "workload": "high_cardinality_3key",
                "agg": "max",
                "sort": True,
                "execution": {
                    "pandas": "pandas.groupby.max",
                    "booster": "booster->rust.profile_groupby_multi_max_f64_sorted",
                },
                "result": make_multi_key_sorted_result(benchmark_module),
                "breakdown": make_multi_key_sorted_breakdown(benchmark_module),
            },
        ]
    )

    assert "| standard | `std` | True | `booster->rust.groupby_std_f64_sorted` |" in rendered
    assert "booster->rust.profile_groupby_multi_max_f64_sorted" not in rendered
    assert (
        "| high_cardinality_3key | `max` | True | "
        "`booster->rust.profile_groupby_multi_max_f64_sorted` |"
        not in rendered
    )
    assert "No Rust-only Booster breakdown rows were available" not in rendered
