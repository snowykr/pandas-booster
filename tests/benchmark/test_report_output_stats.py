"""Benchmark report stats-evidence rendering contract tests."""

from __future__ import annotations

from ._report_output_helpers import _make_result


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
