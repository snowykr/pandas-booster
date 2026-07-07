"""Benchmark report stats-evidence rendering contract tests."""

from __future__ import annotations

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


def test_render_stats_evidence_section_handles_selected_median_profile(benchmark_module):
    rendered = benchmark_module.render_stats_evidence_section(
        [
            {
                "preset": "1key",
                "workload": "standard",
                "agg": "median",
                "sort": True,
                "execution": {
                    "pandas": "pandas.groupby.median",
                    "booster": "booster->rust.groupby_median_f64_sorted",
                },
                "result": _make_result(
                    benchmark_module,
                    preset="1key",
                    agg="median",
                    sort=True,
                ),
                "breakdown": _make_breakdown(
                    benchmark_module,
                    execution="booster->rust.groupby_median_f64_sorted",
                ),
            }
        ]
    )

    assert "## Single-Key `median` Evidence" in rendered
    assert "`median` profile evidence is emitted only when selected" in rendered
    assert "booster->rust.groupby_median_f64_sorted" in rendered
    assert "Unique build" in rendered
    assert "Buffer setup" in rendered
    assert "Median select" in rendered
    assert "direct median phases" in rendered
    assert "`count_s` is zero when accepted-value counting is fused" in rendered
    assert "mergeable `(count, mean, m2)` state" not in rendered
