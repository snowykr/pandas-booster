from __future__ import annotations

import importlib.util
from pathlib import Path

_BENCHMARK_PATH = Path(__file__).resolve().parent.parent / "benches" / "benchmark.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_under_test", _BENCHMARK_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


def _make_result(*, preset: str, agg: str, sort: bool) -> dict:
    stats = benchmark.BenchmarkStats(mean=0.1, std=0.01, min=0.09, max=0.11, samples=[0.1])
    return {
        "preset": preset,
        "combo_cardinality": 100,
        "agg": agg,
        "sort": sort,
        "backends": {
            "pandas": {"cold_stats": stats, "warm_stats": stats},
            "booster": {"cold_stats": stats, "warm_stats": stats},
        },
    }


def test_resolve_core_aggs_preserves_default_behavior():
    assert benchmark.resolve_core_aggs(None) == ["sum"]


def test_resolve_selected_aggs_dedupes_and_preserves_order():
    assert benchmark.resolve_selected_aggs(["std", "var", "std"]) == ["std", "var"]


def test_resolve_stats_evidence_aggs_filters_to_std_var_only():
    assert benchmark.resolve_stats_evidence_aggs(None) == ["std", "var"]
    assert benchmark.resolve_stats_evidence_aggs(["std", "min", "var"]) == ["std", "var"]
    assert benchmark.resolve_stats_evidence_aggs(["sum", "min"]) == []


def test_format_performance_section_separates_multiple_aggs():
    results = [
        _make_result(preset="1key", agg="sum", sort=True),
        _make_result(preset="1key", agg="std", sort=True),
    ]

    rendered = benchmark.format_performance_section(
        results,
        sort_mode="sorted",
        cardinality="standard",
        diagnostic="none",
    )

    assert "### Aggregation: `sum`" in rendered
    assert "### Aggregation: `std`" in rendered
    assert rendered.count("| Single-key |") == 2


def test_render_stats_evidence_section_is_empty_when_no_selected_stats_aggs():
    assert benchmark.render_stats_evidence_section([]) == ""
