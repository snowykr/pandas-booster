"""Benchmark report generation contract tests."""

from __future__ import annotations

from ._report_output_helpers import (
    _BENCHMARK_PATH,
    _GENERATED_BENCHMARK_MARKER,
    _expected_benchmark_report_aggs,
    _generated_markdown,
    _loaded_generate_benchmark_docs_module,
    _make_breakdown,
    _make_result,
)


def test_committed_benchmark_reports_match_generate_docs_contract(benchmark_module):
    reports_dir = _BENCHMARK_PATH.parent / "reports"
    expected_aggs = _expected_benchmark_report_aggs()

    with _loaded_generate_benchmark_docs_module() as module:
        assert expected_aggs == module.SUPPORTED_AGGS

    expected_files = {"README.md", *(f"{agg}.md" for agg in expected_aggs)}
    actual_files = {path.name for path in reports_dir.glob("*.md")}
    assert actual_files == expected_files

    index = (reports_dir / "README.md").read_text(encoding="utf-8")
    expected_index = _generated_markdown(
        benchmark_module.format_benchmark_index(list(expected_aggs))
    )
    expected_table, _expected_environment = expected_index.split(
        "## Environment & Configuration",
        maxsplit=1,
    )
    actual_table, actual_environment = index.split(
        "## Environment & Configuration",
        maxsplit=1,
    )
    assert actual_table == expected_table
    assert "## Environment & Configuration" in index
    assert "replace these values with the environment used for that run" in index
    assert "- **Machine**: MacBook Pro (`Mac15,6`)" in actual_environment
    assert "- **Pandas**: 2.3.3" in actual_environment
    assert "- **Polars**: 1.40.1" in actual_environment

    for agg in expected_aggs:
        report = (reports_dir / f"{agg}.md").read_text(encoding="utf-8")
        assert report.startswith(f"{_GENERATED_BENCHMARK_MARKER}\n\n")
        assert "## Performance" in report
        assert "### Correctness" in report
        assert "Performance characteristics" not in report

def test_format_performance_section_separates_multiple_aggs(benchmark_module):
    results = [
        _make_result(benchmark_module, preset="1key", agg="sum", sort=True),
        _make_result(benchmark_module, preset="1key", agg="std", sort=True),
    ]

    rendered = benchmark_module.format_performance_section(
        results,
        sort_mode="sorted",
        cardinality="standard",
        diagnostic="none",
    )

    assert "### Aggregation: `sum`" in rendered
    assert "### Aggregation: `std`" in rendered
    assert "| Workload | Groups | Sort | Type | Pandas | Polars | Booster |" in rendered
    assert rendered.count("| Single-key |") == 2
    assert "| `sum` | Single-key |" not in rendered
    assert "| `std` | Single-key |" not in rendered
    assert "|  |  |  | Warm |" in rendered

def test_save_results_md_writes_per_aggregation_reports(benchmark_module, tmp_path):
    results = [
        _make_result(benchmark_module, preset="1key", agg="sum", sort=True),
        _make_result(benchmark_module, preset="1key", agg="std", sort=True),
    ]
    evidence = [
        {
            "preset": "1key",
            "workload": "standard",
            "agg": "std",
            "sort": True,
            "execution": {
                "pandas": "pandas.groupby.std",
                "booster": "booster->rust.groupby_std_f64_sorted",
            },
            "result": results[1],
            "breakdown": _make_breakdown(benchmark_module),
        }
    ]
    output_dir = tmp_path / "reports.v1"

    benchmark_module.save_results_md(
        results,
        evidence,
        str(output_dir),
        sort_mode="sorted",
        cardinality="standard",
        diagnostic="none",
    )

    index = (output_dir / "README.md").read_text(encoding="utf-8")
    sum_report = (output_dir / "sum.md").read_text(encoding="utf-8")
    std_report = (output_dir / "std.md").read_text(encoding="utf-8")

    assert index.startswith(f"{_GENERATED_BENCHMARK_MARKER}\n\n")
    assert "| `sum` | [sum.md](sum.md) |" in index
    assert "| `std` | [std.md](std.md) |" in index
    assert sum_report.startswith(f"{_GENERATED_BENCHMARK_MARKER}\n\n")
    assert std_report.startswith(f"{_GENERATED_BENCHMARK_MARKER}\n\n")
    assert "| Single-key |" in sum_report
    assert "| `sum` | Single-key |" not in sum_report
    assert "| `std` | Single-key |" not in std_report
    assert "| Single-key |" in std_report
    assert "## Single-Key `std` Evidence" in std_report
