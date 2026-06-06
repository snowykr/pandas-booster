from __future__ import annotations

from pathlib import Path
from typing import Final

_REPO_ROOT: Final = Path(__file__).resolve().parents[1]
_README: Final = _REPO_ROOT / "README.md"
_REPORTS_DIR: Final = _REPO_ROOT / "benchmarks" / "reports"
_BENCHMARK_TABLE_HEADER: Final = (
    "| Workload | Groups | Sort | Type | Pandas | Polars | Booster |"
)


def test_readme_benchmark_methodology_frames_results_as_relative_same_run_comparisons() -> None:
    # Given: the public README benchmark methodology.
    readme = _README.read_text(encoding="utf-8")

    # When: readers interpret the checked-in benchmark numbers.
    same_run_note = (
        "Benchmarks are relative comparisons against Pandas within the same "
        "benchmark run."
    )
    variability_note = (
        "Absolute timings vary by CPU, core count, memory bandwidth, software "
        "versions, and Rayon thread settings."
    )

    # Then: the README frames the numbers as relative same-run evidence.
    assert same_run_note in readme
    assert variability_note in readme


def test_readme_benchmark_environment_uses_report_provenance_not_machine_bullet() -> None:
    # Given: the public README benchmark reproduction section.
    readme = _README.read_text(encoding="utf-8")

    # When: readers need exact reproduction metadata.
    provenance_note = (
        "Full machine and software metadata belong with the generated "
        "benchmark reports under [`benchmarks/reports/`](benchmarks/reports/)."
    )

    # Then: README delegates detailed hardware/software provenance to reports.
    assert provenance_note in readme
    assert "- **Machine**:" not in readme
    assert "- **Host machine**:" not in readme


def test_readme_benchmark_table_headers_are_unchanged() -> None:
    # Given: generated benchmark report Markdown files.
    reports = sorted(path for path in _REPORTS_DIR.glob("[a-z]*.md"))

    # When: checking the existing benchmark table format contract.
    header_count = sum(
        report.read_text(encoding="utf-8").count(_BENCHMARK_TABLE_HEADER)
        for report in reports
    )

    # Then: every per-aggregation report keeps the two existing table headers.
    assert len(reports) == 9
    assert header_count == 18
