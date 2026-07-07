"""Benchmark CLI command and public contract tests."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest

from ._report_output_helpers import (
    _BENCHMARK_PATH,
    _REPORTING_PATH,
    _loaded_benchmark_module,
    _make_result,
)


@pytest.fixture(scope="module")
def benchmark_module():
    with _loaded_benchmark_module() as module:
        yield module


def test_loading_benchmark_module_restores_sys_path():
    before = list(sys.path)
    benchmark_dir = str(_BENCHMARK_PATH.parent)

    with _loaded_benchmark_module():
        assert sys.path[0] == benchmark_dir

    assert sys.path == before


def test_benchmark_module_loads_repo_local_reporting_module():
    with _loaded_benchmark_module() as module:
        reporting = sys.modules["reporting"]

        assert isinstance(reporting, ModuleType)
        assert reporting.__file__ is not None
        assert Path(reporting.__file__).resolve() == _REPORTING_PATH
        assert module.SUPPORTED_AGGS is reporting.SUPPORTED_AGGS
        assert module.save_results_md is reporting.save_results_md


def test_reporting_module_imports_directly_from_benchmarks_dir():
    sys_path_snapshot = list(sys.path)
    try:
        sys.path.insert(0, str(_BENCHMARK_PATH.parent))
        sys.modules.pop("reporting", None)

        reporting = importlib.import_module("reporting")

        assert reporting.__file__ is not None
        assert Path(reporting.__file__).resolve() == _REPORTING_PATH
        assert reporting.benchmark_report_filename("sum") == "sum.md"
    finally:
        sys.modules.pop("reporting", None)
        sys.path[:] = sys_path_snapshot


def test_benchmark_worker_rejects_unknown_backend_before_dataset_generation(
    benchmark_module,
    monkeypatch,
):
    def fail_generate_dataset(**kwargs):
        _ = kwargs
        raise AssertionError("generate_multi_key_dataset should not run for invalid backend")

    monkeypatch.setattr(benchmark_module, "generate_multi_key_dataset", fail_generate_dataset)

    with pytest.raises(ValueError, match="Unsupported benchmark backend: 'unknown'"):
        benchmark_module.benchmark_worker("1key", "unknown")


def test_resolve_core_aggs_preserves_default_behavior(benchmark_module):
    assert benchmark_module.resolve_core_aggs(None) == ["sum"]


def test_resolve_selected_aggs_dedupes_and_preserves_order(benchmark_module):
    assert benchmark_module.resolve_selected_aggs(["std", "var", "std"]) == ["std", "var"]


def test_resolve_stats_evidence_aggs_includes_explicit_median(benchmark_module):
    resolve = benchmark_module.resolve_stats_evidence_aggs

    assert resolve(None) == ["std", "var"]
    assert resolve(["median"]) == ["median"]
    assert resolve(["std", "median", "var"]) == ["std", "median", "var"]
    assert resolve(["std", "min", "var"]) == ["std", "var"]
    assert resolve(["sum", "min"]) == []


def test_profile_json_help_mentions_selected_median_evidence(
    benchmark_module, monkeypatch, capsys
):
    monkeypatch.setattr(benchmark_module.sys, "argv", ["benchmark.py", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        benchmark_module.main()

    assert exc_info.value.code == 0
    assert "selected median evidence" in capsys.readouterr().out


def test_main_output_default_skips_unemitted_stats_evidence(
    benchmark_module, monkeypatch, tmp_path
):
    results = [_make_result(benchmark_module, preset="1key", agg="sum", sort=True)]
    captured_evidence_args: list[tuple[int, str, str, list[str] | None]] = []
    captured_save_evidence: list[list[dict[str, object]]] = []

    def fake_run_benchmarks(*, cardinality, diagnostic, sort_mode, n_samples, aggs):
        _ = (cardinality, diagnostic, sort_mode, n_samples, aggs)
        return results

    def fake_collect_stats_evidence(n_samples, cardinality, sort_mode, selected_aggs=None):
        captured_evidence_args.append((n_samples, cardinality, sort_mode, selected_aggs))
        return [{"agg": "std"}]

    def fake_save_results_md(
        results_arg, stats_evidence, output_path, sort_mode, cardinality, diagnostic
    ):
        _ = (results_arg, output_path, sort_mode, cardinality, diagnostic)
        captured_save_evidence.append(stats_evidence)

    monkeypatch.setattr(benchmark_module, "run_benchmarks", fake_run_benchmarks)
    monkeypatch.setattr(benchmark_module, "collect_stats_evidence", fake_collect_stats_evidence)
    monkeypatch.setattr(benchmark_module, "save_results_md", fake_save_results_md)
    monkeypatch.setattr(
        benchmark_module.sys,
        "argv",
        [
            "benchmark.py",
            "--cardinality",
            "standard",
            "--sort-mode",
            "sorted",
            "--samples",
            "3",
            "--output",
            str(tmp_path / "reports"),
        ],
    )

    assert benchmark_module.main() == results
    assert captured_evidence_args == []
    assert captured_save_evidence == [[]]


def test_main_output_selected_median_skips_profile_diagnostics(
    benchmark_module, monkeypatch, tmp_path
):
    results = [_make_result(benchmark_module, preset="1key", agg="median", sort=True)]
    captured_evidence_args: list[tuple[int, str, str, list[str] | None, bool]] = []

    def fake_run_benchmarks(*, cardinality, diagnostic, sort_mode, n_samples, aggs):
        _ = (cardinality, diagnostic, sort_mode, n_samples, aggs)
        return results

    def fake_collect_stats_evidence(
        n_samples,
        cardinality,
        sort_mode,
        selected_aggs=None,
        *,
        include_median_diagnostics=False,
    ):
        captured_evidence_args.append(
            (n_samples, cardinality, sort_mode, selected_aggs, include_median_diagnostics)
        )
        return [{"agg": "median"}]

    monkeypatch.setattr(benchmark_module, "run_benchmarks", fake_run_benchmarks)
    monkeypatch.setattr(benchmark_module, "collect_stats_evidence", fake_collect_stats_evidence)
    monkeypatch.setattr(benchmark_module, "save_results_md", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        benchmark_module.sys,
        "argv",
        [
            "benchmark.py",
            "--cardinality",
            "standard",
            "--sort-mode",
            "sorted",
            "--samples",
            "3",
            "--agg",
            "median",
            "--output",
            str(tmp_path / "reports"),
        ],
    )

    assert benchmark_module.main() == results
    assert captured_evidence_args == [(3, "standard", "sorted", ["median"], False)]


def test_prod_and_median_are_supported_benchmark_aggregations(benchmark_module):
    assert "prod" in benchmark_module.SUPPORTED_AGGS
    assert "median" in benchmark_module.SUPPORTED_AGGS
    assert benchmark_module.resolve_selected_aggs(["median", "sum", "median"]) == [
        "median",
        "sum",
    ]


def test_benchmark_worker_type_surface_mentions_median():
    source = (_BENCHMARK_PATH.parent / "runner.py").read_text()
    assert "agg: Literal[" in source
    assert '"median"' in source
    assert '"prod"' in source


def test_main_accepts_prod_agg(benchmark_module, monkeypatch):
    captured: list[dict[str, object]] = []
    results = [{"agg": "prod"}]

    def fake_run_benchmarks(**kwargs):
        captured.append(kwargs)
        return results

    monkeypatch.setattr(benchmark_module, "run_benchmarks", fake_run_benchmarks)
    monkeypatch.setattr(benchmark_module, "collect_stats_evidence", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        benchmark_module.sys,
        "argv",
        [
            "benchmark.py",
            "--cardinality",
            "standard",
            "--sort-mode",
            "sorted",
            "--samples",
            "1",
            "--agg",
            "prod",
        ],
    )

    assert benchmark_module.main() == results
    assert captured == [
        {
            "cardinality": "standard",
            "diagnostic": "none",
            "sort_mode": "sorted",
            "n_samples": 1,
            "aggs": ["prod"],
        }
    ]
