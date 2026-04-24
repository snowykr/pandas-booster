from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import pytest

_BENCHMARK_PATH = Path(__file__).resolve().parent.parent / "benches" / "benchmark.py"


@contextmanager
def _loaded_benchmark_module() -> Generator[ModuleType, None, None]:
    sys_path_snapshot = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location("benchmark_under_test", _BENCHMARK_PATH)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = sys_path_snapshot


@pytest.fixture(scope="module")
def benchmark_module():
    with _loaded_benchmark_module() as module:
        yield module


def _make_result(benchmark_module, *, preset: str, agg: str, sort: bool) -> dict:
    stats = benchmark_module.BenchmarkStats(
        mean=0.1, std=0.01, min=0.09, max=0.11, samples=[0.1]
    )
    return {
        "preset": preset,
        "n_rows": 1_000,
        "n_keys": 1,
        "key_cols": ["key"],
        "combo_cardinality": 100,
        "group_ratio": 0.1,
        "agg": agg,
        "sort": sort,
        "backends": {
            "pandas": {
                "cold_stats": stats,
                "warm_stats": stats,
                "cold_correctness": "not_checked",
                "warm_correctness": "not_checked",
            },
            "booster": {
                "cold_stats": stats,
                "warm_stats": stats,
                "cold_correctness": "pass",
                "warm_correctness": "pass",
            },
        },
    }


def _make_breakdown(
    benchmark_module, *, execution: str = "booster->rust.groupby_std_f64_sorted"
) -> dict:
    stats = benchmark_module.compute_stats([0.1])
    return {
        "execution": execution,
        "phases": {
            "prepare_inputs_s": stats,
            "local_build_s": stats,
            "merge_s": stats,
            "reorder_s": stats,
            "materialize_s": stats,
            "python_normalize_s": stats,
            "python_series_build_s": stats,
            "rust_total_s": stats,
            "python_total_s": stats,
            "total_pipeline_s": stats,
        },
        "rust_total_s": 0.1,
        "python_total_s": 0.1,
        "total_pipeline_s": 0.1,
        "partial_group_total": 1,
        "final_group_count": 1,
        "partial_to_final_ratio": 1.0,
    }


def test_loading_benchmark_module_restores_sys_path():
    before = list(sys.path)
    benchmark_dir = str(_BENCHMARK_PATH.parent)

    with _loaded_benchmark_module():
        assert sys.path[0] == benchmark_dir

    assert sys.path == before


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


def test_resolve_stats_evidence_aggs_filters_to_std_var_only(benchmark_module):
    assert benchmark_module.resolve_stats_evidence_aggs(None) == ["std", "var"]
    assert benchmark_module.resolve_stats_evidence_aggs(["std", "min", "var"]) == ["std", "var"]
    assert benchmark_module.resolve_stats_evidence_aggs(["sum", "min"]) == []


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
    assert rendered.count("| Single-key |") == 2


def test_render_stats_evidence_section_is_empty_when_no_selected_stats_aggs(benchmark_module):
    assert benchmark_module.render_stats_evidence_section([]) == ""


def test_collect_stats_evidence_uses_actual_force_pandas_sort_setting(
    benchmark_module,
    monkeypatch,
):
    captured_execution_flags: list[bool] = []
    captured_breakdown_flags: list[bool] = []

    monkeypatch.setattr(
        benchmark_module,
        "benchmark_single",
        lambda *args, **kwargs: _make_result(
            benchmark_module, preset="1key", agg="std", sort=True
        ),
    )
    monkeypatch.setattr(
        benchmark_module,
        "generate_multi_key_dataset",
        lambda **kwargs: benchmark_module.pd.DataFrame({"key": [1, 2], "value": [1.0, 2.0]}),
    )

    def fake_describe_booster_execution(
        _df, _key_cols, _value_col, _agg, _sort, *, ignore_force_pandas_sort=False
    ):
        captured_execution_flags.append(ignore_force_pandas_sort)
        return "booster->rust.groupby_std_f64_sorted"

    def fake_measure_breakdown(
        _preset_name, _agg, _sort, _n_samples, *, ignore_force_pandas_sort=False
    ):
        captured_breakdown_flags.append(ignore_force_pandas_sort)
        return _make_breakdown(benchmark_module)

    monkeypatch.setattr(
        benchmark_module, "describe_booster_execution", fake_describe_booster_execution
    )
    monkeypatch.setattr(
        benchmark_module, "measure_booster_single_key_breakdown", fake_measure_breakdown
    )

    evidence = benchmark_module.collect_stats_evidence(
        n_samples=1,
        cardinality="standard",
        sort_mode="sorted",
        selected_aggs=["std"],
    )

    assert len(evidence) == 1
    assert captured_execution_flags == [False]
    assert captured_breakdown_flags == [False]


def test_measure_booster_single_key_breakdown_returns_none_when_float_rollback_forces_pandas(
    benchmark_module,
    monkeypatch,
):
    import pandas_booster._groupby_accel as groupby_accel

    monkeypatch.setenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", "1")
    monkeypatch.setattr(
        benchmark_module,
        "generate_multi_key_dataset",
        lambda **kwargs: benchmark_module.pd.DataFrame(
            {"key": [1, 2, 1, 2], "value": [1.0, 2.0, 3.0, 4.0]}
        ),
    )

    def fail_select(*args, **kwargs):
        _ = (args, kwargs)
        raise AssertionError(
            "Rust kernel selection should not run when float rollback forces pandas"
        )

    monkeypatch.setattr(groupby_accel, "select_rust_groupby_func", fail_select)

    assert benchmark_module.measure_booster_single_key_breakdown("1key", "std", True, 1) is None


def test_describe_booster_execution_uses_pandas_label_for_large_single_float_sum_mean_rollback(
    benchmark_module,
    monkeypatch,
):
    import pandas_booster._groupby_accel as groupby_accel

    monkeypatch.setenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", "1")

    def fail_select(*args, **kwargs):
        _ = (args, kwargs)
        raise AssertionError(
            "Rust kernel selection should not run when float sum/mean rollback forces pandas"
        )

    monkeypatch.setattr(groupby_accel, "select_rust_groupby_func", fail_select)

    df = benchmark_module.pd.DataFrame(
        {
            "key": benchmark_module.np.tile([1, 2], 50_001),
            "value": benchmark_module.np.linspace(
                0.0, 1.0, 100_002, dtype=benchmark_module.np.float64
            ),
        }
    )

    assert benchmark_module.describe_booster_execution(df, ["key"], "value", "sum", True) == (
        "booster->pandas.groupby.sum"
    )
    assert benchmark_module.describe_booster_execution(df, ["key"], "value", "mean", False) == (
        "booster->pandas.groupby.mean"
    )


def test_build_profile_json_payload_handles_unavailable_breakdowns(benchmark_module):
    profiled_case = {
        "preset": "1key",
        "workload": "standard",
        "agg": "std",
        "sort": True,
        "execution": {
            "pandas": "pandas.groupby.std",
            "booster": "booster->rust.groupby_std_f64_sorted",
        },
        "result": _make_result(benchmark_module, preset="1key", agg="std", sort=True),
        "breakdown": _make_breakdown(benchmark_module),
    }
    fallback_case = {
        "preset": "1key",
        "workload": "standard",
        "agg": "var",
        "sort": True,
        "execution": {"pandas": "pandas.groupby.var", "booster": "booster->pandas.groupby.var"},
        "result": _make_result(benchmark_module, preset="1key", agg="var", sort=True),
        "breakdown": None,
    }

    payload = benchmark_module.build_profile_json_payload(
        [profiled_case, fallback_case],
        cardinality="standard",
        sort_mode="sorted",
        n_samples=1,
        selected_aggs=["std", "var"],
    )

    assert payload["cases"][1]["breakdown"] is None
    assert payload["single_key_sorted_standard"]["aggs"] == ["std"]


def test_save_profile_json_adds_json_suffix_and_creates_parent_dir(
    benchmark_module,
    monkeypatch,
    tmp_path,
):
    payload = {"ok": True}
    captured_args: list[tuple[list[dict[str, object]], dict[str, object]]] = []

    def fake_build_profile_json_payload(
        evidence,
        *,
        cardinality,
        sort_mode,
        n_samples,
        selected_aggs,
    ):
        captured_args.append(
            (
                evidence,
                {
                    "cardinality": cardinality,
                    "sort_mode": sort_mode,
                    "n_samples": n_samples,
                    "selected_aggs": selected_aggs,
                },
            )
        )
        return payload

    monkeypatch.setattr(
        benchmark_module, "build_profile_json_payload", fake_build_profile_json_payload
    )

    output_path = tmp_path / "profiles" / "profile_output"
    benchmark_module.save_profile_json(
        [{"preset": "1key"}],
        str(output_path),
        cardinality="standard",
        sort_mode="sorted",
        n_samples=3,
        selected_aggs=["std"],
    )

    written_path = output_path.with_suffix(".json")
    assert written_path.exists()
    assert written_path.parent.is_dir()
    assert json.loads(written_path.read_text()) == payload
    assert captured_args == [
        (
            [{"preset": "1key"}],
            {
                "cardinality": "standard",
                "sort_mode": "sorted",
                "n_samples": 3,
                "selected_aggs": ["std"],
            },
        )
    ]


def test_main_profile_json_wires_evidence_collection_and_file_write(
    benchmark_module, monkeypatch, tmp_path
):
    results = [_make_result(benchmark_module, preset="1key", agg="std", sort=True)]
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
            "result": results[0],
            "breakdown": _make_breakdown(benchmark_module),
        }
    ]
    captured_run_args: list[dict[str, object]] = []
    captured_evidence_args: list[tuple[int, str, str, list[str] | None]] = []

    def fake_run_benchmarks(*, cardinality, diagnostic, sort_mode, n_samples, aggs):
        captured_run_args.append(
            {
                "cardinality": cardinality,
                "diagnostic": diagnostic,
                "sort_mode": sort_mode,
                "n_samples": n_samples,
                "aggs": aggs,
            }
        )
        return results

    def fake_collect_stats_evidence(n_samples, cardinality, sort_mode, selected_aggs=None):
        captured_evidence_args.append((n_samples, cardinality, sort_mode, selected_aggs))
        return evidence

    profile_path = tmp_path / "profiles" / "std_profile"
    monkeypatch.setattr(benchmark_module, "run_benchmarks", fake_run_benchmarks)
    monkeypatch.setattr(benchmark_module, "collect_stats_evidence", fake_collect_stats_evidence)
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
            "std",
            "--profile-json",
            str(profile_path),
        ],
    )

    assert benchmark_module.main() == results

    written_path = profile_path.with_suffix(".json")
    payload = json.loads(written_path.read_text())
    assert captured_run_args == [
        {
            "cardinality": "standard",
            "diagnostic": "none",
            "sort_mode": "sorted",
            "n_samples": 3,
            "aggs": ["std"],
        }
    ]
    assert captured_evidence_args == [(3, "standard", "sorted", ["std"])]
    assert payload["metadata"] == {
        "cardinality": "standard",
        "sort_mode": "sorted",
        "samples": 3,
        "selected_aggs": ["std"],
    }
    assert payload["cases"][0]["breakdown"]["execution"] == (
        "booster->rust.groupby_std_f64_sorted"
    )


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
