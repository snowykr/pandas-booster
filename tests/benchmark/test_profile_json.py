"""Benchmark profile JSON tests."""

from __future__ import annotations

import json
import sys
import types

import pytest

from ._report_output_helpers import (
    _loaded_benchmark_module,
    _make_breakdown,
    _make_result,
)

_REQUIRED_MULTI_KEY_SORTED_PHASES = (
    "hash_build_s",
    "partition_scatter_s",
    "partition_aggregation_s",
    "flatten_s",
    "sort_key_construction_s",
    "radix_sort_s",
    "sorted_materialization_s",
    "sort_first_permutation_s",
    "sort_first_segment_scan_s",
    "conversion_s",
    "pandas_index_construction_s",
)


def _make_multi_key_sorted_result(benchmark_module) -> dict:
    result = _make_result(
        benchmark_module,
        preset="high_cardinality_3key",
        agg="max",
        sort=True,
    )
    result.update(
        {
            "n_keys": 3,
            "key_cols": ["k1", "k2", "k3"],
            "combo_cardinality": 900,
            "group_ratio": 0.9,
        }
    )
    return result


def _make_multi_key_sorted_breakdown(benchmark_module) -> dict:
    stats = benchmark_module.compute_stats([0.1])
    return {
        "profile_kind": "multi_key_sorted",
        "route": "sort_first",
        "execution": "booster->rust.profile_groupby_multi_max_f64_sorted",
        "phases": dict.fromkeys(_REQUIRED_MULTI_KEY_SORTED_PHASES, stats),
        "rust_total_s": 0.7,
        "python_total_s": 0.2,
        "total_pipeline_s": 0.9,
        "partial_group_total": 900,
        "final_group_count": 900,
        "partial_to_final_ratio": 1.0,
        "sort_first_segment_scan_count": 1,
    }


@pytest.fixture(scope="module")
def benchmark_module():
    with _loaded_benchmark_module() as module:
        yield module


def test_collect_stats_evidence_uses_actual_force_pandas_sort_setting(
    benchmark_module,
    monkeypatch,
):
    captured_execution_flags: list[bool] = []
    captured_breakdown_flags: list[bool] = []

    monkeypatch.setattr(
        benchmark_module,
        "benchmark_single",
        lambda *args, **kwargs: _make_result(benchmark_module, preset="1key", agg="std", sort=True),
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
    monkeypatch.setitem(sys.modules, "pandas_booster._rust", types.SimpleNamespace())
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


def test_build_profile_json_payload_exposes_multi_key_sorted_profile_section(
    benchmark_module,
):
    profiled_case = {
        "preset": "high_cardinality_3key",
        "workload": "high",
        "agg": "max",
        "sort": True,
        "execution": {
            "pandas": "pandas.groupby.max",
            "booster": "booster->rust.groupby_multi_max_f64_sorted",
        },
        "result": _make_multi_key_sorted_result(benchmark_module),
        "breakdown": _make_multi_key_sorted_breakdown(benchmark_module),
    }

    payload = benchmark_module.build_profile_json_payload(
        [profiled_case],
        cardinality="high",
        sort_mode="sorted",
        n_samples=1,
        selected_aggs=["max"],
    )

    assert "multi_key_sorted_high" in payload, (
        "missing multi-key sorted profile JSON section for "
        "high_cardinality_3key/max/sort=True"
    )
    assert set(payload["multi_key_sorted_high"]["phases"]) == set(
        _REQUIRED_MULTI_KEY_SORTED_PHASES
    )
    assert payload["multi_key_sorted_high"]["preset"] == "high_cardinality_3key"
    assert payload["multi_key_sorted_high"]["aggs"] == ["max"]
    assert payload["multi_key_sorted_high"]["route"] == "sort_first"
    assert payload["multi_key_sorted_high"]["per_agg"]["max"]["route"] == "sort_first"
    assert payload["metadata"]["selected_aggs"] == ["max"]


def test_collect_stats_evidence_requests_multi_key_sorted_profile_for_high_max(
    benchmark_module,
    monkeypatch,
):
    breakdown_requests: list[tuple[str, str, bool, int]] = []

    def fake_benchmark_single(preset_name, *, agg, sort, n_samples, verify_correctness):
        _ = (n_samples, verify_correctness)
        result = _make_multi_key_sorted_result(benchmark_module)
        result["preset"] = preset_name
        result["agg"] = agg
        result["sort"] = sort
        return result

    def fake_generate_multi_key_dataset(**kwargs):
        key_columns = {
            col_name: [0, 1, 2]
            for col_name, _cardinality in kwargs["key_configs"]
        }
        return benchmark_module.pd.DataFrame(
            {**key_columns, "value": [1.0, 2.0, 3.0]}
        )

    def fake_describe_booster_execution(_df, key_cols, _value_col, agg, sort):
        assert key_cols == ["k1", "k2", "k3"], (
            "missing multi-key sorted profile collection contract: "
            "high cardinality profile evidence must inspect the 3-key preset"
        )
        assert agg == "max"
        assert sort is True
        return "booster->rust.groupby_multi_max_f64_sorted"

    def fake_measure_breakdown(preset_name, agg, sort, n_samples):
        breakdown_requests.append((preset_name, agg, sort, n_samples))
        if (preset_name, agg, sort) != ("high_cardinality_3key", "max", True):
            return None
        return _make_multi_key_sorted_breakdown(benchmark_module)

    monkeypatch.setattr(benchmark_module, "benchmark_single", fake_benchmark_single)
    monkeypatch.setattr(
        benchmark_module, "generate_multi_key_dataset", fake_generate_multi_key_dataset
    )
    monkeypatch.setattr(
        benchmark_module, "describe_booster_execution", fake_describe_booster_execution
    )
    monkeypatch.setattr(
        benchmark_module,
        "measure_booster_single_key_breakdown",
        fake_measure_breakdown,
    )

    evidence = benchmark_module.collect_stats_evidence(
        n_samples=1,
        cardinality="high",
        sort_mode="sorted",
        selected_aggs=["max"],
    )

    assert breakdown_requests == [("high_cardinality_3key", "max", True, 1)], (
        "missing multi-key sorted profile collection contract: "
        "collect_stats_evidence must request high_cardinality_3key/max/sort=True breakdown"
    )
    assert len(evidence) == 1
    assert evidence[0]["breakdown"]["profile_kind"] == "multi_key_sorted"


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
    assert payload["cases"][0]["breakdown"]["execution"] == ("booster->rust.groupby_std_f64_sorted")
