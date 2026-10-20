"""Output-aware evidence collection through the real CLI and serializers."""

from __future__ import annotations

import json

import pytest

from ._profile_json_helpers import make_multi_key_sorted_breakdown
from ._report_output_helpers import _make_breakdown, _make_result


@pytest.mark.parametrize("output_mode", ["markdown", "json", "combined"])
@pytest.mark.parametrize("cardinality", ["high", "all"])
@pytest.mark.parametrize("sort_mode", ["sorted", "all", "unsorted"])
@pytest.mark.parametrize("selected_aggs", [["max"], ["max", "std"]])
def test_cli_collects_only_consumed_profile_evidence(
    benchmark_module, monkeypatch, tmp_path, output_mode, cardinality, sort_mode, selected_aggs
):
    benchmark_requests = []
    dataset_requests = []
    breakdown_requests = []
    suite_requests = []
    sorts = benchmark_module.resolve_sorts(sort_mode)
    presets = ["high_cardinality_1key"]
    if cardinality == "all":
        presets.insert(0, "1key")
    results = [
        _make_result(benchmark_module, preset="high_cardinality_3key", agg=agg, sort=sort)
        for agg in selected_aggs
        for sort in sorts
    ]

    def fake_run_benchmarks(**kwargs):
        suite_requests.append(kwargs)
        return results

    def fake_benchmark_single(preset_name, *, agg, sort, n_samples, verify_correctness):
        assert n_samples == 1
        assert verify_correctness is True
        benchmark_requests.append((preset_name, agg, sort))
        return _make_result(benchmark_module, preset=preset_name, agg=agg, sort=sort)

    def fake_generate_multi_key_dataset(**config):
        dataset_requests.append(config)
        keys = {name: [0, 1] for name, _ in config["key_configs"]}
        return benchmark_module.pd.DataFrame({**keys, "value": [1.0, 2.0]})

    def fake_measure_breakdown(preset_name, agg, sort, n_samples):
        assert n_samples == 1
        breakdown_requests.append((preset_name, agg, sort))
        if preset_name == "high_cardinality_3key":
            return make_multi_key_sorted_breakdown(benchmark_module)
        return _make_breakdown(
            benchmark_module, execution=f"booster->rust.groupby_{agg}_f64_sorted"
        )

    monkeypatch.setattr(benchmark_module, "run_benchmarks", fake_run_benchmarks)
    monkeypatch.setattr(benchmark_module, "benchmark_single", fake_benchmark_single)
    monkeypatch.setattr(
        benchmark_module, "generate_multi_key_dataset", fake_generate_multi_key_dataset
    )
    monkeypatch.setattr(
        benchmark_module,
        "describe_booster_execution",
        lambda df, keys, value, agg, sort: f"booster->rust.groupby_{agg}_f64_sorted",
    )
    monkeypatch.setattr(
        benchmark_module, "measure_booster_single_key_breakdown", fake_measure_breakdown
    )
    args = [
        "benchmark.py",
        "--samples",
        "1",
        "--cardinality",
        cardinality,
        "--sort-mode",
        sort_mode,
    ]
    for agg in selected_aggs:
        args.extend(["--agg", agg])
    if output_mode != "json":
        args.extend(["--output", str(tmp_path / "reports")])
    if output_mode != "markdown":
        args.extend(["--profile-json", str(tmp_path / "profile.json")])
    monkeypatch.setattr(benchmark_module.sys, "argv", args)

    assert benchmark_module.main() == results
    assert len(suite_requests) == 1
    expected = []
    if "std" in selected_aggs:
        expected.extend((preset, "std", sort) for preset in presets for sort in sorts)
    if output_mode != "markdown" and sort_mode != "unsorted":
        expected.extend(
            (
                "high_cardinality_3key" if preset == "high_cardinality_1key" and sort else preset,
                "max",
                sort,
            )
            for preset in presets
            for sort in sorts
        )
    assert benchmark_requests == expected
    assert breakdown_requests == expected
    assert len(dataset_requests) == len(expected)

    if output_mode != "json":
        max_report = (tmp_path / "reports" / "max.md").read_text(encoding="utf-8")
        assert "High Cardinality" in max_report
        has_single_key_max = any(
            preset != "high_cardinality_3key" and agg == "max" for preset, agg, sort in expected
        )
        assert ("Booster conversion vs compute breakdown" in max_report) == has_single_key_max
        if "std" in selected_aggs:
            std_report = (tmp_path / "reports" / "std.md").read_text(encoding="utf-8")
            assert "Booster conversion vs compute breakdown" in std_report
            assert "booster->rust.groupby_std_f64_sorted" in std_report
    if output_mode != "markdown":
        payload = json.loads((tmp_path / "profile.json").read_text(encoding="utf-8"))
        assert len(payload["cases"]) == len(expected)
        expected_aggs = {agg for preset, agg, sort in expected}
        assert payload["metadata"]["selected_aggs"] == [
            agg for agg in selected_aggs if agg in expected_aggs
        ]
        if sort_mode != "unsorted":
            profile = payload["multi_key_sorted_high"]
            assert profile["preset"] == "high_cardinality_3key"
            assert profile["aggs"] == ["max"]
            assert profile["per_agg"]["max"]["selected_sort_strategy"] == "packed_u64"
            assert profile["per_agg"]["max"]["sort_key_bit_widths"] == [9, 9, 9]
        else:
            assert "multi_key_sorted_high" not in payload
