from __future__ import annotations

import json

import pytest

from ._profile_json_helpers import (
    REQUIRED_MULTI_KEY_SORTED_PHASES,
    make_multi_key_sorted_breakdown,
    make_multi_key_sorted_result,
)
from ._report_output_helpers import _make_breakdown, _make_result


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
        "result": make_multi_key_sorted_result(benchmark_module),
        "breakdown": make_multi_key_sorted_breakdown(benchmark_module),
    }

    payload = benchmark_module.build_profile_json_payload(
        [profiled_case],
        cardinality="high",
        sort_mode="sorted",
        n_samples=1,
        selected_aggs=["max"],
    )

    assert "multi_key_sorted_high" in payload, (
        "missing multi-key sorted profile JSON section for high_cardinality_3key/max/sort=True"
    )
    assert set(payload["multi_key_sorted_high"]["phases"]) == set(REQUIRED_MULTI_KEY_SORTED_PHASES)
    assert payload["multi_key_sorted_high"]["preset"] == "high_cardinality_3key"
    assert payload["multi_key_sorted_high"]["aggs"] == ["max"]
    assert payload["multi_key_sorted_high"]["route"] == "hash_first"
    assert payload["multi_key_sorted_high"]["sort_first_segment_scan_count"] == 0
    assert payload["multi_key_sorted_high"]["per_agg"]["max"]["route"] == "hash_first"
    assert payload["multi_key_sorted_high"]["per_agg"]["max"]["sort_first_segment_scan_count"] == 0
    assert (
        payload["multi_key_sorted_high"]["per_agg"]["max"]["selected_sort_strategy"] == "packed_u64"
    )
    assert payload["multi_key_sorted_high"]["per_agg"]["max"]["sort_key_bit_widths"] == [9, 9, 9]
    case_breakdown = payload["cases"][0]["breakdown"]
    assert case_breakdown is not None
    assert case_breakdown["route"] == "hash_first"
    assert case_breakdown["sort_first_segment_scan_count"] == 0
    assert case_breakdown["selected_sort_strategy"] == "packed_u64"
    assert case_breakdown["sort_key_bit_widths"] == [9, 9, 9]
    assert payload["metadata"]["selected_aggs"] == ["max"]
    for metadata in (
        case_breakdown,
        payload["multi_key_sorted_high"],
        payload["multi_key_sorted_high"]["per_agg"]["max"],
    ):
        assert metadata["route_reason"] == "low_cardinality"
        assert metadata["hash_first_aggregation_count"] == 1
        assert metadata["post_aggregation_sort_count"] == 1
        assert metadata["distinct_lower_bound"] == 900
        assert metadata["estimated_peak_bytes"] == 123456
        assert metadata["rayon_threads"] == 4


def test_mixed_profiles_preserve_kind_specific_phases_through_json(benchmark_module):
    cases = []
    for agg in ("std", "median", "max"):
        execution = f"booster->rust.groupby_{agg}_f64_sorted"
        breakdown = _make_breakdown(benchmark_module, execution=execution)
        result = _make_result(benchmark_module, preset="1key", agg=agg, sort=True)
        if agg == "std":
            del breakdown["phases"]["median_select_s"]
        if agg == "max":
            breakdown = make_multi_key_sorted_breakdown(benchmark_module)
            result = make_multi_key_sorted_result(benchmark_module)
            execution = "booster->rust.groupby_multi_max_f64_sorted"
            breakdown["phases"] = {
                name: benchmark_module.compute_stats([float(index)])
                for index, name in enumerate(REQUIRED_MULTI_KEY_SORTED_PHASES, start=1)
            }
        cases.append(
            {
                "preset": result["preset"],
                "workload": "high",
                "agg": agg,
                "sort": True,
                "execution": {
                    "pandas": f"pandas.groupby.{agg}",
                    "booster": execution,
                },
                "result": result,
                "breakdown": breakdown,
            }
        )

    payload = json.loads(
        json.dumps(
            benchmark_module.build_profile_json_payload(
                cases,
                cardinality="high",
                sort_mode="sorted",
                n_samples=1,
                selected_aggs=["median", "std", "max"],
            )
        )
    )

    expected = {
        name: float(index) for index, name in enumerate(REQUIRED_MULTI_KEY_SORTED_PHASES, start=1)
    }
    multi_case = payload["cases"][2]["breakdown"]
    assert {name: stats["mean"] for name, stats in multi_case["phases"].items()} == expected
    assert multi_case["phase_means"] == expected
    multi_summary = payload["multi_key_sorted_high"]
    assert multi_summary["phases"] == expected
    assert multi_summary["per_agg"]["max"]["phases"] == expected
    single_summary = payload["single_key_sorted_high"]
    assert single_summary["aggs"] == ["std", "median"]
    assert single_summary["phases"]["median_select_s"] == pytest.approx(0.05)
    assert single_summary["per_agg"]["std"]["phases"]["median_select_s"] == 0.0
    assert single_summary["per_agg"]["median"]["route_kind"] == "direct_dense"
    assert "hash_build_s" not in single_summary["phases"]
    assert "median_select_s" not in multi_summary["phases"]
    assert payload["metadata"]["selected_aggs"] == ["median", "std", "max"]

    # A mixed payload must still reject missing selected-median route evidence.
    del cases[1]["breakdown"]["route_kind"]
    with pytest.raises(ValueError, match="selected median sorted profile breakdown"):
        benchmark_module.build_profile_json_payload(
            cases,
            cardinality="high",
            sort_mode="sorted",
            n_samples=1,
            selected_aggs=["median", "std", "max"],
        )


def test_build_profile_json_payload_requires_explicit_multi_key_sorted_route(
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
        "result": make_multi_key_sorted_result(benchmark_module),
        "breakdown": make_multi_key_sorted_breakdown(benchmark_module),
    }
    del profiled_case["breakdown"]["route"]

    with pytest.raises(KeyError, match="route"):
        benchmark_module.build_profile_json_payload(
            [profiled_case],
            cardinality="high",
            sort_mode="sorted",
            n_samples=1,
            selected_aggs=["max"],
        )


@pytest.mark.parametrize("required_key", ["selected_sort_strategy", "sort_key_bit_widths"])
def test_build_profile_json_payload_requires_explicit_sort_strategy_evidence(
    benchmark_module,
    required_key,
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
        "result": make_multi_key_sorted_result(benchmark_module),
        "breakdown": make_multi_key_sorted_breakdown(benchmark_module),
    }
    del profiled_case["breakdown"][required_key]

    with pytest.raises(KeyError, match=required_key):
        benchmark_module.build_profile_json_payload(
            [profiled_case],
            cardinality="high",
            sort_mode="sorted",
            n_samples=1,
            selected_aggs=["max"],
        )


@pytest.mark.parametrize("agg", ["max", "count"])
def test_collect_stats_evidence_requests_multi_key_sorted_profile_for_high_reducer(
    benchmark_module,
    monkeypatch,
    agg,
):
    breakdown_requests: list[tuple[str, str, bool, int]] = []

    def fake_benchmark_single(preset_name, *, agg, sort, n_samples, verify_correctness):
        _ = (n_samples, verify_correctness)
        result = make_multi_key_sorted_result(benchmark_module)
        result["preset"] = preset_name
        result["agg"] = agg
        result["sort"] = sort
        return result

    def fake_generate_multi_key_dataset(**kwargs):
        key_columns = {col_name: [0, 1, 2] for col_name, _cardinality in kwargs["key_configs"]}
        return benchmark_module.pd.DataFrame({**key_columns, "value": [1.0, 2.0, 3.0]})

    def fake_describe_booster_execution(_df, key_cols, _value_col, requested_agg, sort):
        assert key_cols == ["k1", "k2", "k3"], (
            "missing multi-key sorted profile collection contract: "
            "high cardinality profile evidence must inspect the 3-key preset"
        )
        assert requested_agg == agg
        assert sort is True
        return f"booster->rust.groupby_multi_{agg}_f64_sorted"

    def fake_measure_breakdown(preset_name, requested_agg, sort, n_samples):
        breakdown_requests.append((preset_name, requested_agg, sort, n_samples))
        if (preset_name, requested_agg, sort) != ("high_cardinality_3key", agg, True):
            return None
        return make_multi_key_sorted_breakdown(benchmark_module)

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
        selected_aggs=[agg],
    )

    assert breakdown_requests == [("high_cardinality_3key", agg, True, 1)], (
        "missing multi-key sorted profile collection contract: "
        "collect_stats_evidence must request high_cardinality_3key/max/sort=True breakdown"
    )
    assert len(evidence) == 1
    assert evidence[0]["breakdown"]["profile_kind"] == "multi_key_sorted"
