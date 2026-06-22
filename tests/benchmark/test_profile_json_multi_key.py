from __future__ import annotations

import pytest

from ._profile_json_helpers import (
    REQUIRED_MULTI_KEY_SORTED_PHASES,
    make_multi_key_sorted_breakdown,
    make_multi_key_sorted_result,
)


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
        "missing multi-key sorted profile JSON section for "
        "high_cardinality_3key/max/sort=True"
    )
    assert set(payload["multi_key_sorted_high"]["phases"]) == set(
        REQUIRED_MULTI_KEY_SORTED_PHASES
    )
    assert payload["multi_key_sorted_high"]["preset"] == "high_cardinality_3key"
    assert payload["multi_key_sorted_high"]["aggs"] == ["max"]
    assert payload["multi_key_sorted_high"]["route"] == "hash_first"
    assert payload["multi_key_sorted_high"]["sort_first_segment_scan_count"] == 0
    assert payload["multi_key_sorted_high"]["per_agg"]["max"]["route"] == "hash_first"
    assert payload["multi_key_sorted_high"]["per_agg"]["max"][
        "sort_first_segment_scan_count"
    ] == 0
    assert payload["multi_key_sorted_high"]["per_agg"]["max"][
        "selected_sort_strategy"
    ] == "packed_u64"
    assert payload["multi_key_sorted_high"]["per_agg"]["max"][
        "sort_key_bit_widths"
    ] == [9, 9, 9]
    case_breakdown = payload["cases"][0]["breakdown"]
    assert case_breakdown is not None
    assert case_breakdown["route"] == "hash_first"
    assert case_breakdown["sort_first_segment_scan_count"] == 0
    assert case_breakdown["selected_sort_strategy"] == "packed_u64"
    assert case_breakdown["sort_key_bit_widths"] == [9, 9, 9]
    assert payload["metadata"]["selected_aggs"] == ["max"]

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

def test_collect_stats_evidence_requests_multi_key_sorted_profile_for_high_max(
    benchmark_module,
    monkeypatch,
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
        key_columns = {
            col_name: [0, 1, 2] for col_name, _cardinality in kwargs["key_configs"]
        }
        return benchmark_module.pd.DataFrame({**key_columns, "value": [1.0, 2.0, 3.0]})

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
        selected_aggs=["max"],
    )

    assert breakdown_requests == [("high_cardinality_3key", "max", True, 1)], (
        "missing multi-key sorted profile collection contract: "
        "collect_stats_evidence must request high_cardinality_3key/max/sort=True breakdown"
    )
    assert len(evidence) == 1
    assert evidence[0]["breakdown"]["profile_kind"] == "multi_key_sorted"
