from __future__ import annotations

import pytest
from conftest import _loaded_benchmark_module

from ._report_output_helpers import _make_breakdown, _make_result


@pytest.fixture(scope="module")
def benchmark_module():
    with _loaded_benchmark_module() as module:
        yield module


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


def test_profile_payload_phase_summary_uses_ordered_union_defaults(benchmark_module):
    legacy_breakdown = _make_breakdown(benchmark_module)
    for phase_name in (
        "unique_build_s",
        "key_sort_s",
        "count_s",
        "buffer_setup_s",
        "scatter_s",
        "median_select_s",
    ):
        legacy_breakdown["phases"].pop(phase_name, None)

    direct_breakdown = _make_breakdown(
        benchmark_module, execution="booster->rust.groupby_median_f64_sorted"
    )
    direct_breakdown["phases"]["buffer_setup_s"] = benchmark_module.compute_stats([0.1])
    cases = [
        {
            "preset": "1key",
            "workload": "standard",
            "agg": "std",
            "sort": True,
            "execution": {
                "pandas": "pandas.groupby.std",
                "booster": "booster->rust.groupby_std_f64_sorted",
            },
            "result": _make_result(benchmark_module, preset="1key", agg="std", sort=True),
            "breakdown": legacy_breakdown,
        },
        {
            "preset": "1key",
            "workload": "standard",
            "agg": "median",
            "sort": True,
            "execution": {
                "pandas": "pandas.groupby.median",
                "booster": "booster->rust.groupby_median_f64_sorted",
            },
            "result": _make_result(benchmark_module, preset="1key", agg="median", sort=True),
            "breakdown": direct_breakdown,
        },
    ]

    payload = benchmark_module.build_profile_json_payload(
        cases,
        cardinality="standard",
        sort_mode="sorted",
        n_samples=1,
        selected_aggs=["std", "median"],
    )

    summary_phases = payload["single_key_sorted_standard"]["phases"]
    assert list(summary_phases) == [
        "prepare_inputs_s",
        "unique_build_s",
        "key_sort_s",
        "count_s",
        "buffer_setup_s",
        "scatter_s",
        "median_select_s",
        "local_build_s",
        "merge_s",
        "reorder_s",
        "materialize_s",
        "python_normalize_s",
        "python_series_build_s",
        "rust_total_s",
        "python_total_s",
        "total_pipeline_s",
    ]
    assert summary_phases["unique_build_s"] == pytest.approx(0.05)
    assert summary_phases["buffer_setup_s"] == pytest.approx(0.05)
    assert "median_select_s" in payload["cases"][0]["breakdown"]["phase_means"]
    assert payload["cases"][0]["breakdown"]["phase_means"]["median_select_s"] == 0.0
    assert payload["cases"][0]["breakdown"]["phase_means"]["buffer_setup_s"] == 0.0


def test_selected_median_profile_json_requires_breakdown_when_hook_expected(benchmark_module):
    case = {
        "agg": "median",
        "sort": True,
        "execution": {
            "pandas": "pandas.groupby.median",
            "booster": "booster->rust.groupby_median_f64_sorted",
        },
        "breakdown": None,
    }

    with pytest.raises(ValueError, match="selected median sorted profile breakdown"):
        benchmark_module.build_profile_json_payload(
            [case],
            cardinality="standard",
            sort_mode="sorted",
            n_samples=1,
            selected_aggs=["median"],
        )


def test_selected_median_profile_json_keeps_fallback_breakdown_null(benchmark_module):
    case = {
        "preset": "1key",
        "workload": "standard",
        "agg": "median",
        "sort": True,
        "execution": {
            "pandas": "pandas.groupby.median",
            "booster": "booster->pandas.groupby.median",
        },
        "result": _make_result(benchmark_module, preset="1key", agg="median", sort=True),
        "breakdown": None,
    }

    payload = benchmark_module.build_profile_json_payload(
        [case],
        cardinality="standard",
        sort_mode="sorted",
        n_samples=1,
        selected_aggs=["median"],
    )

    assert payload["cases"][0]["breakdown"] is None
