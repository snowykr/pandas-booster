"""Benchmark profile JSON tests."""

from __future__ import annotations

from ._report_output_helpers import _make_breakdown, _make_result


def test_false_low_sample_preset_hides_sparse_cardinality_from_router(benchmark_module):
    from pandas_booster import _rust

    config = benchmark_module.PRESETS["median_false_low_sample_tail_unique"]
    df = benchmark_module.generate_multi_key_dataset(**config)
    keys = df["key"].to_numpy()
    stride = (len(keys) + 16_383) // 16_384
    assert len(benchmark_module.np.unique(keys)) == 500_000
    assert len(benchmark_module.np.unique(keys[::stride])) == 1
    assert int(keys.max()) - int(keys.min()) > 1_000_000_000
    _, _, profile = _rust.profile_groupby_median_f64_sorted(keys, df["value"].to_numpy())
    assert profile["route_kind"] == "existing_sorted_fallback"
    assert profile["route_reason"] == "sparse_exact_groups_exceed_cap"


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
