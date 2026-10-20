from __future__ import annotations

import pytest

from ._profile_json_helpers import REQUIRED_MULTI_KEY_SORTED_PHASES


@pytest.mark.parametrize("integer_values", [False, True])
def test_multi_key_profile_timing_accounts_for_each_stage_once(
    benchmark_module, monkeypatch, integer_values
):
    import pandas_booster._rust as rust
    import profile_json_multi_key as collector

    df = benchmark_module.pd.DataFrame(
        {"k1": [0, 1], "k2": [1, 0], "value": [1, 2] if integer_values else [1.0, 2.0]}
    )
    profile_calls = []

    def groupby_multi_max_sorted(_keys, _values):
        raise AssertionError("the unprofiled function must not be called")

    def profile_groupby_multi_max_sorted(keys, values):
        profile_calls.append((keys, values))
        profile = dict.fromkeys(REQUIRED_MULTI_KEY_SORTED_PHASES, 0.0)
        profile.update(
            hash_build_s=3.0,
            conversion_s=2.0,
            rust_total_s=3.0,
            partial_group_total=2,
            final_group_count=2,
            partial_to_final_ratio=1.0,
            route="hash_first",
            sort_first_segment_scan_count=0,
            selected_sort_strategy="packed_u64",
            sort_key_bit_widths=[1, 1],
        )
        return keys, values, profile

    monkeypatch.setattr(
        collector,
        "resolve_booster_benchmark_dispatch",
        lambda *_args: {
            "rust_func": groupby_multi_max_sorted,
            "needs_python_sort": False,
        },
    )
    monkeypatch.setattr(
        rust,
        "profile_groupby_multi_max_sorted",
        profile_groupby_multi_max_sorted,
        raising=False,
    )
    # Per sample: preparation 2s, Rust engine 3s + conversion 2s,
    # Python normalization 3s, and Series/index construction 4s.
    ticks = iter([0, 0, 2, 7, 10, 10, 14, 14, 20, 20, 22, 27, 30, 30, 34, 34])
    monkeypatch.setattr(collector.time, "perf_counter", lambda: next(ticks))

    breakdown = collector.measure_booster_multi_key_sorted_breakdown(
        df, ["k1", "k2"], "max", True, 2
    )

    assert breakdown is not None
    assert len(profile_calls) == 2
    assert set(breakdown["phases"]) == set(REQUIRED_MULTI_KEY_SORTED_PHASES)
    expected = dict.fromkeys(REQUIRED_MULTI_KEY_SORTED_PHASES, 0.0)
    expected.update(
        prepare_inputs_s=2.0,
        hash_build_s=3.0,
        conversion_s=2.0,
        python_normalize_s=3.0,
        pandas_index_construction_s=4.0,
    )
    for name, duration in expected.items():
        assert breakdown["phases"][name].samples == [duration, duration]
    assert breakdown["rust_total_s"] == 5.0
    assert breakdown["python_total_s"] == 9.0
    assert breakdown["total_pipeline_s"] == 14.0
    assert sum(stats.mean for stats in breakdown["phases"].values()) == 14.0
    assert breakdown["rust_total_s"] + breakdown["python_total_s"] == 14.0
    assert next(ticks, None) is None


@pytest.mark.parametrize(
    "agg,sort,force_python_sort,threshold",
    [
        ("sum", True, False, 0),
        ("max", False, False, 0),
        ("max", True, True, 0),
        ("max", True, False, 3),
    ],
)
def test_multi_key_profile_rejects_unsupported_dispatch(
    benchmark_module, monkeypatch, agg, sort, force_python_sort, threshold
):
    import pandas_booster._rust as rust
    import profile_json_multi_key as collector

    df = benchmark_module.pd.DataFrame({"k1": [0, 1], "k2": [1, 0], "value": [1.0, 2.0]})
    monkeypatch.setattr(rust, "get_fallback_threshold", lambda: threshold)
    monkeypatch.setenv("PANDAS_BOOSTER_FORCE_PANDAS_SORT", str(int(force_python_sort)))
    monkeypatch.delenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", raising=False)

    dispatch = collector.resolve_booster_benchmark_dispatch(df, ["k1", "k2"], "value", agg, sort)
    if force_python_sort:
        assert dispatch["rust_func"] is not None
        assert dispatch["needs_python_sort"] is True
    elif threshold:
        assert dispatch["rust_func"] is None

    def unexpected_profile(*_args):
        raise AssertionError("unsupported dispatch must not invoke the profile kernel")

    monkeypatch.setattr(rust, "profile_groupby_multi_max_f64_sorted", unexpected_profile)
    assert (
        collector.measure_booster_multi_key_sorted_breakdown(df, ["k1", "k2"], agg, sort, 1) is None
    )
