from __future__ import annotations

from ._report_output_helpers import _make_breakdown, _make_result


def test_collect_stats_evidence_scopes_nan_median_diagnostics_to_median_agg(
    benchmark_module,
):
    captured_verify_by_pair: dict[tuple[str, str], bool] = {}

    def fake_benchmark_single(preset_name, *, agg, sort, n_samples, verify_correctness):
        _ = (sort, n_samples)
        captured_verify_by_pair[(agg, preset_name)] = verify_correctness
        return _make_result(benchmark_module, preset=preset_name, agg=agg, sort=True)

    def fake_generate_multi_key_dataset(**kwargs):
        _ = kwargs
        return benchmark_module.pd.DataFrame({"key": [1, 2], "value": [1.0, 2.0]})

    def fake_describe_booster_execution(*_args, **_kwargs):
        return "booster->rust.groupby_stats_f64_sorted"

    def fake_measure_booster_single_key_breakdown(*_args, **_kwargs):
        return _make_breakdown(benchmark_module, execution="stats")

    benchmark_module.collect_stats_evidence(
        n_samples=1,
        cardinality="standard",
        sort_mode="sorted",
        selected_aggs=["std", "median"],
        include_median_diagnostics=True,
        benchmark_single_func=fake_benchmark_single,
        generate_multi_key_dataset_func=fake_generate_multi_key_dataset,
        describe_booster_execution_func=fake_describe_booster_execution,
        measure_booster_single_key_breakdown_func=fake_measure_booster_single_key_breakdown,
    )

    std_presets = {preset for agg, preset in captured_verify_by_pair if agg == "std"}
    assert std_presets == {"1key"}
    assert captured_verify_by_pair[("median", "median_nan_dense_1key_5m_1k_p0")] is True
    assert captured_verify_by_pair[("median", "median_nan_dense_1key_5m_1k_p50")] is False
    assert captured_verify_by_pair[("median", "median_nan_dense_1key_5m_1k_p95")] is False
    assert captured_verify_by_pair[("median", "median_nan_dense_1key_5m_1k_p100")] is False
    assert captured_verify_by_pair[("median", "median_sparse_gap_1key_5m_1k")] is True


def test_collect_stats_evidence_skips_median_diagnostics_by_default(
    benchmark_module,
):
    captured_pairs: list[tuple[str, str]] = []

    def fake_benchmark_single(preset_name, *, agg, sort, n_samples, verify_correctness):
        _ = (sort, n_samples, verify_correctness)
        captured_pairs.append((agg, preset_name))
        return _make_result(benchmark_module, preset=preset_name, agg=agg, sort=True)

    def fake_generate_multi_key_dataset(**kwargs):
        _ = kwargs
        return benchmark_module.pd.DataFrame({"key": [1, 2], "value": [1.0, 2.0]})

    def fake_describe_booster_execution(*_args, **_kwargs):
        return "booster->rust.groupby_median_f64_sorted"

    def fake_measure_booster_single_key_breakdown(*_args, **_kwargs):
        return _make_breakdown(benchmark_module, execution="median")

    benchmark_module.collect_stats_evidence(
        n_samples=1,
        cardinality="standard",
        sort_mode="sorted",
        selected_aggs=["median"],
        benchmark_single_func=fake_benchmark_single,
        generate_multi_key_dataset_func=fake_generate_multi_key_dataset,
        describe_booster_execution_func=fake_describe_booster_execution,
        measure_booster_single_key_breakdown_func=fake_measure_booster_single_key_breakdown,
    )

    assert captured_pairs == [("median", "1key")]
