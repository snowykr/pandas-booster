use super::api_f64::parallel_groupby_median_f64_sorted;
use super::api_f64_stats::profile_parallel_groupby_median_f64_sorted;
use super::api_i64::parallel_groupby_median_i64_sorted;
use super::routing::{
    should_use_partitioned_median_engine, sorted_median_route_decision, SortedMedianRouteKind,
    SortedMedianRouteReason, SortedMedianValueKind,
};

#[test]
fn full_i64_span_preserves_partitioned_sample_boundaries() {
    for (rows, groups, partitioned) in [
        (4_096, 4_096, false),
        (4_097, 4_095, false),
        (4_097, 4_096, true),
        (20_000, 2, false),
        (1_048_576, 5_003, true),
    ] {
        let keys: Vec<i64> = (0..rows)
            .map(|row| match row % groups {
                0 => i64::MIN,
                1 => i64::MAX,
                group => group as i64,
            })
            .collect();
        assert_eq!(should_use_partitioned_median_engine(&keys), partitioned);
        for value_kind in [SortedMedianValueKind::F64, SortedMedianValueKind::I64] {
            let decision = sorted_median_route_decision(&keys, value_kind);
            let (kind, reason) = if partitioned {
                (
                    SortedMedianRouteKind::PartitionedFallback,
                    SortedMedianRouteReason::HighCardinalityPartitionedSample,
                )
            } else {
                (
                    SortedMedianRouteKind::ExistingSortedFallback,
                    SortedMedianRouteReason::DenseSpanOverflow,
                )
            };
            assert_eq!(decision.kind, kind, "rows={rows}, groups={groups}");
            assert_eq!(decision.reason, reason);
            assert_eq!(decision.dense_span, None);
            assert_eq!(decision.dense_min_key, None);
        }
    }
}

#[test]
fn profile_groupby_median_f64_sorted_rejects_mismatched_lengths() {
    pyo3::Python::initialize();

    let keys = [1, 2, 1];
    let values = [2.0, 3.0];

    let error = profile_parallel_groupby_median_f64_sorted(&keys, &values).unwrap_err();

    assert!(
        error.to_string().contains("keys and values"),
        "unexpected error: {error}"
    );
}

#[test]
fn profile_groupby_median_f64_sorted_returns_sorted_profiled_result() {
    let keys = [2, 1, 2, 1, 3];
    let values = [6.0, 10.0, 2.0, 20.0, f64::NAN];

    let profiled =
        profile_parallel_groupby_median_f64_sorted(&keys, &values).expect("profile succeeds");

    assert_eq!(profiled.result.keys, vec![1, 2, 3]);
    assert_eq!(profiled.result.values[0].to_bits(), 15.0f64.to_bits());
    assert_eq!(profiled.result.values[1].to_bits(), 4.0f64.to_bits());
    assert!(profiled.result.values[2].is_nan());
    assert_eq!(profiled.profile.final_group_count, 3);
}

#[test]
fn profile_groupby_median_f64_sorted_reports_direct_dense_and_sparse_shapes() {
    for (key_scale, route_kind) in [(1, "direct_dense"), (1_000_003, "direct_sparse")] {
        let mut keys = Vec::with_capacity(6_000);
        let mut values = Vec::with_capacity(6_000);
        for row in 0..6_000_i64 {
            let key = ((row * 37).rem_euclid(251) - 125) * key_scale;
            keys.push(key);
            values.push(((row * 17).rem_euclid(1_009) - 500) as f64);
        }

        let public_result =
            parallel_groupby_median_f64_sorted(&keys, &values).expect("public route succeeds");
        let profiled =
            profile_parallel_groupby_median_f64_sorted(&keys, &values).expect("profile succeeds");

        assert_eq!(profiled.result.keys, public_result.keys);
        assert_eq!(profiled.result.values, public_result.values);
        assert_eq!(profiled.result.keys.len(), 251);
        assert_eq!(profiled.profile.route_kind, Some(route_kind));
        assert_eq!(profiled.profile.merge_s, 0.0);
        assert_eq!(profiled.profile.reorder_s, 0.0);
        // Only route planning contributes here; no performance threshold is imposed.
        assert!(profiled.profile.local_build_s > 0.0);
        assert!(profiled.profile.unique_build_s >= 0.0);
        assert!(profiled.profile.key_sort_s >= 0.0);
        assert_eq!(profiled.profile.count_s, 0.0);
        assert!(profiled.profile.buffer_setup_s >= 0.0);
        assert!(profiled.profile.scatter_s >= 0.0);
        assert!(profiled.profile.median_select_s >= 0.0);
        assert_eq!(
            profiled.profile.partial_group_total,
            profiled.profile.final_group_count
        );
    }
}

#[test]
fn profile_groupby_median_f64_sorted_high_cardinality_uses_partitioned_shape() {
    let n = 20_000usize;
    let keys: Vec<i64> = (0..n).map(|row| row as i64).collect();
    let values: Vec<f64> = (0..n).map(|row| row as f64).collect();

    assert!(should_use_partitioned_median_engine(&keys));

    let public_result =
        parallel_groupby_median_f64_sorted(&keys, &values).expect("public route succeeds");
    let profiled =
        profile_parallel_groupby_median_f64_sorted(&keys, &values).expect("profile succeeds");

    assert_eq!(profiled.result.keys, public_result.keys);
    assert_eq!(profiled.result.values, public_result.values);
    assert_eq!(profiled.profile.route_kind, Some("partitioned_fallback"));
    assert!(profiled.profile.local_build_s > 0.0);
    assert_eq!(profiled.profile.merge_s, 0.0);
    assert_eq!(profiled.profile.unique_build_s, 0.0);
    assert_eq!(profiled.profile.key_sort_s, 0.0);
    assert_eq!(profiled.profile.count_s, 0.0);
    assert_eq!(profiled.profile.buffer_setup_s, 0.0);
    assert_eq!(profiled.profile.scatter_s, 0.0);
    assert_eq!(profiled.profile.median_select_s, 0.0);
    assert_eq!(
        profiled.profile.partial_group_total,
        profiled.profile.final_group_count
    );
}

#[test]
fn full_i64_span_high_cardinality_median_uses_partitioned_aggregation() {
    let groups = 5_003;
    let keys: Vec<i64> = (0..1_048_576)
        .map(|row| match row % groups {
            0 => i64::MIN,
            1 => i64::MAX,
            group => i64::from(group),
        })
        .collect();
    let values_i64: Vec<i64> = (0..keys.len()).map(|row| (row % 5_003) as i64).collect();
    let values_f64: Vec<f64> = values_i64.iter().map(|value| *value as f64).collect();
    let mut expected: Vec<(i64, f64)> = (0..groups)
        .map(|group| {
            let key = match group {
                0 => i64::MIN,
                1 => i64::MAX,
                group => i64::from(group),
            };
            (key, f64::from(group))
        })
        .collect();
    expected.sort_unstable_by_key(|(key, _)| *key);
    let (expected_keys, expected_values): (Vec<_>, Vec<_>) = expected.into_iter().unzip();

    let float_result = parallel_groupby_median_f64_sorted(&keys, &values_f64).unwrap();
    let int_result = parallel_groupby_median_i64_sorted(&keys, &values_i64).unwrap();
    let profiled = profile_parallel_groupby_median_f64_sorted(&keys, &values_f64).unwrap();

    for result in [float_result, int_result, profiled.result] {
        assert_eq!(result.keys, expected_keys);
        assert_eq!(result.values, expected_values);
    }
    assert_eq!(profiled.profile.route_kind, Some("partitioned_fallback"));
    assert_eq!(
        profiled.profile.route_reason,
        Some("high_cardinality_partitioned_sample")
    );
    assert_eq!(profiled.profile.merge_s, 0.0);
    assert_eq!(profiled.profile.partial_group_total, groups as usize);
    assert_eq!(profiled.profile.final_group_count, groups as usize);
}

#[test]
fn profile_groupby_median_f64_sorted_reports_existing_fallback_shape() {
    let keys = [i64::MAX, i64::MIN, i64::MAX, i64::MIN];
    let values = [6.0, 10.0, 2.0, 20.0];

    let profiled =
        profile_parallel_groupby_median_f64_sorted(&keys, &values).expect("profile succeeds");

    assert_eq!(profiled.result.keys, vec![i64::MIN, i64::MAX]);
    assert_eq!(profiled.result.values, vec![15.0, 4.0]);
    assert_eq!(
        profiled.profile.route_kind,
        Some("existing_sorted_fallback")
    );
    assert_eq!(profiled.profile.route_reason, Some("dense_span_overflow"));
    assert!(profiled.profile.local_build_s > 0.0);
    assert_eq!(profiled.profile.unique_build_s, 0.0);
    assert_eq!(profiled.profile.key_sort_s, 0.0);
    assert_eq!(profiled.profile.count_s, 0.0);
    assert_eq!(profiled.profile.buffer_setup_s, 0.0);
    assert_eq!(profiled.profile.scatter_s, 0.0);
    assert_eq!(profiled.profile.median_select_s, 0.0);
    assert_eq!(profiled.profile.final_group_count, 2);
}
