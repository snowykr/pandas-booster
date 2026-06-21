use super::routing::{
    should_use_direct_sorted_median_engine, should_use_partitioned_firstseen_engine,
    should_use_partitioned_median_engine, should_use_partitioned_prod_engine,
    should_use_partitioned_std_var_engine,
};
use super::sorted_median::{
    reset_sorted_median_direct_call_count, sorted_median_direct_call_count,
};
use super::test_support::{
    assert_float_kernel_bitwise_deterministic, make_partitioned_single_key_float_data,
    make_sensitive_single_key_float_data,
};
use super::*;

#[test]
fn firstseen_partitioned_routing_rejects_samples_at_low_sample_boundary() {
    let keys: Vec<i64> = (0..4_096).map(i64::from).collect();

    assert!(!should_use_partitioned_firstseen_engine(&keys));
}

#[test]
fn firstseen_partitioned_routing_accepts_high_uniqueness_sample() {
    let n = 20_000i64;
    let keys: Vec<i64> = (0..n).collect();

    assert!(should_use_partitioned_firstseen_engine(&keys));
}

#[test]
fn firstseen_partitioned_routing_rejects_below_min_unique_boundary() {
    let n = 20_000i64;
    let keys: Vec<i64> = (0..n).map(|i| i % 4_095).collect();

    assert!(!should_use_partitioned_firstseen_engine(&keys));
}

#[test]
fn firstseen_partitioned_routing_accepts_at_min_unique_boundary_when_sample_is_large_enough() {
    let mut keys: Vec<i64> = (0..4_096).map(i64::from).collect();
    keys.push(0);

    assert!(should_use_partitioned_firstseen_engine(&keys));
}

#[test]
fn firstseen_partitioned_routing_keeps_compatibility_wrappers_in_sync() {
    let low_sample_keys: Vec<i64> = (0..4_096).map(i64::from).collect();
    let boundary_keys: Vec<i64> = (0..4_097i64).map(|i| i % 4_096).collect();
    let high_unique_keys: Vec<i64> = (0..20_000).map(i64::from).collect();

    for keys in [&low_sample_keys, &boundary_keys, &high_unique_keys] {
        let firstseen = should_use_partitioned_firstseen_engine(keys);
        assert_eq!(should_use_partitioned_std_var_engine(keys), firstseen);
        assert_eq!(should_use_partitioned_median_engine(keys), firstseen);
        assert_eq!(should_use_partitioned_prod_engine(keys), firstseen);
    }
}

#[test]
fn prod_routing_prefers_ordered_low_for_standard_cardinality() {
    let n = 20_000usize;
    let keys: Vec<i64> = (0..n).map(|row| (row % 1_000) as i64).collect();

    assert!(!should_use_partitioned_prod_engine(&keys));
}

#[test]
fn prod_routing_keeps_partitioned_engine_for_high_uniqueness() {
    let n = 20_000usize;
    let keys: Vec<i64> = (0..n).map(|row| row as i64).collect();

    assert!(should_use_partitioned_prod_engine(&keys));
}

#[test]
fn test_std_var_routing_prefers_legacy_engine_for_low_cardinality_samples() {
    let n = 20_000usize;
    let keys: Vec<i64> = (0..n).map(|i| (i % 1_000) as i64).collect();
    let values: Vec<f64> = (0..n).map(|i| i as f64).collect();

    let profiled = profile_parallel_groupby_std_f64_firstseen_u32(&keys, &values).unwrap();

    assert!(profiled.profile.merge_s > 0.0);
    assert!(profiled.profile.partial_group_total > profiled.profile.final_group_count);
}

#[test]
fn test_std_var_routing_prefers_partitioned_engine_for_high_uniqueness_samples() {
    let n = 20_000usize;
    let keys: Vec<i64> = (0..n).map(|i| i as i64).collect();
    let values: Vec<f64> = (0..n).map(|i| i as f64).collect();

    let profiled = profile_parallel_groupby_std_f64_firstseen_u32(&keys, &values).unwrap();

    assert_eq!(profiled.profile.merge_s, 0.0);
    assert_eq!(
        profiled.profile.partial_group_total,
        profiled.profile.final_group_count
    );
}

#[test]
fn test_median_routing_prefers_legacy_engine_for_low_cardinality_samples() {
    let n = 20_000usize;
    let keys: Vec<i64> = (0..n).map(|i| (i % 1_000) as i64).collect();

    assert!(!should_use_partitioned_median_engine(&keys));
}

#[test]
fn test_median_routing_prefers_partitioned_engine_for_high_uniqueness_samples() {
    let n = 20_000usize;
    let keys: Vec<i64> = (0..n).map(|i| i as i64).collect();
    let values: Vec<f64> = (0..n).map(|i| i as f64).collect();

    assert!(should_use_partitioned_median_engine(&keys));

    let result = parallel_groupby_median_f64_firstseen_u32(&keys, &values).unwrap();
    assert_eq!(result.keys.len(), n);
    assert_eq!(result.keys[0], 0);
    assert_eq!(result.keys[n - 1], (n - 1) as i64);
    assert_eq!(result.values[0], 0.0);
    assert_eq!(result.values[n - 1], (n - 1) as f64);
}

#[test]
fn sorted_median_standard_cardinality_uses_direct_engine() {
    // Given: standard-cardinality single-key median inputs below the
    // partitioned-engine threshold.
    let keys = vec![3, 1, 2, 3, 1, 2, 4, 1, 5, 5, 6, 6, 7, 7];
    let values_f64 = vec![
        f64::NAN,
        f64::INFINITY,
        -0.0,
        f64::NAN,
        2.0,
        0.0,
        f64::NAN,
        -f64::INFINITY,
        f64::MAX,
        f64::MAX,
        -f64::MAX,
        -f64::MAX,
        8.0,
        f64::NAN,
    ];
    let values_i64 = vec![30, 10, 20, 36, 14, 24, 40, 12, 50, 52, 60, 62, 70, 72];

    assert!(!should_use_partitioned_median_engine(&keys));
    assert!(should_use_direct_sorted_median_engine(&keys));

    reset_sorted_median_direct_call_count();

    // When: both public sorted median routes are invoked.
    let result_f64 = parallel_groupby_median_f64_sorted(&keys, &values_f64).unwrap();
    let result_i64 = parallel_groupby_median_i64_sorted(&keys, &values_i64).unwrap();

    // Then: both routes use the direct sorted engine and preserve sorted keys.
    assert_eq!(sorted_median_direct_call_count(), 2);
    assert_eq!(result_f64.keys, vec![1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(result_i64.keys, vec![1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn sorted_median_large_dense_standard_cardinality_uses_direct_engine() {
    // Given: a benchmark-shaped standard-cardinality input with dense integer
    // keys where the direct path can address groups by key offset.
    let n = 300_001usize;
    let keys: Vec<i64> = (0..n).map(|row| (row % 1_000) as i64).collect();
    let values: Vec<f64> = (0..n).map(|row| row as f64).collect();

    assert!(!should_use_partitioned_median_engine(&keys));
    assert!(should_use_direct_sorted_median_engine(&keys));

    reset_sorted_median_direct_call_count();

    // When: the public sorted median route runs.
    let result = parallel_groupby_median_f64_sorted(&keys, &values).unwrap();

    // Then: it preserves sorted output through the dense direct median path.
    assert_eq!(sorted_median_direct_call_count(), 1);
    assert_eq!(result.keys.len(), 1_000);
    assert_eq!(result.keys[0], 0);
    assert_eq!(result.keys[999], 999);
}

#[test]
fn sorted_median_large_sparse_standard_cardinality_keeps_existing_route() {
    // Given: low-cardinality rows with a sparse i64 key range where dense
    // offset addressing would allocate far more groups than are present.
    let n = 300_001usize;
    let keys: Vec<i64> = (0..n)
        .map(|row| ((row % 1_000) as i64).saturating_mul(1_000_000_000))
        .collect();
    let values: Vec<f64> = (0..n).map(|row| row as f64).collect();

    assert!(!should_use_partitioned_median_engine(&keys));
    assert!(!should_use_direct_sorted_median_engine(&keys));

    reset_sorted_median_direct_call_count();

    // When: the public sorted median route runs.
    let result = parallel_groupby_median_f64_sorted(&keys, &values).unwrap();

    // Then: it avoids the dense direct path and still returns sorted keys.
    assert_eq!(sorted_median_direct_call_count(), 0);
    assert_eq!(result.keys.len(), 1_000);
    assert_eq!(result.keys[0], 0);
    assert_eq!(result.keys[999], 999_000_000_000);
}

#[test]
fn sorted_median_high_cardinality_keeps_partitioned_route() {
    // Given: high-cardinality inputs that satisfy the partitioned-engine
    // predicate.
    let n = 20_000usize;
    let keys: Vec<i64> = (0..20_000).map(i64::from).collect();
    let values_f64: Vec<f64> = (0..20_000).map(f64::from).collect();
    let values_i64: Vec<i64> = (0..20_000).map(i64::from).collect();

    assert!(should_use_partitioned_median_engine(&keys));

    reset_sorted_median_direct_call_count();

    // When: both public sorted median routes are invoked.
    let result_f64 = parallel_groupby_median_f64_sorted(&keys, &values_f64).unwrap();
    let result_i64 = parallel_groupby_median_i64_sorted(&keys, &values_i64).unwrap();

    // Then: neither route uses the direct engine, and final output is still
    // sorted by key.
    assert_eq!(sorted_median_direct_call_count(), 0);
    assert_eq!(result_f64.keys.len(), n);
    assert_eq!(result_i64.keys.len(), n);
    assert_eq!(result_f64.keys[0], 0);
    assert_eq!(result_i64.keys[0], 0);
    assert_eq!(result_f64.keys[n - 1], 19_999);
    assert_eq!(result_i64.keys[n - 1], 19_999);
}

#[test]
fn sorted_median_public_api_matches_direct_sorted_semantics() {
    // Given: unsorted rows with f64 NaNs and repeated i64 groups.
    let keys = vec![4, 2, 4, 1, 2, 1, 3, 3];
    let values_f64 = vec![f64::NAN, 8.0, 2.0, 10.0, 4.0, 14.0, 1.0, 9.0];
    let values_i64 = vec![40, 8, 20, 10, 4, 14, 1, 9];

    // When: the public sorted median APIs run.
    let result_f64 = parallel_groupby_median_f64_sorted(&keys, &values_f64).unwrap();
    let result_i64 = parallel_groupby_median_i64_sorted(&keys, &values_i64).unwrap();

    // Then: outputs are sorted by key and match direct sorted median semantics.
    assert_eq!(result_f64.keys, vec![1, 2, 3, 4]);
    assert_eq!(result_f64.values, vec![12.0, 6.0, 5.0, 2.0]);
    assert_eq!(result_i64.keys, vec![1, 2, 3, 4]);
    assert_eq!(result_i64.values, vec![12.0, 6.0, 5.0, 30.0]);
}

#[test]
fn test_sorted_std_var_routing_prefers_legacy_engine_for_low_cardinality_samples() {
    let n = 20_000usize;
    let keys: Vec<i64> = (0..n).map(|i| (i % 1_000) as i64).collect();
    let values: Vec<f64> = (0..n).map(|i| i as f64).collect();

    let profiled = profile_parallel_groupby_std_f64_sorted(&keys, &values).unwrap();

    assert!(profiled.profile.merge_s > 0.0);
    assert!(profiled.profile.partial_group_total > profiled.profile.final_group_count);
}

#[test]
fn test_sorted_std_var_routing_prefers_partitioned_engine_for_high_uniqueness_samples() {
    let n = 20_000usize;
    let keys: Vec<i64> = (0..n).map(|i| i as i64).collect();
    let values: Vec<f64> = (0..n).map(|i| i as f64).collect();

    let profiled = profile_parallel_groupby_std_f64_sorted(&keys, &values).unwrap();

    assert_eq!(profiled.profile.merge_s, 0.0);
    assert_eq!(
        profiled.profile.partial_group_total,
        profiled.profile.final_group_count
    );
}

#[test]
fn test_std_f64_partitioned_sorted_bitwise_deterministic_across_threads() {
    let (keys, values) = make_partitioned_single_key_float_data();
    assert_float_kernel_bitwise_deterministic(parallel_groupby_std_f64_sorted, &keys, &values);
}

#[test]
fn test_var_f64_partitioned_firstseen_u32_bitwise_deterministic_across_threads() {
    let (keys, values) = make_partitioned_single_key_float_data();
    assert_float_kernel_bitwise_deterministic(
        parallel_groupby_var_f64_firstseen_u32,
        &keys,
        &values,
    );
}

#[test]
fn test_var_f64_firstseen_u32_bitwise_deterministic_across_threads() {
    let (keys, values) = make_sensitive_single_key_float_data();
    assert_float_kernel_bitwise_deterministic(
        parallel_groupby_var_f64_firstseen_u32,
        &keys,
        &values,
    );
}

#[test]
fn test_std_f64_firstseen_u32_bitwise_deterministic_across_threads() {
    let (keys, values) = make_sensitive_single_key_float_data();
    assert_float_kernel_bitwise_deterministic(
        parallel_groupby_std_f64_firstseen_u32,
        &keys,
        &values,
    );
}

#[test]
fn test_var_f64_firstseen_u64_bitwise_deterministic_across_threads() {
    let (keys, values) = make_sensitive_single_key_float_data();
    assert_float_kernel_bitwise_deterministic(
        parallel_groupby_var_f64_firstseen_u64,
        &keys,
        &values,
    );
}

#[test]
fn test_std_f64_firstseen_u64_bitwise_deterministic_across_threads() {
    let (keys, values) = make_sensitive_single_key_float_data();
    assert_float_kernel_bitwise_deterministic(
        parallel_groupby_std_f64_firstseen_u64,
        &keys,
        &values,
    );
}
