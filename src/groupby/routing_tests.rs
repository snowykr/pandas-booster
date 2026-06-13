use super::routing::{
    should_use_partitioned_firstseen_engine, should_use_partitioned_median_engine,
    should_use_partitioned_std_var_engine,
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
    }
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
