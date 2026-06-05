use super::routing::should_use_partitioned_median_engine;
use super::test_support::{
    assert_float_kernel_bitwise_deterministic, make_partitioned_single_key_float_data,
    make_sensitive_single_key_float_data,
};
use super::*;

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
