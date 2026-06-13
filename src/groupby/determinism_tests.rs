use super::test_support::{
    assert_float_kernel_bitwise_deterministic, make_sensitive_single_key_float_data,
};
use super::*;

fn assert_f64_values_eq(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual_value, &expected_value) in actual.iter().zip(expected.iter()) {
        if expected_value.is_nan() {
            assert!(actual_value.is_nan());
        } else {
            assert_eq!(actual_value.to_bits(), expected_value.to_bits());
        }
    }
}

#[test]
fn test_sum_f64_sorted_bitwise_deterministic_across_threads() {
    let (keys, values) = make_sensitive_single_key_float_data();
    assert_float_kernel_bitwise_deterministic(parallel_groupby_sum_f64_sorted, &keys, &values);
}

#[test]
fn test_mean_f64_sorted_bitwise_deterministic_across_threads() {
    let (keys, values) = make_sensitive_single_key_float_data();
    assert_float_kernel_bitwise_deterministic(parallel_groupby_mean_f64_sorted, &keys, &values);
}

#[test]
fn test_sum_f64_firstseen_u32_bitwise_deterministic_across_threads() {
    let (keys, values) = make_sensitive_single_key_float_data();
    assert_float_kernel_bitwise_deterministic(
        parallel_groupby_sum_f64_firstseen_u32,
        &keys,
        &values,
    );
}

#[test]
fn test_mean_f64_firstseen_u32_bitwise_deterministic_across_threads() {
    let (keys, values) = make_sensitive_single_key_float_data();
    assert_float_kernel_bitwise_deterministic(
        parallel_groupby_mean_f64_firstseen_u32,
        &keys,
        &values,
    );
}

#[test]
fn test_sum_f64_firstseen_u64_bitwise_deterministic_across_threads() {
    let (keys, values) = make_sensitive_single_key_float_data();
    assert_float_kernel_bitwise_deterministic(
        parallel_groupby_sum_f64_firstseen_u64,
        &keys,
        &values,
    );
}

#[test]
fn test_mean_f64_firstseen_u64_bitwise_deterministic_across_threads() {
    let (keys, values) = make_sensitive_single_key_float_data();
    assert_float_kernel_bitwise_deterministic(
        parallel_groupby_mean_f64_firstseen_u64,
        &keys,
        &values,
    );
}

#[test]
fn test_scalar_firstseen_partitioned_routes_preserve_first_seen_order() {
    let n = 5_000usize;
    let keys: Vec<i64> = (0..n).map(|i| ((i + n / 2) % n) as i64).collect();
    let values_f64: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let values_i64: Vec<i64> = (0..n).map(|i| i as i64).collect();

    let count_f64 = parallel_groupby_count_f64_firstseen_u32(&keys, &values_f64).unwrap();
    assert_eq!(count_f64.keys, keys);
    assert_eq!(count_f64.values, vec![1_i64; n]);

    let max_i64 = parallel_groupby_max_i64_firstseen_u64(&keys, &values_i64).unwrap();
    assert_eq!(max_i64.keys, keys);
    assert_eq!(max_i64.values, values_i64);
}

#[test]
fn test_f64_firstseen_target_aggs_preserve_special_value_semantics_u32_u64() {
    let keys = vec![8, 1, 8, 2, 1, 3, 3, 4, 4];
    let values = vec![
        1e16,
        f64::NAN,
        -1e16,
        f64::INFINITY,
        5.0,
        f64::NAN,
        f64::NAN,
        f64::NEG_INFINITY,
        7.0,
    ];
    let expected_keys = vec![8, 1, 2, 3, 4];

    let sum_u32 = parallel_groupby_sum_f64_firstseen_u32(&keys, &values).unwrap();
    let sum_u64 = parallel_groupby_sum_f64_firstseen_u64(&keys, &values).unwrap();
    assert_eq!(sum_u32.keys, expected_keys);
    assert_eq!(sum_u64.keys, expected_keys);
    assert_f64_values_eq(
        &sum_u32.values,
        &[0.0, 5.0, f64::INFINITY, 0.0, f64::NEG_INFINITY],
    );
    assert_f64_values_eq(&sum_u64.values, &sum_u32.values);

    let mean_u32 = parallel_groupby_mean_f64_firstseen_u32(&keys, &values).unwrap();
    let mean_u64 = parallel_groupby_mean_f64_firstseen_u64(&keys, &values).unwrap();
    assert_eq!(mean_u32.keys, expected_keys);
    assert_eq!(mean_u64.keys, expected_keys);
    assert_f64_values_eq(
        &mean_u32.values,
        &[0.0, 5.0, f64::INFINITY, f64::NAN, f64::NEG_INFINITY],
    );
    assert_f64_values_eq(&mean_u64.values, &mean_u32.values);

    let min_u32 = parallel_groupby_min_f64_firstseen_u32(&keys, &values).unwrap();
    let min_u64 = parallel_groupby_min_f64_firstseen_u64(&keys, &values).unwrap();
    assert_eq!(min_u32.keys, expected_keys);
    assert_eq!(min_u64.keys, expected_keys);
    assert_f64_values_eq(
        &min_u32.values,
        &[-1e16, 5.0, f64::INFINITY, f64::NAN, f64::NEG_INFINITY],
    );
    assert_f64_values_eq(&min_u64.values, &min_u32.values);

    let max_u32 = parallel_groupby_max_f64_firstseen_u32(&keys, &values).unwrap();
    let max_u64 = parallel_groupby_max_f64_firstseen_u64(&keys, &values).unwrap();
    assert_eq!(max_u32.keys, expected_keys);
    assert_eq!(max_u64.keys, expected_keys);
    assert_f64_values_eq(&max_u32.values, &[1e16, 5.0, f64::INFINITY, f64::NAN, 7.0]);
    assert_f64_values_eq(&max_u64.values, &max_u32.values);

    let count_u32 = parallel_groupby_count_f64_firstseen_u32(&keys, &values).unwrap();
    let count_u64 = parallel_groupby_count_f64_firstseen_u64(&keys, &values).unwrap();
    assert_eq!(count_u32.keys, expected_keys);
    assert_eq!(count_u64.keys, expected_keys);
    assert_eq!(count_u32.values, vec![2, 1, 1, 0, 2]);
    assert_eq!(count_u64.values, count_u32.values);
}

#[test]
fn test_var_f64_firstseen_u32_preserves_first_seen_order_and_nan_semantics() {
    let keys = vec![5, 1, 5, 9, 1, 7, 7];
    let values = vec![10.0, 2.0, 14.0, f64::NAN, 4.0, 3.0, 3.0];

    let result = parallel_groupby_var_f64_firstseen_u32(&keys, &values).unwrap();

    assert_eq!(result.keys, vec![5, 1, 9, 7]);
    assert!((result.values[0] - 8.0).abs() < 1e-10);
    assert!((result.values[1] - 2.0).abs() < 1e-10);
    assert!(result.values[2].is_nan());
    assert_eq!(result.values[3].to_bits(), 0.0f64.to_bits());
}

#[test]
fn test_std_i64_firstseen_u64_preserves_first_seen_order_and_singleton_nan() {
    let keys = vec![5, 1, 5, 9, 1];
    let values = vec![10_i64, 2, 14, 7, 4];

    let result = parallel_groupby_std_i64_firstseen_u64(&keys, &values).unwrap();

    assert_eq!(result.keys, vec![5, 1, 9]);
    assert!((result.values[0] - (8.0f64).sqrt()).abs() < 1e-10);
    assert!((result.values[1] - (2.0f64).sqrt()).abs() < 1e-10);
    assert!(result.values[2].is_nan());
}
