use super::test_support::{
    assert_float_kernel_bitwise_deterministic, make_sensitive_single_key_float_data,
};
use super::*;

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
