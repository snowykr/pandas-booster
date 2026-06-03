use super::test_support::*;
use super::*;
use ahash::AHashMap;

#[test]
fn test_radix_groupby_prod_f64_sorted_and_firstseen_semantics() {
    let col1 = vec![2i64, 1, 2, 1, 3, 3, 4, 4];
    let col2 = vec![20i64, 10, 20, 10, 30, 30, 40, 40];
    let values = vec![
        2.0,
        3.0,
        f64::NAN,
        4.0,
        f64::NAN,
        f64::NAN,
        f64::INFINITY,
        0.0,
    ];
    let key_slices: Vec<&[i64]> = vec![&col1, &col2];

    let sorted = radix_groupby_prod_f64_sorted(&key_slices, &values).unwrap();
    assert_eq!(sorted.n_keys, 2);
    assert_eq!(sorted.values.len(), 4);
    assert_eq!(key_at_out(&sorted, 0, 0), 1);
    assert_eq!(key_at_out(&sorted, 0, 1), 10);
    assert_eq!(value_at_out(&sorted, 0), 12.0);
    assert_eq!(key_at_out(&sorted, 2, 0), 3);
    assert_eq!(value_at_out(&sorted, 2), 1.0);
    assert!(value_at_out(&sorted, 3).is_nan());

    let firstseen = radix_groupby_prod_f64_firstseen_u64(&key_slices, &values).unwrap();
    assert_eq!(key_at_out(&firstseen, 0, 0), 2);
    assert_eq!(value_at_out(&firstseen, 0), 2.0);
    assert_eq!(key_at_out(&firstseen, 1, 0), 1);
    assert_eq!(value_at_out(&firstseen, 1), 12.0);
}

#[test]
fn test_radix_groupby_prod_i64_wraps_firstseen_u32_and_sorted() {
    let col1 = vec![9i64, 1, 9, 1, 2];
    let col2 = vec![90i64, 10, 90, 10, 20];
    let values = vec![i64::MAX, 3, 2, 4, 5];
    let key_slices: Vec<&[i64]> = vec![&col1, &col2];

    let firstseen = radix_groupby_prod_i64_firstseen_u32(&key_slices, &values).unwrap();
    assert_eq!(key_at_out(&firstseen, 0, 0), 9);
    assert_eq!(value_at_out(&firstseen, 0), i64::MAX.wrapping_mul(2));
    assert_eq!(key_at_out(&firstseen, 1, 0), 1);
    assert_eq!(value_at_out(&firstseen, 1), 12);

    let sorted = radix_groupby_prod_i64_sorted(&key_slices, &values).unwrap();
    assert_eq!(key_at_out(&sorted, 0, 0), 1);
    assert_eq!(value_at_out(&sorted, 0), 12);
    assert_eq!(key_at_out(&sorted, 2, 0), 9);
    assert_eq!(value_at_out(&sorted, 2), i64::MAX.wrapping_mul(2));
}

#[test]
fn test_radix_groupby_mean() {
    let col1 = vec![1i64, 2, 1, 2, 1];
    let col2 = vec![10i64, 20, 10, 20, 10];
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let key_slices: Vec<&[i64]> = vec![&col1, &col2];
    let result = radix_groupby_mean_f64(&key_slices, &values).unwrap();

    let mut groups: AHashMap<(i64, i64), f64> = AHashMap::new();
    for i in 0..result.values.len() {
        let k0 = result.keys_flat[i * 2];
        let k1 = result.keys_flat[i * 2 + 1];
        groups.insert((k0, k1), result.values[i]);
    }

    assert!((groups[&(1, 10)] - 3.0).abs() < 1e-10); // (1+3+5)/3
    assert!((groups[&(2, 20)] - 3.0).abs() < 1e-10); // (2+4)/2
}

#[test]
fn test_radix_groupby_var_f64_sorted() {
    let col1 = vec![2i64, 1, 2, 1, 1];
    let col2 = vec![20i64, 10, 20, 10, 10];
    let values = vec![2.0, 1.0, 4.0, 3.0, 5.0];

    let key_slices: Vec<&[i64]> = vec![&col1, &col2];
    let result = radix_groupby_var_f64_sorted(&key_slices, &values).unwrap();

    assert_eq!(result.n_keys, 2);
    assert_eq!(result.values.len(), 2);

    assert_eq!(key_at_out(&result, 0, 0), 1);
    assert_eq!(key_at_out(&result, 0, 1), 10);
    assert!((value_at_out(&result, 0) - 4.0).abs() < 1e-10);

    assert_eq!(key_at_out(&result, 1, 0), 2);
    assert_eq!(key_at_out(&result, 1, 1), 20);
    assert!((value_at_out(&result, 1) - 2.0).abs() < 1e-10);
}

#[test]
fn test_radix_groupby_std_i64_firstseen_u32_returns_f64_in_first_seen_order() {
    let col1 = vec![5i64, 1, 5, 1, 9];
    let col2 = vec![50i64, 10, 50, 10, 90];
    let values = vec![10i64, 2, 14, 4, 7];

    let key_slices: Vec<&[i64]> = vec![&col1, &col2];
    let result = radix_groupby_std_i64_firstseen_u32(&key_slices, &values).unwrap();

    assert_eq!(result.n_keys, 2);
    assert_eq!(result.values.len(), 3);

    assert_eq!(key_at_out(&result, 0, 0), 5);
    assert_eq!(key_at_out(&result, 0, 1), 50);
    assert!((value_at_out(&result, 0) - (8.0f64).sqrt()).abs() < 1e-10);

    assert_eq!(key_at_out(&result, 1, 0), 1);
    assert_eq!(key_at_out(&result, 1, 1), 10);
    assert!((value_at_out(&result, 1) - (2.0f64).sqrt()).abs() < 1e-10);

    assert_eq!(key_at_out(&result, 2, 0), 9);
    assert_eq!(key_at_out(&result, 2, 1), 90);
    assert!(value_at_out(&result, 2).is_nan());
}

#[test]
fn test_radix_groupby_i64() {
    let col1 = vec![1i64, 2, 1];
    let col2 = vec![10i64, 20, 10];
    let values: Vec<i64> = vec![100, 200, 300];

    let key_slices: Vec<&[i64]> = vec![&col1, &col2];
    let result = radix_groupby_sum_i64(&key_slices, &values).unwrap();

    let mut groups: AHashMap<(i64, i64), i64> = AHashMap::new();
    for i in 0..result.values.len() {
        let k0 = result.keys_flat[i * 2];
        let k1 = result.keys_flat[i * 2 + 1];
        groups.insert((k0, k1), result.values[i]);
    }

    assert_eq!(groups[&(1, 10)], 400);
    assert_eq!(groups[&(2, 20)], 200);
}
