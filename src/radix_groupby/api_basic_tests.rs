use super::dispatch::{radix_groupby, radix_groupby_firstseen_u32, radix_groupby_firstseen_u64};
use super::test_support::*;
use super::*;
use ahash::AHashMap;

#[test]
fn test_radix_groupby_sum_basic() {
    let col1 = vec![1i64, 2, 1, 2, 1];
    let col2 = vec![10i64, 20, 10, 20, 10];
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let key_slices: Vec<&[i64]> = vec![&col1, &col2];
    let result = radix_groupby_sum_f64(&key_slices, &values).unwrap();

    assert_eq!(result.n_keys, 2);
    assert_eq!(result.values.len(), 2);

    let mut groups: AHashMap<(i64, i64), f64> = AHashMap::new();
    for i in 0..result.values.len() {
        let k0 = result.keys_flat[i * 2];
        let k1 = result.keys_flat[i * 2 + 1];
        groups.insert((k0, k1), result.values[i]);
    }

    assert!((groups[&(1, 10)] - 9.0).abs() < 1e-10);
    assert!((groups[&(2, 20)] - 6.0).abs() < 1e-10);
}

#[test]
fn test_radix_groupby_materialization_uses_owned_finalize() {
    let col1 = vec![1i64, 2, 1, 2, 1];
    let col2 = vec![10i64, 20, 10, 20, 10];
    let values = vec![100_i64, 200, 300, 400, 500];
    let key_slices: Vec<&[i64]> = vec![&col1, &col2];

    let result = radix_groupby::<i64, OwnedOnlyAgg, usize>(&key_slices, &values).unwrap();

    let mut groups: AHashMap<(i64, i64), usize> = AHashMap::new();
    for i in 0..result.values.len() {
        let k0 = result.keys_flat[i * 2];
        let k1 = result.keys_flat[i * 2 + 1];
        groups.insert((k0, k1), result.values[i]);
    }

    assert_eq!(groups[&(1, 10)], 3);
    assert_eq!(groups[&(2, 20)], 2);
}

#[test]
fn test_radix_firstseen_materialization_uses_owned_finalize() {
    let col1 = vec![2i64, 1, 2, 1, 3];
    let col2 = vec![20i64, 10, 20, 10, 30];
    let values = vec![100_i64, 200, 300, 400, 500];
    let key_slices: Vec<&[i64]> = vec![&col1, &col2];

    let result =
        radix_groupby_firstseen_u32::<i64, OwnedOnlyAgg, usize>(&key_slices, &values).unwrap();
    assert_eq!(key_at_out(&result, 0, 0), 2);
    assert_eq!(value_at_out(&result, 0), 2);
    assert_eq!(key_at_out(&result, 1, 0), 1);
    assert_eq!(value_at_out(&result, 1), 2);
    assert_eq!(key_at_out(&result, 2, 0), 3);
    assert_eq!(value_at_out(&result, 2), 1);

    let result =
        radix_groupby_firstseen_u64::<i64, OwnedOnlyAgg, usize>(&key_slices, &values).unwrap();
    assert_eq!(key_at_out(&result, 0, 0), 2);
    assert_eq!(value_at_out(&result, 0), 2);
    assert_eq!(key_at_out(&result, 1, 0), 1);
    assert_eq!(value_at_out(&result, 1), 2);
    assert_eq!(key_at_out(&result, 2, 0), 3);
    assert_eq!(value_at_out(&result, 2), 1);
}

#[test]
fn test_radix_groupby_sorted_3keys() {
    let col1 = vec![1i64, 2, 1];
    let col2 = vec![10i64, 20, 10];
    let col3 = vec![100i64, 200, 100];
    let values = vec![1.0, 2.0, 3.0];

    let key_slices: Vec<&[i64]> = vec![&col1, &col2, &col3];
    let result = radix_groupby_sum_f64_sorted(&key_slices, &values).unwrap();

    assert_eq!(result.n_keys, 3);
    assert_eq!(result.values.len(), 2);

    // Verify sorted order: (1,10,100), (2,20,200)
    assert_eq!(key_at_out(&result, 0, 0), 1);
    assert_eq!(key_at_out(&result, 0, 1), 10);
    assert_eq!(key_at_out(&result, 0, 2), 100);
    assert!((value_at_out(&result, 0) - 4.0).abs() < 1e-10);

    assert_eq!(key_at_out(&result, 1, 0), 2);
    assert_eq!(key_at_out(&result, 1, 1), 20);
    assert_eq!(key_at_out(&result, 1, 2), 200);
    assert!((value_at_out(&result, 1) - 2.0).abs() < 1e-10);
}

#[test]
fn test_radix_groupby_sorted() {
    let col1 = vec![2i64, 1, 2, 1, 3];
    let col2 = vec![20i64, 10, 20, 10, 30];
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let key_slices: Vec<&[i64]> = vec![&col1, &col2];
    let result = radix_groupby_sum_f64_sorted(&key_slices, &values).unwrap();

    assert_eq!(result.n_keys, 2);
    assert_eq!(result.values.len(), 3);

    // Verify sorted order: (1,10), (2,20), (3,30)
    assert_eq!(key_at_out(&result, 0, 0), 1);
    assert_eq!(key_at_out(&result, 0, 1), 10);
    assert_eq!(key_at_out(&result, 1, 0), 2);
    assert_eq!(key_at_out(&result, 1, 1), 20);
    assert_eq!(key_at_out(&result, 2, 0), 3);
    assert_eq!(key_at_out(&result, 2, 1), 30);

    assert!((value_at_out(&result, 0) - 6.0).abs() < 1e-10); // (1,10): 2+4
    assert!((value_at_out(&result, 1) - 4.0).abs() < 1e-10); // (2,20): 1+3
    assert!((value_at_out(&result, 2) - 5.0).abs() < 1e-10); // (3,30): 5
}

#[test]
fn test_radix_groupby_sorted_negative_keys() {
    let col1 = vec![0i64, -1, 2, -1, -3];
    let col2 = vec![5i64, 4, 6, 4, 3];
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let key_slices: Vec<&[i64]> = vec![&col1, &col2];
    let result = radix_groupby_sum_f64_sorted(&key_slices, &values).unwrap();

    assert_eq!(result.n_keys, 2);
    assert_eq!(result.values.len(), 4);

    // Verify sorted order: (-3,3), (-1,4), (0,5), (2,6)
    assert_eq!(key_at_out(&result, 0, 0), -3);
    assert_eq!(key_at_out(&result, 0, 1), 3);
    assert_eq!(key_at_out(&result, 1, 0), -1);
    assert_eq!(key_at_out(&result, 1, 1), 4);
    assert_eq!(key_at_out(&result, 2, 0), 0);
    assert_eq!(key_at_out(&result, 2, 1), 5);
    assert_eq!(key_at_out(&result, 3, 0), 2);
    assert_eq!(key_at_out(&result, 3, 1), 6);

    assert!((value_at_out(&result, 0) - 5.0).abs() < 1e-10);
    assert!((value_at_out(&result, 1) - 6.0).abs() < 1e-10);
    assert!((value_at_out(&result, 2) - 1.0).abs() < 1e-10);
    assert!((value_at_out(&result, 3) - 3.0).abs() < 1e-10);
}

#[test]
fn test_radix_groupby_empty() {
    let col1: Vec<i64> = vec![];
    let col2: Vec<i64> = vec![];
    let values: Vec<f64> = vec![];

    let key_slices: Vec<&[i64]> = vec![&col1, &col2];
    let result = radix_groupby_sum_f64(&key_slices, &values).unwrap();

    assert_eq!(result.n_keys, 2);
    assert_eq!(result.values.len(), 0);
}

#[test]
fn test_radix_groupby_with_nan() {
    let col1 = vec![1i64, 1, 1];
    let col2 = vec![10i64, 10, 10];
    let values = vec![1.0, f64::NAN, 2.0];

    let key_slices: Vec<&[i64]> = vec![&col1, &col2];
    let result = radix_groupby_sum_f64(&key_slices, &values).unwrap();

    assert_eq!(result.values.len(), 1);
    assert!((result.values[0] - 3.0).abs() < 1e-10);
}

#[test]
fn test_radix_groupby_three_keys() {
    let col1 = vec![1i64, 1, 2];
    let col2 = vec![10i64, 10, 20];
    let col3 = vec![100i64, 100, 200];
    let values = vec![1.0, 2.0, 3.0];

    let key_slices: Vec<&[i64]> = vec![&col1, &col2, &col3];
    let result = radix_groupby_sum_f64(&key_slices, &values).unwrap();

    assert_eq!(result.n_keys, 3);
    assert_eq!(result.values.len(), 2);
}
