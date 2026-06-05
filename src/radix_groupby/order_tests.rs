use super::test_support::*;
use super::*;

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
fn test_radix_groupby_sorted_4keys() {
    let col1 = vec![1i64, 2, 1];
    let col2 = vec![10i64, 20, 10];
    let col3 = vec![100i64, 200, 100];
    let col4 = vec![1000i64, 2000, 1000];
    let values = vec![1.0, 2.0, 3.0];

    let key_slices: Vec<&[i64]> = vec![&col1, &col2, &col3, &col4];
    let result = radix_groupby_sum_f64_sorted(&key_slices, &values).unwrap();

    assert_eq!(result.n_keys, 4);
    assert_eq!(result.values.len(), 2);

    // (1,10,100,1000) -> 1.0 + 3.0 = 4.0
    assert_eq!(key_at_out(&result, 0, 0), 1);
    assert_eq!(key_at_out(&result, 0, 1), 10);
    assert_eq!(key_at_out(&result, 0, 2), 100);
    assert_eq!(key_at_out(&result, 0, 3), 1000);
    assert!((value_at_out(&result, 0) - 4.0).abs() < 1e-10);

    // (2,20,200,2000) -> 2.0
    assert_eq!(key_at_out(&result, 1, 0), 2);
    assert_eq!(key_at_out(&result, 1, 1), 20);
    assert_eq!(key_at_out(&result, 1, 2), 200);
    assert_eq!(key_at_out(&result, 1, 3), 2000);
    assert!((value_at_out(&result, 1) - 2.0).abs() < 1e-10);
}

#[test]
fn test_radix_groupby_sorted_5keys() {
    let col1 = vec![1i64, 1];
    let col2 = vec![2i64, 2];
    let col3 = vec![3i64, 3];
    let col4 = vec![4i64, 4];
    let col5 = vec![5i64, 5];
    let values = vec![1.0, 2.0];

    let key_slices: Vec<&[i64]> = vec![&col1, &col2, &col3, &col4, &col5];
    let result = radix_groupby_sum_f64_sorted(&key_slices, &values).unwrap();

    assert_eq!(result.n_keys, 5);
    assert_eq!(result.values.len(), 1);

    for (col, expected) in [1i64, 2, 3, 4, 5].into_iter().enumerate() {
        assert_eq!(key_at_out(&result, 0, col), expected);
    }
    assert!((value_at_out(&result, 0) - 3.0).abs() < 1e-10);
}

#[test]
fn test_radix_groupby_sorted_empty() {
    let col1: Vec<i64> = vec![];
    let col2: Vec<i64> = vec![];
    let values: Vec<f64> = vec![];

    let key_slices: Vec<&[i64]> = vec![&col1, &col2];
    let result = radix_groupby_sum_f64_sorted(&key_slices, &values).unwrap();

    assert_eq!(result.n_keys, 2);
    assert_eq!(result.values.len(), 0);
    assert!(result.keys_flat.is_empty());
}

#[test]
fn test_radix_groupby_sorted_i64_count() {
    let col1 = vec![1i64, 1, 2, 2, 1];
    let col2 = vec![10i64, 10, 20, 20, 10];
    let values = vec![100i64, 200, 300, 400, 500]; // Values don't matter for count

    let key_slices: Vec<&[i64]> = vec![&col1, &col2];
    let result = radix_groupby_count_i64_sorted(&key_slices, &values).unwrap();

    // Groups: (1,10) -> 3 rows, (2,20) -> 2 rows
    // Sorted: (1,10), (2,20)

    assert_eq!(result.n_keys, 2);
    assert_eq!(result.values.len(), 2);

    assert_eq!(key_at_out(&result, 0, 0), 1);
    assert_eq!(key_at_out(&result, 0, 1), 10);
    assert_eq!(value_at_out(&result, 0), 3);

    assert_eq!(key_at_out(&result, 1, 0), 2);
    assert_eq!(key_at_out(&result, 1, 1), 20);
    assert_eq!(value_at_out(&result, 1), 2);
}

#[test]
fn test_firstseen_large_output_returns_perm_without_materialized_reorder() {
    let n_groups = 120_000usize; // n_groups * n_keys (2) > SMALL_DIRECT_THRESHOLD_ELEMS
    let k1: Vec<i64> = (0..n_groups as i64).collect();
    let k2: Vec<i64> = (0..n_groups as i64)
        .map(|i| i.wrapping_mul(0x9E37_79B9_i64))
        .collect();
    let values: Vec<f64> = (0..n_groups).map(|i| i as f64).collect();

    let key_slices: Vec<&[i64]> = vec![&k1, &k2];
    let result = radix_groupby_sum_f64_firstseen_u32(&key_slices, &values).unwrap();

    assert_eq!(result.n_keys, 2);
    assert_eq!(result.values.len(), n_groups);
    assert_eq!(result.keys_flat.len(), n_groups * 2);
    assert!(result.perm.is_some());

    // Applying perm must restore exact first-seen order.
    let perm = result.perm.as_ref().unwrap();
    assert_eq!(perm.len(), n_groups);
    for (out_g, &src_g) in perm.iter().enumerate().take(n_groups) {
        assert_eq!(result.values[src_g], out_g as f64);
    }
}
