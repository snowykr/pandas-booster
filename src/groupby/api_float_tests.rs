use super::test_support::row_order_prod_for_key;
use super::*;
use ahash::AHashMap;

#[test]
fn test_groupby_sum_f64() {
    let keys = vec![1, 2, 1, 2, 1];
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = parallel_groupby_sum_f64(&keys, &values).unwrap();

    let mut map: AHashMap<i64, f64> = AHashMap::new();
    for (k, v) in result.keys.iter().zip(result.values.iter()) {
        map.insert(*k, *v);
    }

    assert!((map[&1] - 9.0).abs() < 1e-10);
    assert!((map[&2] - 6.0).abs() < 1e-10);
}

#[test]
fn test_groupby_prod_f64_nan_and_arithmetic_nan_semantics() {
    let keys = vec![1, 1, 1, 2, 2, 3, 3];
    let values = vec![2.0, f64::NAN, 3.0, f64::NAN, f64::NAN, f64::INFINITY, 0.0];
    let result = parallel_groupby_prod_f64(&keys, &values).unwrap();

    let mut map: AHashMap<i64, f64> = AHashMap::new();
    for (k, v) in result.keys.iter().zip(result.values.iter()) {
        map.insert(*k, *v);
    }

    assert_eq!(map[&1], 6.0);
    assert_eq!(map[&2], 1.0);
    assert!(map[&3].is_nan());
}

#[test]
fn test_groupby_prod_f64_sorted_and_firstseen() {
    let keys = vec![3, 1, 2, 1, 3];
    let values = vec![2.0, 10.0, 100.0, 0.5, 4.0];

    let sorted = parallel_groupby_prod_f64_sorted(&keys, &values).unwrap();
    assert_eq!(sorted.keys, vec![1, 2, 3]);
    assert_eq!(sorted.values, vec![5.0, 100.0, 8.0]);

    let firstseen = parallel_groupby_prod_f64_firstseen_u32(&keys, &values).unwrap();
    assert_eq!(firstseen.keys, vec![3, 1, 2]);
    assert_eq!(firstseen.values, vec![8.0, 5.0, 100.0]);
}

#[test]
fn test_groupby_prod_f64_preserves_row_order_ieee_semantics() {
    let keys = vec![7, 7, 7, 7, 9, 9, 9, 9, 11, 11, 11, 11, 13, 13, 13, 13];
    let values = vec![
        1e308,
        1e308,
        1e-308,
        1e-308,
        0.0,
        f64::INFINITY,
        2.0,
        f64::NAN,
        -0.0,
        2.0,
        3.0,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
    ];

    let sorted = parallel_groupby_prod_f64_sorted(&keys, &values).unwrap();
    assert_eq!(sorted.keys, vec![7, 9, 11, 13]);
    for (idx, &key) in sorted.keys.iter().enumerate() {
        let expected = row_order_prod_for_key(&keys, &values, key);
        assert_eq!(sorted.values[idx].to_bits(), expected.to_bits());
    }
    assert!(sorted.values[0].is_infinite());
    assert!(sorted.values[1].is_nan());
    assert_eq!(sorted.values[2].to_bits(), (-0.0f64).to_bits());
    assert_eq!(sorted.values[3], 1.0);

    let firstseen = parallel_groupby_prod_f64_firstseen_u32(&keys, &values).unwrap();
    assert_eq!(firstseen.keys, vec![7, 9, 11, 13]);
    for (idx, &key) in firstseen.keys.iter().enumerate() {
        let expected = row_order_prod_for_key(&keys, &values, key);
        assert_eq!(firstseen.values[idx].to_bits(), expected.to_bits());
    }
}

#[test]
fn test_groupby_mean_f64() {
    let keys = vec![1, 2, 1, 2, 1];
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = parallel_groupby_mean_f64(&keys, &values).unwrap();

    let mut map: AHashMap<i64, f64> = AHashMap::new();
    for (k, v) in result.keys.iter().zip(result.values.iter()) {
        map.insert(*k, *v);
    }

    assert!((map[&1] - 3.0).abs() < 1e-10);
    assert!((map[&2] - 3.0).abs() < 1e-10);
}

#[test]
fn test_groupby_with_nan() {
    let keys = vec![1, 1, 1];
    let values = vec![1.0, f64::NAN, 2.0];
    let result = parallel_groupby_sum_f64(&keys, &values).unwrap();

    assert!((result.values[0] - 3.0).abs() < 1e-10);
}

#[test]
fn test_groupby_all_nan_group() {
    let keys = vec![1, 1, 2, 2];
    let values = vec![f64::NAN, f64::NAN, 1.0, 2.0];
    let result = parallel_groupby_sum_f64(&keys, &values).unwrap();

    let mut map: AHashMap<i64, f64> = AHashMap::new();
    for (k, v) in result.keys.iter().zip(result.values.iter()) {
        map.insert(*k, *v);
    }

    // All-NaN group returns 0.0, matching pandas behavior: df.groupby('k')['v'].sum()
    assert!((map[&1] - 0.0).abs() < 1e-10);
    assert!((map[&2] - 3.0).abs() < 1e-10);
}

#[test]
fn test_groupby_sorted_orders_by_key() {
    let keys = vec![3, 1, 2, 1, 3];
    let values = vec![1.0, 10.0, 100.0, 1.0, 2.0];
    let result = parallel_groupby_sum_f64_sorted(&keys, &values).unwrap();

    assert_eq!(result.keys, vec![1, 2, 3]);
    assert_eq!(result.values.len(), 3);
    assert!((result.values[0] - 11.0).abs() < 1e-10);
    assert!((result.values[1] - 100.0).abs() < 1e-10);
    assert!((result.values[2] - 3.0).abs() < 1e-10);
}

#[test]
fn test_groupby_sorted_orders_negative_keys() {
    let keys = vec![0, -1, 2, -3, -1];
    let values = vec![1.0, 1.0, 1.0, 1.0, 2.0];
    let result = parallel_groupby_sum_f64_sorted(&keys, &values).unwrap();

    assert_eq!(result.keys, vec![-3, -1, 0, 2]);
    assert_eq!(result.values.len(), 4);
    assert!((result.values[0] - 1.0).abs() < 1e-10);
    assert!((result.values[1] - 3.0).abs() < 1e-10);
    assert!((result.values[2] - 1.0).abs() < 1e-10);
    assert!((result.values[3] - 1.0).abs() < 1e-10);
}

#[test]
fn test_groupby_mean_all_nan_group() {
    let keys = vec![1, 1, 2, 2];
    let values = vec![f64::NAN, f64::NAN, 1.0, 2.0];
    let result = parallel_groupby_mean_f64(&keys, &values).unwrap();

    let mut map: AHashMap<i64, f64> = AHashMap::new();
    for (k, v) in result.keys.iter().zip(result.values.iter()) {
        map.insert(*k, *v);
    }

    assert!(map[&1].is_nan());
    assert!((map[&2] - 1.5).abs() < 1e-10);
}

#[test]
fn test_groupby_min_all_nan_group() {
    let keys = vec![1, 1, 2, 2];
    let values = vec![f64::NAN, f64::NAN, 1.0, 2.0];
    let result = parallel_groupby_min_f64(&keys, &values).unwrap();

    let mut map: AHashMap<i64, f64> = AHashMap::new();
    for (k, v) in result.keys.iter().zip(result.values.iter()) {
        map.insert(*k, *v);
    }

    assert!(map[&1].is_nan());
    assert!((map[&2] - 1.0).abs() < 1e-10);
}

#[test]
fn test_groupby_max_all_nan_group() {
    let keys = vec![1, 1, 2, 2];
    let values = vec![f64::NAN, f64::NAN, 1.0, 2.0];
    let result = parallel_groupby_max_f64(&keys, &values).unwrap();

    let mut map: AHashMap<i64, f64> = AHashMap::new();
    for (k, v) in result.keys.iter().zip(result.values.iter()) {
        map.insert(*k, *v);
    }

    assert!(map[&1].is_nan());
    assert!((map[&2] - 2.0).abs() < 1e-10);
}

#[test]
fn test_public_single_key_firstseen_baseline_matrix_when_representative_inputs() {
    let keys = vec![2, 1, 2, 3, 1, 4];
    let values = vec![10.0, 2.0, 14.0, 8.0, 4.0, 6.0];

    let sum = parallel_groupby_sum_f64_firstseen_u32(&keys, &values).unwrap();
    assert_eq!(sum.keys, vec![2, 1, 3, 4]);
    assert_eq!(sum.values, vec![24.0, 6.0, 8.0, 6.0]);

    let prod = parallel_groupby_prod_f64_firstseen_u32(&keys, &values).unwrap();
    assert_eq!(prod.keys, vec![2, 1, 3, 4]);
    assert_eq!(prod.values, vec![140.0, 8.0, 8.0, 6.0]);

    let var = parallel_groupby_var_f64_firstseen_u32(&keys, &values).unwrap();
    assert_eq!(var.keys, vec![2, 1, 3, 4]);
    assert_eq!(var.values[0], 8.0);
    assert_eq!(var.values[1], 2.0);
    assert!(var.values[2].is_nan());
    assert!(var.values[3].is_nan());

    let std = parallel_groupby_std_f64_firstseen_u32(&keys, &values).unwrap();
    assert_eq!(std.keys, vec![2, 1, 3, 4]);
    assert_eq!(std.values[0], 8.0f64.sqrt());
    assert_eq!(std.values[1], 2.0f64.sqrt());
    assert!(std.values[2].is_nan());
    assert!(std.values[3].is_nan());

    let median = parallel_groupby_median_f64_firstseen_u32(&keys, &values).unwrap();
    assert_eq!(median.keys, vec![2, 1, 3, 4]);
    assert_eq!(median.values, vec![12.0, 3.0, 8.0, 6.0]);

    let empty = parallel_groupby_sum_f64_firstseen_u32(&[], &[]).unwrap();
    assert!(empty.keys.is_empty());
    assert!(empty.values.is_empty());
}
