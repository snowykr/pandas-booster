use super::api_f64::parallel_groupby_prod_f64_sorted;
use super::engine::parallel_groupby_prod_f64_ordered_impl;
use super::order::reorder_single_result_by_key;
use super::prod_ordered::groupby_prod_f64_ordered_low;
use super::routing::should_use_partitioned_prod_engine;
use super::test_support::row_order_prod_for_key;

fn high_cardinality_duplicate_prod_data() -> (Vec<i64>, Vec<f64>, i64) {
    let target_key = -7;
    let mut keys = Vec::with_capacity(4_105);
    let mut values = Vec::with_capacity(4_105);

    keys.push(10_000);
    values.push(2.0);
    keys.push(target_key);
    values.push(1e-308);

    for key in 0..4_100 {
        if key == 97 {
            keys.push(target_key);
            values.push(1e308);
        }

        keys.push(key);
        values.push(1.0);
    }

    keys.push(target_key);
    values.push(1e308);

    (keys, values, target_key)
}

#[test]
fn partitioned_prod_preserves_duplicate_key_row_order_when_firstseen() {
    let (keys, values, target_key) = high_cardinality_duplicate_prod_data();
    assert!(should_use_partitioned_prod_engine(&keys));

    let result = parallel_groupby_prod_f64_ordered_impl(&keys, &values).unwrap();

    let target_position = result
        .keys
        .iter()
        .position(|&key| key == target_key)
        .unwrap();
    assert_eq!(result.keys[0], 10_000);
    assert_eq!(result.keys[1], target_key);
    assert_eq!(
        result.values[target_position].to_bits(),
        row_order_prod_for_key(&keys, &values, target_key).to_bits()
    );
}

#[test]
fn partitioned_prod_preserves_duplicate_key_row_order_when_sorted() {
    let (keys, values, target_key) = high_cardinality_duplicate_prod_data();
    assert!(should_use_partitioned_prod_engine(&keys));

    let result = parallel_groupby_prod_f64_sorted(&keys, &values).unwrap();

    assert_eq!(result.keys[0], target_key);
    assert_eq!(
        result.values[0].to_bits(),
        row_order_prod_for_key(&keys, &values, target_key).to_bits()
    );
}

#[test]
fn prod_ordered_impl_preserves_interleaved_row_order_semantics() {
    let keys = vec![
        7, 9, 7, 11, 9, 7, 13, 11, 13, 15, 15, 17, 17, 19, 19, 21, 21,
    ];
    let values = vec![
        1e308,
        0.0,
        1e308,
        -0.0,
        f64::INFINITY,
        1e-308,
        f64::NAN,
        2.0,
        f64::NAN,
        f64::INFINITY,
        0.0,
        1e-308,
        1e308,
        f64::NAN,
        3.0,
        2.0,
        f64::NAN,
    ];

    let result = parallel_groupby_prod_f64_ordered_impl(&keys, &values).unwrap();

    assert_eq!(result.keys, vec![7, 9, 11, 13, 15, 17, 19, 21]);
    assert_eq!(result.values[0].to_bits(), f64::INFINITY.to_bits());
    assert!(result.values[1].is_nan());
    assert_eq!(result.values[2].to_bits(), (-0.0f64).to_bits());
    assert_eq!(result.values[3].to_bits(), 1.0f64.to_bits());
    assert!(result.values[4].is_nan());
    assert_eq!(result.values[5].to_bits(), 0.9999999999999999f64.to_bits());
    assert_eq!(result.values[6].to_bits(), 3.0f64.to_bits());
    assert_eq!(result.values[7].to_bits(), 2.0f64.to_bits());
}

#[test]
fn prod_ordered_low_preserves_sorted_and_firstseen_order() {
    let keys = vec![3, 1, 2, 1, 3, 2];
    let values = vec![2.0, 10.0, 100.0, 0.5, 4.0, 0.25];

    let firstseen = groupby_prod_f64_ordered_low::<u32>(&keys, &values).unwrap();

    assert_eq!(firstseen.keys, vec![3, 1, 2]);
    assert_eq!(firstseen.values, vec![8.0, 5.0, 25.0]);

    let mut sorted = groupby_prod_f64_ordered_low::<u64>(&keys, &values).unwrap();
    reorder_single_result_by_key(&mut sorted);

    assert_eq!(sorted.keys, vec![1, 2, 3]);
    assert_eq!(sorted.values, vec![5.0, 25.0, 8.0]);
}

#[test]
fn prod_ordered_low_validates_lengths() {
    pyo3::Python::initialize();

    let keys = vec![1, 2, 1];
    let values = vec![2.0, 3.0];

    let error = groupby_prod_f64_ordered_low::<u32>(&keys, &values).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("keys and values must have same length"),
        "unexpected error: {error}"
    );
}
