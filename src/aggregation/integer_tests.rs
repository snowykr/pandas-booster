use super::*;

#[test]
fn test_prod_i64_update_and_merge_wrap() {
    let mut left = ProdAggI64::init();
    left.update(i64::MAX);
    left.update(2);
    assert_eq!(left.finalize(), i64::MAX.wrapping_mul(2));

    let mut right = ProdAggI64::init();
    right.update(3);
    left.merge(right);
    assert_eq!(left.finalize(), i64::MAX.wrapping_mul(2).wrapping_mul(3));
}

#[test]
fn test_sum_i64_uses_i128_no_overflow() {
    let mut agg = SumAggI64::init();
    agg.update(i64::MAX);
    agg.update(i64::MAX);
    let result = agg.finalize();
    let expected = (i64::MAX as i128 * 2) as i64;
    assert_eq!(result, expected);
}

#[test]
fn test_mean_i64_empty_returns_nan() {
    let agg = MeanAggI64::init();
    assert!(agg.finalize().is_nan());
}

#[test]
fn test_median_i64_odd_count_returns_middle_as_f64() {
    let mut agg = MedianAggI64::init();
    for value in [9_i64, 1, 5] {
        agg.update(value);
    }

    assert_eq!(agg.finalize(), 5.0);
}

#[test]
fn test_median_i64_even_count_averages_as_f64_without_overflow() {
    let mut agg = MedianAggI64::init();
    agg.update(i64::MAX - 2);
    agg.update(i64::MAX);

    assert_eq!(agg.finalize(), (i64::MAX - 1) as f64);
}

#[test]
fn test_median_i64_empty_returns_nan() {
    let agg = MedianAggI64::init();

    assert!(agg.finalize().is_nan());
}

#[test]
fn test_median_i64_single_value() {
    let mut agg = MedianAggI64::init();
    agg.update(-7);

    assert_eq!(agg.finalize(), -7.0);
}

#[test]
fn test_median_i64_mixed_sign_even_count() {
    let mut agg = MedianAggI64::init();
    for value in [-10_i64, 5, -2, 12] {
        agg.update(value);
    }

    assert_eq!(agg.finalize(), 1.5);
}

#[test]
fn test_median_i64_merge_order_is_deterministic() {
    let mut left = MedianAggI64::init();
    for value in [100_i64, -10] {
        left.update(value);
    }

    let mut right = MedianAggI64::init();
    for value in [20_i64, 0, 10] {
        right.update(value);
    }

    let mut left_first = left.clone();
    left_first.merge(right.clone());

    let mut right_first = right;
    right_first.merge(left);

    assert_eq!(left_first.finalize(), 10.0);
    assert_eq!(right_first.finalize(), 10.0);
}

#[test]
fn test_median_i64_finalize_owned_matches_finalize() {
    let mut extreme = MedianAggI64::init();
    extreme.update(i64::MAX - 2);
    extreme.update(i64::MAX);
    assert_eq!(extreme.clone().finalize_owned(), extreme.finalize());

    let empty = MedianAggI64::init();
    assert!(empty.finalize_owned().is_nan());
}

#[test]
#[should_panic(expected = "MinAggI64 finalized without values")]
fn test_min_i64_empty_panics() {
    let agg = MinAggI64::init();
    let _ = agg.finalize();
}

#[test]
#[should_panic(expected = "MaxAggI64 finalized without values")]
fn test_max_i64_empty_panics() {
    let agg = MaxAggI64::init();
    let _ = agg.finalize();
}

#[test]
fn test_aggregator_merge() {
    let mut agg1 = SumAggF64::init();
    agg1.update(1.0);
    agg1.update(2.0);

    let mut agg2 = SumAggF64::init();
    agg2.update(3.0);
    agg2.update(4.0);

    agg1.merge(agg2);
    assert!((agg1.finalize() - 10.0).abs() < 1e-10);
}

#[test]
fn test_min_i64_merge() {
    let mut agg1 = MinAggI64::init();
    agg1.update(5);

    let mut agg2 = MinAggI64::init();
    agg2.update(2);

    agg1.merge(agg2);
    assert_eq!(agg1.finalize(), 2);
}

#[test]
fn test_max_i64_merge_with_empty() {
    let mut agg1 = MaxAggI64::init();
    agg1.update(10);

    let agg2 = MaxAggI64::init();

    agg1.merge(agg2);
    assert_eq!(agg1.finalize(), 10);
}

#[test]
fn test_count_i64() {
    let mut agg = CountAggI64::init();
    agg.update(1);
    agg.update(2);
    agg.update(3);
    assert_eq!(agg.finalize(), 3);
}

#[test]
fn test_count_merge() {
    let mut agg1 = CountAggF64::init();
    agg1.update(1.0);
    agg1.update(2.0);

    let mut agg2 = CountAggF64::init();
    agg2.update(3.0);

    agg1.merge(agg2);
    assert_eq!(agg1.finalize(), 3);
}
