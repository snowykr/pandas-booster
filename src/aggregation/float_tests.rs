use super::*;

#[test]
fn test_sum_f64_skips_nan() {
    let mut agg = SumAggF64::init();
    agg.update(1.0);
    agg.update(f64::NAN);
    agg.update(2.0);
    assert!((agg.finalize() - 3.0).abs() < 1e-10);
}

#[test]
fn test_sum_f64_all_nan_returns_zero() {
    let mut agg = SumAggF64::init();
    agg.update(f64::NAN);
    agg.update(f64::NAN);
    assert!((agg.finalize() - 0.0).abs() < 1e-10);
}

#[test]
fn test_prod_f64_skips_input_nan_and_all_nan_returns_one() {
    let mut agg = ProdAggF64::init();
    agg.update(f64::NAN);
    agg.update(2.0);
    agg.update(3.0);
    assert_eq!(agg.finalize(), 6.0);

    let mut all_nan = ProdAggF64::init();
    all_nan.update(f64::NAN);
    all_nan.update(f64::NAN);
    assert_eq!(all_nan.finalize(), 1.0);
}

#[test]
fn test_prod_f64_preserves_arithmetic_nan() {
    let mut agg = ProdAggF64::init();
    agg.update(f64::INFINITY);
    agg.update(0.0);
    assert!(agg.finalize().is_nan());

    agg.update(f64::NAN);
    assert!(agg.finalize().is_nan());
}

#[test]
fn test_mean_f64_empty_returns_nan() {
    let agg = MeanAggF64::init();
    assert!(agg.finalize().is_nan());
}

#[test]
fn test_mean_f64_all_nan_returns_nan() {
    let mut agg = MeanAggF64::init();
    agg.update(f64::NAN);
    agg.update(f64::NAN);
    assert!(agg.finalize().is_nan());
}

#[test]
fn test_mean_f64_with_values() {
    let mut agg = MeanAggF64::init();
    agg.update(2.0);
    agg.update(4.0);
    assert!((agg.finalize() - 3.0).abs() < 1e-10);
}

#[test]
fn test_median_f64_odd_count_skips_nan() {
    let mut agg = MedianAggF64::init();
    for value in [3.0, f64::NAN, 1.0, 2.0] {
        agg.update(value);
    }

    assert_eq!(agg.finalize(), 2.0);
}

#[test]
fn test_median_f64_even_count_averages_middle_values() {
    let mut agg = MedianAggF64::init();
    for value in [10.0, 2.0, 4.0, 8.0] {
        agg.update(value);
    }

    assert_eq!(agg.finalize(), 6.0);
}

#[test]
fn test_median_f64_even_signed_zero_matches_pandas_bits() {
    let mut negative_then_positive = MedianAggF64::init();
    negative_then_positive.update(-0.0);
    negative_then_positive.update(0.0);

    let mut positive_then_negative = MedianAggF64::init();
    positive_then_negative.update(0.0);
    positive_then_negative.update(-0.0);

    assert_eq!(
        negative_then_positive.finalize().to_bits(),
        (-0.0f64).to_bits()
    );
    assert_eq!(
        positive_then_negative.finalize().to_bits(),
        0.0f64.to_bits()
    );
}

#[test]
fn test_median_f64_even_count_uses_pandas_overflow_semantics() {
    let mut positive = MedianAggF64::init();
    positive.update(f64::MAX);
    positive.update(f64::MAX);
    assert_eq!(positive.finalize(), f64::INFINITY);

    let mut negative = MedianAggF64::init();
    negative.update(-f64::MAX);
    negative.update(-f64::MAX);
    assert_eq!(negative.finalize(), f64::NEG_INFINITY);

    let mut opposite_sign = MedianAggF64::init();
    opposite_sign.update(-f64::MAX);
    opposite_sign.update(f64::MAX);
    assert_eq!(opposite_sign.finalize(), 0.0);
}

#[test]
fn test_median_f64_empty_and_all_nan_return_nan() {
    let empty = MedianAggF64::init();
    assert!(empty.finalize().is_nan());

    let mut all_nan = MedianAggF64::init();
    all_nan.update(f64::NAN);
    all_nan.update(f64::NAN);
    assert!(all_nan.finalize().is_nan());
}

#[test]
fn test_median_f64_single_value() {
    let mut agg = MedianAggF64::init();
    agg.update(42.5);

    assert_eq!(agg.finalize(), 42.5);
}

#[test]
fn test_median_f64_merge_order_is_deterministic() {
    let mut left = MedianAggF64::init();
    for value in [9.0, 1.0] {
        left.update(value);
    }

    let mut right = MedianAggF64::init();
    for value in [5.0, f64::NAN, 3.0] {
        right.update(value);
    }

    let mut left_first = left.clone();
    left_first.merge(right.clone());

    let mut right_first = right;
    right_first.merge(left);

    assert_eq!(left_first.finalize(), 4.0);
    assert_eq!(right_first.finalize(), 4.0);
}

#[test]
fn test_median_f64_finalize_owned_matches_finalize() {
    let mut agg = MedianAggF64::init();
    for value in [10.0, f64::NAN, 2.0, 4.0, 8.0] {
        agg.update(value);
    }

    assert_eq!(agg.clone().finalize_owned(), agg.finalize());
}

#[test]
fn test_min_f64_empty_returns_nan() {
    let agg = MinAggF64::init();
    assert!(agg.finalize().is_nan());
}

#[test]
fn test_min_f64_all_nan_returns_nan() {
    let mut agg = MinAggF64::init();
    agg.update(f64::NAN);
    assert!(agg.finalize().is_nan());
}

#[test]
fn test_min_f64_with_values() {
    let mut agg = MinAggF64::init();
    agg.update(5.0);
    agg.update(2.0);
    agg.update(f64::NAN);
    agg.update(3.0);
    assert!((agg.finalize() - 2.0).abs() < 1e-10);
}

#[test]
fn test_max_f64_empty_returns_nan() {
    let agg = MaxAggF64::init();
    assert!(agg.finalize().is_nan());
}

#[test]
fn test_max_f64_all_nan_returns_nan() {
    let mut agg = MaxAggF64::init();
    agg.update(f64::NAN);
    assert!(agg.finalize().is_nan());
}

#[test]
fn test_max_f64_with_values() {
    let mut agg = MaxAggF64::init();
    agg.update(1.0);
    agg.update(f64::NAN);
    agg.update(5.0);
    agg.update(3.0);
    assert!((agg.finalize() - 5.0).abs() < 1e-10);
}

#[test]
fn test_count_f64_skips_nan() {
    let mut agg = CountAggF64::init();
    agg.update(1.0);
    agg.update(f64::NAN);
    agg.update(2.0);
    assert_eq!(agg.finalize(), 2);
}

#[test]
fn test_count_f64_all_nan_returns_zero() {
    let mut agg = CountAggF64::init();
    agg.update(f64::NAN);
    agg.update(f64::NAN);
    assert_eq!(agg.finalize(), 0);
}
