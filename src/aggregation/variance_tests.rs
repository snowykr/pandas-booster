use super::*;

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-10,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn test_var_f64_ddof1_skips_nan_and_std_matches_sqrt_var() {
    let mut var = VarAggF64::init();
    let mut std = StdAggF64::init();

    for value in [1.0, f64::NAN, 2.0, 3.0] {
        var.update(value);
        std.update(value);
    }

    let variance = var.finalize();
    let standard_deviation = std.finalize();

    assert_close(variance, 1.0);
    assert_close(standard_deviation, variance.sqrt());
}

#[test]
fn test_var_and_std_f64_return_nan_for_empty_all_nan_and_singleton_groups() {
    let empty_var = VarAggF64::init();
    let empty_std = StdAggF64::init();
    assert!(empty_var.finalize().is_nan());
    assert!(empty_std.finalize().is_nan());

    let mut all_nan_var = VarAggF64::init();
    let mut all_nan_std = StdAggF64::init();
    all_nan_var.update(f64::NAN);
    all_nan_std.update(f64::NAN);
    assert!(all_nan_var.finalize().is_nan());
    assert!(all_nan_std.finalize().is_nan());

    let mut singleton_var = VarAggF64::init();
    let mut singleton_std = StdAggF64::init();
    singleton_var.update(42.0);
    singleton_std.update(42.0);
    assert!(singleton_var.finalize().is_nan());
    assert!(singleton_std.finalize().is_nan());
}

#[test]
fn test_var_i64_uses_sample_variance_and_returns_f64() {
    let mut var = VarAggI64::init();
    let mut std = StdAggI64::init();

    for value in [1_i64, 2, 3, 4] {
        var.update(value);
        std.update(value);
    }

    let variance = var.finalize();
    let standard_deviation = std.finalize();

    assert_close(variance, 5.0 / 3.0);
    assert_close(standard_deviation, variance.sqrt());
}

#[test]
fn variance_merge_invariance() {
    let mut sequential_var = VarAggF64::init();
    let mut sequential_std = StdAggF64::init();
    for value in [1.0, f64::NAN, 2.0, 5.0, 7.0] {
        sequential_var.update(value);
        sequential_std.update(value);
    }

    let mut left_var = VarAggF64::init();
    let mut left_std = StdAggF64::init();
    for value in [1.0, f64::NAN, 2.0] {
        left_var.update(value);
        left_std.update(value);
    }

    let mut right_var = VarAggF64::init();
    let mut right_std = StdAggF64::init();
    for value in [5.0, 7.0] {
        right_var.update(value);
        right_std.update(value);
    }

    left_var.merge(right_var);
    left_std.merge(right_std);

    let expected_variance: f64 = 91.0 / 12.0;
    let expected_std = expected_variance.sqrt();

    assert_close(sequential_var.finalize(), expected_variance);
    assert_close(left_var.finalize(), expected_variance);
    assert_close(sequential_std.finalize(), expected_std);
    assert_close(left_std.finalize(), expected_std);
}

#[test]
fn test_variance_merge_is_deterministic_and_finalizes_non_negative() {
    let mut left_assoc_var = VarAggF64::init();
    for value in [10.0, 12.0] {
        left_assoc_var.update(value);
    }
    let mut middle_var = VarAggF64::init();
    for value in [14.0, 16.0] {
        middle_var.update(value);
    }
    let mut right_assoc_var = VarAggF64::init();
    for value in [18.0, 20.0] {
        right_assoc_var.update(value);
    }

    let mut merge_left = left_assoc_var.clone();
    merge_left.merge(middle_var.clone());
    merge_left.merge(right_assoc_var.clone());

    let mut merge_right = middle_var;
    merge_right.merge(right_assoc_var);
    let mut merge_right_root = left_assoc_var;
    merge_right_root.merge(merge_right);

    let left_variance = merge_left.finalize();
    let right_variance = merge_right_root.finalize();
    assert_close(left_variance, 14.0);
    assert_close(right_variance, 14.0);
    assert!(left_variance >= 0.0);
    assert!(right_variance >= 0.0);
}
