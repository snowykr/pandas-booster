pub(crate) fn median_f64_from_values(mut values: Vec<f64>) -> f64 {
    median_f64_from_mut_slice(&mut values)
}

pub(crate) fn median_f64_from_mut_slice(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }

    if let [left, right] = values {
        if *left == 0.0 && *right == 0.0 {
            return *left;
        }
    }

    let mid = values.len() / 2;
    let is_odd = values.len() % 2 == 1;
    let (lower, median, _) = values.select_nth_unstable_by(mid, f64::total_cmp);

    if is_odd {
        *median
    } else if let Some((&first, rest)) = lower.split_first() {
        let lower_max = rest.iter().copied().fold(first, |current, value| {
            if current.total_cmp(&value).is_lt() {
                value
            } else {
                current
            }
        });
        average_f64_middle_values(lower_max, *median)
    } else {
        f64::NAN
    }
}

pub(crate) fn average_f64_middle_values(lower: f64, upper: f64) -> f64 {
    // Keep pandas/NumPy observable semantics for even-length float medians:
    // very large same-sign finite middle values intentionally overflow to
    // +/-inf instead of using a numerically stable midpoint formula.
    (lower + upper) / 2.0
}

pub(crate) fn median_i64_from_values(mut values: Vec<i64>) -> f64 {
    median_i64_from_mut_slice(&mut values)
}

pub(crate) fn median_i64_from_mut_slice(values: &mut [i64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }

    let mid = values.len() / 2;
    let is_odd = values.len() % 2 == 1;
    let (lower, median, _) = values.select_nth_unstable(mid);

    if is_odd {
        *median as f64
    } else if let Some((&first, rest)) = lower.split_first() {
        let lower_max = rest.iter().copied().fold(first, i64::max);
        (lower_max as f64 + *median as f64) / 2.0
    } else {
        f64::NAN
    }
}

#[cfg(test)]
mod tests {
    use crate::aggregation::{Aggregator, MedianAggF64, MedianAggI64};

    use super::{average_f64_middle_values, median_f64_from_values, median_i64_from_values};

    #[test]
    fn median_helper_f64_skips_nan_and_overflows_like_existing_aggregator() {
        // Given: the f64 cases that encode current MedianAggF64 behavior.
        let helper_input = vec![f64::MAX, f64::MAX];

        let mut overflow_agg = MedianAggF64::init();
        overflow_agg.update(f64::MAX);
        overflow_agg.update(f64::NAN);
        overflow_agg.update(f64::MAX);

        let mut all_nan_agg = MedianAggF64::init();
        all_nan_agg.update(f64::NAN);
        all_nan_agg.update(f64::NAN);

        // When: the helper and aggregator both finalize medians.
        let helper_overflow = median_f64_from_values(helper_input);
        let helper_empty = median_f64_from_values(Vec::new());

        // Then: NaN exclusion remains owned by the aggregator, while helper
        // median math preserves overflow and empty-group results.
        assert_eq!(helper_overflow, f64::INFINITY);
        assert_eq!(overflow_agg.finalize(), helper_overflow);
        assert!(helper_empty.is_nan());
        assert!(all_nan_agg.finalize().is_nan());
        assert_eq!(
            average_f64_middle_values(-0.0, 0.0).to_bits(),
            0.0_f64.to_bits()
        );
    }

    #[test]
    fn median_helper_f64_two_value_zero_tie_matches_pandas_bits() {
        assert_eq!(
            median_f64_from_values(vec![-0.0, 0.0]).to_bits(),
            (-0.0_f64).to_bits()
        );
        assert_eq!(
            median_f64_from_values(vec![0.0, -0.0]).to_bits(),
            0.0_f64.to_bits()
        );
    }

    #[test]
    fn median_helper_i64_averages_extreme_middle_values_like_existing_aggregator() {
        // Given: i64 median inputs covering extremes, precision loss above
        // 2**53, and negative/positive even-middle pairs.
        let cases = [
            (vec![i64::MIN, i64::MAX], 0.0),
            (
                vec![(1_i64 << 53) + 1, (1_i64 << 53) + 3],
                ((1_i64 << 53) + 1) as f64 / 2.0 + ((1_i64 << 53) + 3) as f64 / 2.0,
            ),
            (vec![-9_i64, 4], -2.5),
            (vec![i64::MAX - 2, i64::MAX], (i64::MAX - 1) as f64),
        ];

        for (values, expected) in cases {
            let mut agg = MedianAggI64::init();
            for value in values.iter().copied() {
                agg.update(value);
            }

            // When: the shared helper and aggregator finalize the same values.
            let helper_result = median_i64_from_values(values);

            // Then: the helper preserves f64-returning even-middle semantics.
            assert_eq!(helper_result, expected);
            assert_eq!(helper_result, agg.finalize());
        }
    }

    #[test]
    fn median_helper_and_median_agg_finalize_owned_match_finalize() {
        // Given: f64 and i64 aggregators with unsorted even-length values.
        let mut f64_agg = MedianAggF64::init();
        for value in [10.0, f64::NAN, 2.0, 4.0, 8.0] {
            f64_agg.update(value);
        }

        let mut i64_agg = MedianAggI64::init();
        for value in [-10_i64, 5, -2, 12] {
            i64_agg.update(value);
        }

        // When: finalize_owned consumes the aggregators after helper extraction.
        let f64_owned = f64_agg.clone().finalize_owned();
        let i64_owned = i64_agg.clone().finalize_owned();

        // Then: owned and borrowed finalization still agree.
        assert_eq!(f64_owned, f64_agg.finalize());
        assert_eq!(i64_owned, i64_agg.finalize());
    }
}
