//! Aggregation primitives for parallel groupby operations.
//!
//! This module defines the [`Aggregator`] trait and implementations for common
//! statistical operations (sum, prod, mean, min, max) on both `f64` and `i64` types.
//!
//! ## Design Decisions
//!
//! - **NaN Handling**: Float aggregators skip NaN values (matching Pandas behavior).
//! - **Integer Output**: Integer aggregations follow Pandas semantics:
//!   `sum/prod/min/max/count` return integer outputs and `mean` returns `f64`.
//! - **Thread Safety**: All aggregators implement `Send + Sync` for parallel execution.

/// Core trait for streaming aggregation with support for parallel merge.
///
/// Implementors must be thread-safe (`Send + Sync`) to support Rayon's parallel fold/reduce.
pub trait Aggregator<T, O>: Send + Sync {
    /// Creates a new aggregator with identity state.
    fn init() -> Self;
    /// Incorporates a single value into the aggregation.
    fn update(&mut self, value: T);
    /// Merges another aggregator's state into this one (for parallel reduction).
    fn merge(&mut self, other: Self);
    /// Computes the final aggregated result.
    fn finalize(&self) -> O;
    /// Computes the final result when the accumulator is no longer needed.
    ///
    /// The default implementation preserves the immutable finalization contract,
    /// while Vec-backed aggregators can override this to consume their buffers
    /// and avoid a materialization-time clone. Implementations must return the
    /// same logical result as [`Aggregator::finalize`] for the same state.
    fn finalize_owned(self) -> O
    where
        Self: Sized,
    {
        self.finalize()
    }
}

/// Sum aggregator for f64. Skips NaN values.
#[derive(Clone, Default)]
pub struct SumAggF64 {
    pub sum: f64,
}

impl Aggregator<f64, f64> for SumAggF64 {
    fn init() -> Self {
        Self { sum: 0.0 }
    }

    fn update(&mut self, value: f64) {
        if !value.is_nan() {
            self.sum += value;
        }
    }

    fn merge(&mut self, other: Self) {
        self.sum += other.sum;
    }

    fn finalize(&self) -> f64 {
        self.sum
    }
}

/// Product aggregator for f64. Skips input NaN values only.
///
/// The identity is `1.0`, so empty/all-input-NaN groups finalize to `1.0`
/// (pandas `min_count=0` behavior). Arithmetic NaNs produced by
/// multiplication, such as `inf * 0`, are preserved because only the incoming
/// value is checked before multiplying.
#[derive(Clone, Default)]
pub struct ProdAggF64 {
    pub prod: f64,
}

impl Aggregator<f64, f64> for ProdAggF64 {
    fn init() -> Self {
        Self { prod: 1.0 }
    }

    fn update(&mut self, value: f64) {
        if !value.is_nan() {
            self.prod *= value;
        }
    }

    fn merge(&mut self, other: Self) {
        self.prod *= other.prod;
    }

    fn finalize(&self) -> f64 {
        self.prod
    }
}

/// Mean aggregator for f64. Returns NaN for empty/all-NaN groups.
#[derive(Clone, Default)]
pub struct MeanAggF64 {
    pub sum: f64,
    pub count: u64,
}

impl Aggregator<f64, f64> for MeanAggF64 {
    fn init() -> Self {
        Self { sum: 0.0, count: 0 }
    }

    fn update(&mut self, value: f64) {
        if !value.is_nan() {
            self.sum += value;
            self.count += 1;
        }
    }

    fn merge(&mut self, other: Self) {
        self.sum += other.sum;
        self.count += other.count;
    }

    fn finalize(&self) -> f64 {
        if self.count == 0 {
            f64::NAN
        } else {
            self.sum / self.count as f64
        }
    }
}

/// Median aggregator for f64. Skips NaN values and returns NaN for empty/all-NaN groups.
#[derive(Clone, Default)]
pub struct MedianAggF64 {
    pub values: Vec<f64>,
}

impl Aggregator<f64, f64> for MedianAggF64 {
    fn init() -> Self {
        Self { values: Vec::new() }
    }

    fn update(&mut self, value: f64) {
        if !value.is_nan() {
            self.values.push(value);
        }
    }

    fn merge(&mut self, other: Self) {
        self.values.reserve(other.values.len());
        self.values.extend(other.values);
    }

    fn finalize(&self) -> f64 {
        median_f64_from_values(self.values.clone())
    }

    fn finalize_owned(self) -> f64 {
        median_f64_from_values(self.values)
    }
}

fn median_f64_from_values(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }

    let mid = values.len() / 2;
    let is_odd = values.len() % 2 == 1;
    let (lower, median, _) = values.select_nth_unstable_by(mid, f64::total_cmp);

    if is_odd {
        *median
    } else {
        let lower_max = lower
            .iter()
            .copied()
            .max_by(f64::total_cmp)
            .expect("even-length median requires a lower partition");
        (lower_max + *median) / 2.0
    }
}

/// Shared mergeable variance state using a numerically stable mean/M2 update.
#[derive(Clone, Default)]
struct VarianceState {
    count: u64,
    mean: f64,
    m2: f64,
}

impl VarianceState {
    fn update(&mut self, value: f64) {
        self.count += 1;

        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    fn merge(&mut self, other: Self) {
        if other.count == 0 {
            return;
        }

        if self.count == 0 {
            *self = other;
            return;
        }

        let self_count = self.count as f64;
        let other_count = other.count as f64;
        let total_count = self.count + other.count;
        let total_count_f64 = total_count as f64;
        let delta = other.mean - self.mean;

        self.m2 += other.m2 + delta * delta * (self_count * other_count / total_count_f64);
        self.mean += delta * (other_count / total_count_f64);
        self.count = total_count;
    }

    fn sample_variance(&self) -> f64 {
        if self.count <= 1 {
            return f64::NAN;
        }

        let variance = self.m2 / (self.count - 1) as f64;
        if variance < 0.0 {
            0.0
        } else {
            variance
        }
    }
}

/// Variance aggregator for f64. Skips NaN values and returns sample variance.
#[derive(Clone, Default)]
pub struct VarAggF64 {
    state: VarianceState,
}

impl Aggregator<f64, f64> for VarAggF64 {
    fn init() -> Self {
        Self::default()
    }

    fn update(&mut self, value: f64) {
        if !value.is_nan() {
            self.state.update(value);
        }
    }

    fn merge(&mut self, other: Self) {
        self.state.merge(other.state);
    }

    fn finalize(&self) -> f64 {
        self.state.sample_variance()
    }
}

/// Standard deviation aggregator for f64. Skips NaN values and derives from variance state.
#[derive(Clone, Default)]
pub struct StdAggF64 {
    state: VarianceState,
}

impl Aggregator<f64, f64> for StdAggF64 {
    fn init() -> Self {
        Self::default()
    }

    fn update(&mut self, value: f64) {
        if !value.is_nan() {
            self.state.update(value);
        }
    }

    fn merge(&mut self, other: Self) {
        self.state.merge(other.state);
    }

    fn finalize(&self) -> f64 {
        let variance = self.state.sample_variance();
        if variance.is_nan() {
            variance
        } else {
            variance.sqrt()
        }
    }
}

/// Min aggregator for f64. Returns NaN for empty/all-NaN groups.
#[derive(Clone)]
pub struct MinAggF64 {
    pub min: f64,
    pub seen_valid: bool,
}

impl Default for MinAggF64 {
    fn default() -> Self {
        Self {
            min: f64::INFINITY,
            seen_valid: false,
        }
    }
}

impl Aggregator<f64, f64> for MinAggF64 {
    fn init() -> Self {
        Self::default()
    }

    fn update(&mut self, value: f64) {
        if !value.is_nan() {
            self.seen_valid = true;
            if value < self.min {
                self.min = value;
            }
        }
    }

    fn merge(&mut self, other: Self) {
        if other.seen_valid {
            self.seen_valid = true;
            if other.min < self.min {
                self.min = other.min;
            }
        }
    }

    fn finalize(&self) -> f64 {
        if self.seen_valid {
            self.min
        } else {
            f64::NAN
        }
    }
}

/// Max aggregator for f64. Returns NaN for empty/all-NaN groups.
#[derive(Clone)]
pub struct MaxAggF64 {
    pub max: f64,
    pub seen_valid: bool,
}

impl Default for MaxAggF64 {
    fn default() -> Self {
        Self {
            max: f64::NEG_INFINITY,
            seen_valid: false,
        }
    }
}

impl Aggregator<f64, f64> for MaxAggF64 {
    fn init() -> Self {
        Self::default()
    }

    fn update(&mut self, value: f64) {
        if !value.is_nan() {
            self.seen_valid = true;
            if value > self.max {
                self.max = value;
            }
        }
    }

    fn merge(&mut self, other: Self) {
        if other.seen_valid {
            self.seen_valid = true;
            if other.max > self.max {
                self.max = other.max;
            }
        }
    }

    fn finalize(&self) -> f64 {
        if self.seen_valid {
            self.max
        } else {
            f64::NAN
        }
    }
}

/// Sum aggregator for i64. Uses i128 internally and truncates back to i64,
/// matching Pandas integer-overflow wrap behavior.
#[derive(Clone, Default)]
pub struct SumAggI64 {
    pub sum: i128,
}

impl Aggregator<i64, i64> for SumAggI64 {
    fn init() -> Self {
        Self { sum: 0 }
    }

    fn update(&mut self, value: i64) {
        self.sum += value as i128;
    }

    fn merge(&mut self, other: Self) {
        self.sum += other.sum;
    }

    fn finalize(&self) -> i64 {
        self.sum as i64
    }
}

/// Product aggregator for i64. Uses explicit wrapping multiplication.
#[derive(Clone, Default)]
pub struct ProdAggI64 {
    pub prod: i64,
}

impl Aggregator<i64, i64> for ProdAggI64 {
    fn init() -> Self {
        Self { prod: 1 }
    }

    fn update(&mut self, value: i64) {
        self.prod = self.prod.wrapping_mul(value);
    }

    fn merge(&mut self, other: Self) {
        self.prod = self.prod.wrapping_mul(other.prod);
    }

    fn finalize(&self) -> i64 {
        self.prod
    }
}

/// Mean aggregator for i64. Returns NaN for empty groups.
#[derive(Clone, Default)]
pub struct MeanAggI64 {
    pub sum: i128,
    pub count: u64,
}

impl Aggregator<i64, f64> for MeanAggI64 {
    fn init() -> Self {
        Self { sum: 0, count: 0 }
    }

    fn update(&mut self, value: i64) {
        self.sum += value as i128;
        self.count += 1;
    }

    fn merge(&mut self, other: Self) {
        self.sum += other.sum;
        self.count += other.count;
    }

    fn finalize(&self) -> f64 {
        if self.count == 0 {
            f64::NAN
        } else {
            self.sum as f64 / self.count as f64
        }
    }
}

/// Median aggregator for i64. Returns f64 and NaN for empty groups.
#[derive(Clone, Default)]
pub struct MedianAggI64 {
    pub values: Vec<i64>,
}

impl Aggregator<i64, f64> for MedianAggI64 {
    fn init() -> Self {
        Self { values: Vec::new() }
    }

    fn update(&mut self, value: i64) {
        self.values.push(value);
    }

    fn merge(&mut self, other: Self) {
        self.values.reserve(other.values.len());
        self.values.extend(other.values);
    }

    fn finalize(&self) -> f64 {
        median_i64_from_values(self.values.clone())
    }

    fn finalize_owned(self) -> f64 {
        median_i64_from_values(self.values)
    }
}

fn median_i64_from_values(mut values: Vec<i64>) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }

    let mid = values.len() / 2;
    let is_odd = values.len() % 2 == 1;
    let (lower, median, _) = values.select_nth_unstable(mid);

    if is_odd {
        *median as f64
    } else {
        let lower_max = lower
            .iter()
            .copied()
            .max()
            .expect("even-length median requires a lower partition");
        (lower_max as f64 + *median as f64) / 2.0
    }
}

/// Variance aggregator for i64. Returns sample variance as f64.
#[derive(Clone, Default)]
pub struct VarAggI64 {
    state: VarianceState,
}

impl Aggregator<i64, f64> for VarAggI64 {
    fn init() -> Self {
        Self::default()
    }

    fn update(&mut self, value: i64) {
        self.state.update(value as f64);
    }

    fn merge(&mut self, other: Self) {
        self.state.merge(other.state);
    }

    fn finalize(&self) -> f64 {
        self.state.sample_variance()
    }
}

/// Standard deviation aggregator for i64. Returns sample standard deviation as f64.
#[derive(Clone, Default)]
pub struct StdAggI64 {
    state: VarianceState,
}

impl Aggregator<i64, f64> for StdAggI64 {
    fn init() -> Self {
        Self::default()
    }

    fn update(&mut self, value: i64) {
        self.state.update(value as f64);
    }

    fn merge(&mut self, other: Self) {
        self.state.merge(other.state);
    }

    fn finalize(&self) -> f64 {
        let variance = self.state.sample_variance();
        if variance.is_nan() {
            variance
        } else {
            variance.sqrt()
        }
    }
}

/// Min aggregator for i64.
#[derive(Clone, Default)]
pub struct MinAggI64 {
    pub min: Option<i64>,
}

impl Aggregator<i64, i64> for MinAggI64 {
    fn init() -> Self {
        Self::default()
    }

    fn update(&mut self, value: i64) {
        self.min = Some(match self.min {
            Some(current) => current.min(value),
            None => value,
        });
    }

    fn merge(&mut self, other: Self) {
        self.min = match (self.min, other.min) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
    }

    fn finalize(&self) -> i64 {
        self.min.expect("MinAggI64 finalized without values")
    }
}

/// Max aggregator for i64.
#[derive(Clone, Default)]
pub struct MaxAggI64 {
    pub max: Option<i64>,
}

impl Aggregator<i64, i64> for MaxAggI64 {
    fn init() -> Self {
        Self::default()
    }

    fn update(&mut self, value: i64) {
        self.max = Some(match self.max {
            Some(current) => current.max(value),
            None => value,
        });
    }

    fn merge(&mut self, other: Self) {
        self.max = match (self.max, other.max) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
    }

    fn finalize(&self) -> i64 {
        self.max.expect("MaxAggI64 finalized without values")
    }
}

/// Count aggregator for f64. Counts non-NaN values (matching Pandas behavior).
#[derive(Clone, Default)]
pub struct CountAggF64 {
    pub count: u64,
}

impl Aggregator<f64, i64> for CountAggF64 {
    fn init() -> Self {
        Self { count: 0 }
    }

    fn update(&mut self, value: f64) {
        if !value.is_nan() {
            self.count += 1;
        }
    }

    fn merge(&mut self, other: Self) {
        self.count += other.count;
    }

    fn finalize(&self) -> i64 {
        self.count as i64
    }
}

/// Count aggregator for i64. Counts all values (integers have no NaN).
#[derive(Clone, Default)]
pub struct CountAggI64 {
    pub count: u64,
}

impl Aggregator<i64, i64> for CountAggI64 {
    fn init() -> Self {
        Self { count: 0 }
    }

    fn update(&mut self, _value: i64) {
        self.count += 1;
    }

    fn merge(&mut self, other: Self) {
        self.count += other.count;
    }

    fn finalize(&self) -> i64 {
        self.count as i64
    }
}

#[cfg(test)]
mod tests {
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

        let empty = MedianAggF64::init();
        assert!(empty.finalize_owned().is_nan());

        let mut all_nan = MedianAggF64::init();
        all_nan.update(f64::NAN);
        all_nan.update(f64::NAN);
        assert!(all_nan.finalize_owned().is_nan());
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

        let mut mixed_sign = MedianAggI64::init();
        for value in [-10_i64, 5, -2, 12] {
            mixed_sign.update(value);
        }
        assert_eq!(mixed_sign.clone().finalize_owned(), mixed_sign.finalize());

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

        let mut left_assoc_std = StdAggF64::init();
        for value in [10.0, 12.0] {
            left_assoc_std.update(value);
        }
        let mut middle_std = StdAggF64::init();
        for value in [14.0, 16.0] {
            middle_std.update(value);
        }
        let mut right_assoc_std = StdAggF64::init();
        for value in [18.0, 20.0] {
            right_assoc_std.update(value);
        }

        let mut merge_left_std = left_assoc_std.clone();
        merge_left_std.merge(middle_std.clone());
        merge_left_std.merge(right_assoc_std.clone());

        let mut merge_right_std = middle_std;
        merge_right_std.merge(right_assoc_std);
        let mut merge_right_std_root = left_assoc_std;
        merge_right_std_root.merge(merge_right_std);

        let left_standard_deviation = merge_left_std.finalize();
        let right_standard_deviation = merge_right_std_root.finalize();
        assert_close(left_standard_deviation, 14.0_f64.sqrt());
        assert_close(right_standard_deviation, 14.0_f64.sqrt());
        assert!(left_standard_deviation >= 0.0);
        assert!(right_standard_deviation >= 0.0);
    }
}
