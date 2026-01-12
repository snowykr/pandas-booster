//! Aggregation primitives for parallel groupby operations.
//!
//! This module defines the [`Aggregator`] trait and implementations for common
//! statistical operations (sum, mean, min, max) on both `f64` and `i64` types.
//!
//! ## Design Decisions
//!
//! - **NaN Handling**: Float aggregators skip NaN values (matching Pandas behavior).
//! - **Integer Output**: All `i64` aggregations return `f64` to match Pandas' overflow
//!   promotion behavior and avoid silent integer clamping.
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

/// Sum aggregator for i64. Uses i128 internally to prevent overflow.
/// Returns f64 to match Pandas overflow promotion behavior. Avoids silent clamping.
#[derive(Clone, Default)]
pub struct SumAggI64 {
    pub sum: i128,
}

impl Aggregator<i64, f64> for SumAggI64 {
    fn init() -> Self {
        Self { sum: 0 }
    }

    fn update(&mut self, value: i64) {
        self.sum += value as i128;
    }

    fn merge(&mut self, other: Self) {
        self.sum += other.sum;
    }

    fn finalize(&self) -> f64 {
        self.sum as f64
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

/// Min aggregator for i64. Returns NaN for empty groups.
#[derive(Clone, Default)]
pub struct MinAggI64 {
    pub min: Option<i64>,
}

impl Aggregator<i64, f64> for MinAggI64 {
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

    fn finalize(&self) -> f64 {
        self.min.map(|v| v as f64).unwrap_or(f64::NAN)
    }
}

/// Max aggregator for i64. Returns NaN for empty groups.
#[derive(Clone, Default)]
pub struct MaxAggI64 {
    pub max: Option<i64>,
}

impl Aggregator<i64, f64> for MaxAggI64 {
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

    fn finalize(&self) -> f64 {
        self.max.map(|v| v as f64).unwrap_or(f64::NAN)
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
        let expected = (i64::MAX as i128 * 2) as f64;
        assert!((result - expected).abs() < 1e6);
    }

    #[test]
    fn test_mean_i64_empty_returns_nan() {
        let agg = MeanAggI64::init();
        assert!(agg.finalize().is_nan());
    }

    #[test]
    fn test_min_i64_empty_returns_nan() {
        let agg = MinAggI64::init();
        assert!(agg.finalize().is_nan());
    }

    #[test]
    fn test_max_i64_empty_returns_nan() {
        let agg = MaxAggI64::init();
        assert!(agg.finalize().is_nan());
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
        assert!((agg1.finalize() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_i64_merge_with_empty() {
        let mut agg1 = MaxAggI64::init();
        agg1.update(10);

        let agg2 = MaxAggI64::init();

        agg1.merge(agg2);
        assert!((agg1.finalize() - 10.0).abs() < 1e-10);
    }
}
