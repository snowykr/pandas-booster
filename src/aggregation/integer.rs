use smallvec::SmallVec;

use super::median::median_i64_from_values;
use super::Aggregator;

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

#[derive(Clone, Default)]
pub struct MedianAggI64 {
    pub values: SmallVec<[i64; 4]>,
}

impl Aggregator<i64, f64> for MedianAggI64 {
    fn init() -> Self {
        Self {
            values: SmallVec::new(),
        }
    }

    fn update(&mut self, value: i64) {
        self.values.push(value);
    }

    fn merge(&mut self, other: Self) {
        self.values.reserve(other.values.len());
        self.values.extend(other.values);
    }

    fn finalize(&self) -> f64 {
        median_i64_from_values(self.values.clone().into_vec())
    }

    fn finalize_owned(self) -> f64 {
        median_i64_from_values(self.values.into_vec())
    }
}

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
