use smallvec::SmallVec;

use super::median::median_f64_from_values;
use super::Aggregator;

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

#[derive(Clone, Default)]
pub struct MedianAggF64 {
    pub values: SmallVec<[f64; 4]>,
}

impl Aggregator<f64, f64> for MedianAggF64 {
    fn init() -> Self {
        Self {
            values: SmallVec::new(),
        }
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
        median_f64_from_values(self.values.clone().into_vec())
    }

    fn finalize_owned(self) -> f64 {
        median_f64_from_values(self.values.into_vec())
    }
}

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
