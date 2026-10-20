use super::Aggregator;

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
