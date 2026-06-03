use super::result::GroupByResultF64;
use crate::aggregation::Aggregator;
use pyo3::prelude::PyResult;
use rayon::ThreadPoolBuilder;

pub(super) struct NonCloneVecAgg {
    values: Vec<i64>,
}

impl Aggregator<i64, usize> for NonCloneVecAgg {
    fn init() -> Self {
        Self { values: Vec::new() }
    }

    fn update(&mut self, value: i64) {
        self.values.push(value);
    }

    fn merge(&mut self, other: Self) {
        self.values.extend(other.values);
    }

    fn finalize(&self) -> usize {
        panic!("materialization should use finalize_owned for NonCloneVecAgg")
    }

    fn finalize_owned(self) -> usize {
        self.values.len()
    }
}

#[derive(Clone, Default)]
pub(super) struct OwnedOnlyAgg {
    values: Vec<i64>,
}

impl Aggregator<i64, usize> for OwnedOnlyAgg {
    fn init() -> Self {
        Self::default()
    }

    fn update(&mut self, value: i64) {
        self.values.push(value);
    }

    fn merge(&mut self, other: Self) {
        self.values.extend(other.values);
    }

    fn finalize(&self) -> usize {
        panic!("materialization should use finalize_owned for OwnedOnlyAgg")
    }

    fn finalize_owned(self) -> usize {
        self.values.len()
    }
}

pub(super) fn make_sensitive_single_key_float_data() -> (Vec<i64>, Vec<f64>) {
    let n = 260_000usize;
    let mut keys = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);
    for i in 0..n {
        keys.push((i % 257) as i64);
        let mut v = match i % 4 {
            0 => 1e16,
            1 => 1.0,
            2 => -1e16,
            _ => 0.25,
        };
        if i % 997 == 0 {
            v = f64::NAN;
        }
        if i % 991 == 0 {
            v = -0.0;
        }
        values.push(v);
    }
    (keys, values)
}

pub(super) fn make_partitioned_single_key_float_data() -> (Vec<i64>, Vec<f64>) {
    let n_groups = 10_000usize;
    let mut keys = Vec::with_capacity(n_groups * 2);
    let mut values = Vec::with_capacity(n_groups * 2);

    for i in 0..n_groups {
        let key = ((i * 8_191) % n_groups) as i64;
        let base = i as f64;
        keys.push(key);
        keys.push(key);
        values.push(base);
        values.push(base + 0.5);
    }

    (keys, values)
}

fn fingerprint_by_key(result: &GroupByResultF64) -> Vec<(i64, u64)> {
    let mut out: Vec<(i64, u64)> = result
        .keys
        .iter()
        .zip(result.values.iter())
        .map(|(&k, &v)| (k, v.to_bits()))
        .collect();
    out.sort_unstable_by_key(|(k, _)| *k);
    out
}

pub(super) fn assert_float_kernel_bitwise_deterministic(
    kernel: fn(&[i64], &[f64]) -> PyResult<GroupByResultF64>,
    keys: &[i64],
    values: &[f64],
) {
    let baseline = {
        let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let result = pool.install(|| kernel(keys, values).unwrap());
        fingerprint_by_key(&result)
    };

    for &threads in &[2usize, 4, 8, 16] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let result = pool.install(|| kernel(keys, values).unwrap());
        assert_eq!(
            fingerprint_by_key(&result),
            baseline,
            "bitwise mismatch at thread count {threads}"
        );
    }
}

pub(super) fn row_order_prod_for_key(keys: &[i64], values: &[f64], target_key: i64) -> f64 {
    let mut prod = 1.0;
    for (&key, &value) in keys.iter().zip(values.iter()) {
        if key == target_key && !value.is_nan() {
            prod *= value;
        }
    }
    prod
}
