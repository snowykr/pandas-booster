use super::GroupByMultiResult;
use crate::aggregation::Aggregator;

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
        panic!("radix materialization should use finalize_owned for OwnedOnlyAgg")
    }

    fn finalize_owned(self) -> usize {
        self.values.len()
    }
}

#[inline]
pub(super) fn src_group_for_out<V>(res: &GroupByMultiResult<V>, out_g: usize) -> usize {
    match &res.perm {
        Some(p) => p[out_g],
        None => out_g,
    }
}

#[inline]
pub(super) fn key_at_out<V>(res: &GroupByMultiResult<V>, out_g: usize, col: usize) -> i64 {
    let src_g = src_group_for_out(res, out_g);
    res.keys_flat[src_g * res.n_keys + col]
}

#[inline]
pub(super) fn value_at_out<V: Copy>(res: &GroupByMultiResult<V>, out_g: usize) -> V {
    let src_g = src_group_for_out(res, out_g);
    res.values[src_g]
}
