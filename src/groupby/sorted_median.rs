use pyo3::prelude::*;
use rayon::prelude::*;

use crate::aggregation::median::{median_f64_from_mut_slice, median_i64_from_mut_slice};

use super::result::GroupByResultF64;
use super::sorted_median_dense::{groupby_median_dense_direct_with_stats, DenseKeyRange};
use super::sorted_median_sparse::groupby_median_sparse_direct_with_stats;

#[cfg(test)]
pub(super) use super::sorted_median_hooks::{
    reset_sorted_median_direct_call_count, reset_sorted_median_parallel_slice_median_count,
    sorted_median_direct_call_count, sorted_median_parallel_slice_median_count,
};

use super::sorted_median_hooks::{
    record_parallel_slice_median_call, record_sorted_median_direct_call,
};

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct SortedMedianDirectStats {
    pub(super) partial_group_total: usize,
    pub(super) final_group_count: usize,
    #[cfg(test)]
    pub(super) accepted_value_count: usize,
    #[cfg(test)]
    pub(super) value_buffer_len: usize,
    #[cfg(test)]
    pub(super) scatter_write_count: usize,
    pub(super) unique_build_s: f64,
    pub(super) key_sort_s: f64,
    pub(super) count_s: f64,
    pub(super) buffer_setup_s: f64,
    pub(super) scatter_s: f64,
    pub(super) median_select_s: f64,
}

#[derive(Debug)]
pub(super) struct ProfiledSortedMedianDirect {
    pub(super) result: GroupByResultF64,
    pub(super) stats: SortedMedianDirectStats,
}

pub(super) trait SortedMedianValue: Copy + Send {
    fn is_accepted(self) -> bool;
    fn median_from_group(values: &mut [Self]) -> f64;
}

impl SortedMedianValue for f64 {
    fn is_accepted(self) -> bool {
        !self.is_nan()
    }

    fn median_from_group(values: &mut [Self]) -> f64 {
        median_f64_from_mut_slice(values)
    }
}

impl SortedMedianValue for i64 {
    fn is_accepted(self) -> bool {
        true
    }

    fn median_from_group(values: &mut [Self]) -> f64 {
        median_i64_from_mut_slice(values)
    }
}

pub(super) fn groupby_median_f64_sorted_direct(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    record_sorted_median_direct_call();
    Ok(groupby_median_sorted_direct_with_stats(keys, values)?.result)
}

pub(super) fn groupby_median_i64_sorted_direct(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultF64> {
    record_sorted_median_direct_call();
    Ok(groupby_median_sorted_direct_with_stats(keys, values)?.result)
}

pub(super) fn groupby_median_f64_sorted_direct_with_stats(
    keys: &[i64],
    values: &[f64],
) -> PyResult<ProfiledSortedMedianDirect> {
    groupby_median_sorted_direct_with_stats(keys, values)
}

fn groupby_median_sorted_direct_with_stats<T>(
    keys: &[i64],
    values: &[T],
) -> PyResult<ProfiledSortedMedianDirect>
where
    T: SortedMedianValue,
{
    if keys.len() != values.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "keys and values must have same length",
        ));
    }

    if keys.is_empty() {
        return Ok(ProfiledSortedMedianDirect {
            result: GroupByResultF64 {
                keys: Vec::new(),
                values: Vec::new(),
            },
            stats: SortedMedianDirectStats::default(),
        });
    }

    if let Some(dense_range) = DenseKeyRange::from_keys(keys) {
        return groupby_median_dense_direct_with_stats(keys, values, dense_range);
    }

    groupby_median_sparse_direct_with_stats(keys, values)
}

#[cfg(test)]
pub(super) fn increment_scatter_write_count(scatter_write_count: &mut usize) -> PyResult<()> {
    *scatter_write_count = scatter_write_count
        .checked_add(1)
        .ok_or_else(count_overflow)?;
    Ok(())
}

pub(super) fn checked_total(counts: &[usize]) -> PyResult<usize> {
    let mut total = 0usize;
    for count in counts.iter().copied() {
        total = total.checked_add(count).ok_or_else(count_overflow)?;
    }
    Ok(total)
}

pub(super) fn materialize_group_medians<T>(grouped_values: &mut [Vec<T>]) -> Vec<f64>
where
    T: SortedMedianValue,
{
    grouped_values
        .par_iter_mut()
        .map(|group_values| {
            record_parallel_slice_median_call();
            T::median_from_group(group_values)
        })
        .collect()
}

pub(super) fn materialize_group_medians_from_pairs<T>(groups: &mut [(i64, Vec<T>)]) -> Vec<f64>
where
    T: SortedMedianValue,
{
    groups
        .par_iter_mut()
        .map(|(_, group_values)| {
            record_parallel_slice_median_call();
            T::median_from_group(group_values)
        })
        .collect()
}
pub(super) fn remap_incomplete_error() -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err("median sorted key remap is incomplete")
}

pub(super) fn count_overflow() -> PyErr {
    pyo3::exceptions::PyOverflowError::new_err("median value count overflow")
}
