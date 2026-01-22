//! Multi-column groupby using Radix Partitioning.
//!
//! This module provides the public API for multi-key groupby operations, delegating
//! to the radix partitioning engine for high-performance aggregation.

use pyo3::prelude::*;

use crate::radix_groupby::{
    self, radix_groupby_count_f64, radix_groupby_count_f64_firstseen_u32,
    radix_groupby_count_f64_firstseen_u64, radix_groupby_count_i64,
    radix_groupby_count_i64_firstseen_u32, radix_groupby_count_i64_firstseen_u64,
    radix_groupby_max_f64, radix_groupby_max_f64_firstseen_u32,
    radix_groupby_max_f64_firstseen_u64, radix_groupby_max_i64,
    radix_groupby_max_i64_firstseen_u32, radix_groupby_max_i64_firstseen_u64,
    radix_groupby_mean_f64, radix_groupby_mean_f64_firstseen_u32,
    radix_groupby_mean_f64_firstseen_u64, radix_groupby_mean_i64,
    radix_groupby_mean_i64_firstseen_u32, radix_groupby_mean_i64_firstseen_u64,
    radix_groupby_min_f64, radix_groupby_min_f64_firstseen_u32,
    radix_groupby_min_f64_firstseen_u64, radix_groupby_min_i64,
    radix_groupby_min_i64_firstseen_u32, radix_groupby_min_i64_firstseen_u64,
    radix_groupby_sum_f64, radix_groupby_sum_f64_firstseen_u32,
    radix_groupby_sum_f64_firstseen_u64, radix_groupby_sum_i64,
    radix_groupby_sum_i64_firstseen_u32, radix_groupby_sum_i64_firstseen_u64,
};

pub use radix_groupby::GroupByMultiResult;

pub fn multi_groupby_sum_f64(
    key_slices: &[&[i64]],
    values: &[f64],
) -> PyResult<GroupByMultiResult> {
    radix_groupby_sum_f64(key_slices, values).map_err(pyo3::exceptions::PyValueError::new_err)
}

pub fn multi_groupby_mean_f64(
    key_slices: &[&[i64]],
    values: &[f64],
) -> PyResult<GroupByMultiResult> {
    radix_groupby_mean_f64(key_slices, values).map_err(pyo3::exceptions::PyValueError::new_err)
}

pub fn multi_groupby_min_f64(
    key_slices: &[&[i64]],
    values: &[f64],
) -> PyResult<GroupByMultiResult> {
    radix_groupby_min_f64(key_slices, values).map_err(pyo3::exceptions::PyValueError::new_err)
}

pub fn multi_groupby_max_f64(
    key_slices: &[&[i64]],
    values: &[f64],
) -> PyResult<GroupByMultiResult> {
    radix_groupby_max_f64(key_slices, values).map_err(pyo3::exceptions::PyValueError::new_err)
}

pub fn multi_groupby_sum_i64(
    key_slices: &[&[i64]],
    values: &[i64],
) -> PyResult<GroupByMultiResult> {
    radix_groupby_sum_i64(key_slices, values).map_err(pyo3::exceptions::PyValueError::new_err)
}

pub fn multi_groupby_mean_i64(
    key_slices: &[&[i64]],
    values: &[i64],
) -> PyResult<GroupByMultiResult> {
    radix_groupby_mean_i64(key_slices, values).map_err(pyo3::exceptions::PyValueError::new_err)
}

pub fn multi_groupby_min_i64(
    key_slices: &[&[i64]],
    values: &[i64],
) -> PyResult<GroupByMultiResult> {
    radix_groupby_min_i64(key_slices, values).map_err(pyo3::exceptions::PyValueError::new_err)
}

pub fn multi_groupby_max_i64(
    key_slices: &[&[i64]],
    values: &[i64],
) -> PyResult<GroupByMultiResult> {
    radix_groupby_max_i64(key_slices, values).map_err(pyo3::exceptions::PyValueError::new_err)
}

pub fn multi_groupby_count_f64(
    key_slices: &[&[i64]],
    values: &[f64],
) -> PyResult<GroupByMultiResult> {
    radix_groupby_count_f64(key_slices, values).map_err(pyo3::exceptions::PyValueError::new_err)
}

pub fn multi_groupby_count_i64(
    key_slices: &[&[i64]],
    values: &[i64],
) -> PyResult<GroupByMultiResult> {
    radix_groupby_count_i64(key_slices, values).map_err(pyo3::exceptions::PyValueError::new_err)
}

// =============================================================================
// First-seen ordered variants (sort=False semantics)
// =============================================================================

pub fn multi_groupby_sum_f64_firstseen_u32(
    key_slices: &[&[i64]],
    values: &[f64],
) -> PyResult<GroupByMultiResult> {
    radix_groupby_sum_f64_firstseen_u32(key_slices, values)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

pub fn multi_groupby_mean_f64_firstseen_u32(
    key_slices: &[&[i64]],
    values: &[f64],
) -> PyResult<GroupByMultiResult> {
    radix_groupby_mean_f64_firstseen_u32(key_slices, values)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

pub fn multi_groupby_min_f64_firstseen_u32(
    key_slices: &[&[i64]],
    values: &[f64],
) -> PyResult<GroupByMultiResult> {
    radix_groupby_min_f64_firstseen_u32(key_slices, values)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

pub fn multi_groupby_max_f64_firstseen_u32(
    key_slices: &[&[i64]],
    values: &[f64],
) -> PyResult<GroupByMultiResult> {
    radix_groupby_max_f64_firstseen_u32(key_slices, values)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

pub fn multi_groupby_sum_i64_firstseen_u32(
    key_slices: &[&[i64]],
    values: &[i64],
) -> PyResult<GroupByMultiResult> {
    radix_groupby_sum_i64_firstseen_u32(key_slices, values)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

pub fn multi_groupby_mean_i64_firstseen_u32(
    key_slices: &[&[i64]],
    values: &[i64],
) -> PyResult<GroupByMultiResult> {
    radix_groupby_mean_i64_firstseen_u32(key_slices, values)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

pub fn multi_groupby_min_i64_firstseen_u32(
    key_slices: &[&[i64]],
    values: &[i64],
) -> PyResult<GroupByMultiResult> {
    radix_groupby_min_i64_firstseen_u32(key_slices, values)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

pub fn multi_groupby_max_i64_firstseen_u32(
    key_slices: &[&[i64]],
    values: &[i64],
) -> PyResult<GroupByMultiResult> {
    radix_groupby_max_i64_firstseen_u32(key_slices, values)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

pub fn multi_groupby_count_f64_firstseen_u32(
    key_slices: &[&[i64]],
    values: &[f64],
) -> PyResult<GroupByMultiResult> {
    radix_groupby_count_f64_firstseen_u32(key_slices, values)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

pub fn multi_groupby_count_i64_firstseen_u32(
    key_slices: &[&[i64]],
    values: &[i64],
) -> PyResult<GroupByMultiResult> {
    radix_groupby_count_i64_firstseen_u32(key_slices, values)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

pub fn multi_groupby_sum_f64_firstseen_u64(
    key_slices: &[&[i64]],
    values: &[f64],
) -> PyResult<GroupByMultiResult> {
    radix_groupby_sum_f64_firstseen_u64(key_slices, values)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

pub fn multi_groupby_mean_f64_firstseen_u64(
    key_slices: &[&[i64]],
    values: &[f64],
) -> PyResult<GroupByMultiResult> {
    radix_groupby_mean_f64_firstseen_u64(key_slices, values)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

pub fn multi_groupby_min_f64_firstseen_u64(
    key_slices: &[&[i64]],
    values: &[f64],
) -> PyResult<GroupByMultiResult> {
    radix_groupby_min_f64_firstseen_u64(key_slices, values)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

pub fn multi_groupby_max_f64_firstseen_u64(
    key_slices: &[&[i64]],
    values: &[f64],
) -> PyResult<GroupByMultiResult> {
    radix_groupby_max_f64_firstseen_u64(key_slices, values)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

pub fn multi_groupby_sum_i64_firstseen_u64(
    key_slices: &[&[i64]],
    values: &[i64],
) -> PyResult<GroupByMultiResult> {
    radix_groupby_sum_i64_firstseen_u64(key_slices, values)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

pub fn multi_groupby_mean_i64_firstseen_u64(
    key_slices: &[&[i64]],
    values: &[i64],
) -> PyResult<GroupByMultiResult> {
    radix_groupby_mean_i64_firstseen_u64(key_slices, values)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

pub fn multi_groupby_min_i64_firstseen_u64(
    key_slices: &[&[i64]],
    values: &[i64],
) -> PyResult<GroupByMultiResult> {
    radix_groupby_min_i64_firstseen_u64(key_slices, values)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

pub fn multi_groupby_max_i64_firstseen_u64(
    key_slices: &[&[i64]],
    values: &[i64],
) -> PyResult<GroupByMultiResult> {
    radix_groupby_max_i64_firstseen_u64(key_slices, values)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

pub fn multi_groupby_count_f64_firstseen_u64(
    key_slices: &[&[i64]],
    values: &[f64],
) -> PyResult<GroupByMultiResult> {
    radix_groupby_count_f64_firstseen_u64(key_slices, values)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

pub fn multi_groupby_count_i64_firstseen_u64(
    key_slices: &[&[i64]],
    values: &[i64],
) -> PyResult<GroupByMultiResult> {
    radix_groupby_count_i64_firstseen_u64(key_slices, values)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}
