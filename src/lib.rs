//! # pandas-booster
//!
//! High-performance numerical acceleration library for Pandas using Rust.
//!
//! This crate provides a Python module (`_rust`) that exposes parallel groupby operations
//! for Pandas DataFrames. It leverages:
//!
//! - **Rayon**: Work-stealing parallelism for multi-core CPU utilization
//! - **AHash**: Fast, DOS-resistant hashing for groupby key aggregation
//! - **PyO3**: Seamless Python/Rust interop with zero-copy NumPy array access
//!
//! ## Architecture
//!
//! The module is structured as follows:
//! - [`aggregation`]: Defines the [`Aggregator`](aggregation::Aggregator) trait and implementations
//!   for sum, mean, min, max operations on both `f64` and `i64` types.
//! - [`groupby`]: Implements parallel map-reduce groupby using Rayon's `par_chunks`.
//! - [`zero_copy`]: Utilities for safely borrowing NumPy arrays as Rust slices.
//!
//! ## Performance Notes
//!
//! - **GIL Release**: All heavy computations use `py.allow_threads()` to release the Python
//!   Global Interpreter Lock, enabling true parallelism.
//! - **Fallback Threshold**: Datasets smaller than [`FALLBACK_THRESHOLD`] (100,000 rows) should
//!   use native Pandas instead, as the overhead of Rust dispatch outweighs benefits.
//! - **Zero-Copy**: NumPy arrays are accessed directly without copying when C-contiguous.

#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;

pub mod aggregation;
pub mod groupby;
pub mod zero_copy;

use groupby::GroupByResultF64;
use numpy::PyReadonlyArray1;

/// Minimum dataset size for Rust acceleration to be beneficial.
/// Below this threshold, Python/Pandas overhead dominates and native Pandas is faster.
const FALLBACK_THRESHOLD: usize = 100_000;

/// Validates that input arrays have matching lengths and meet the minimum size threshold.
///
/// # Errors
/// - `PyValueError` if `keys_len != values_len`
/// - `PyValueError` if `keys_len < FALLBACK_THRESHOLD`
fn validate_inputs(keys_len: usize, values_len: usize) -> PyResult<()> {
    if keys_len != values_len {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "keys and values must have the same length, got {} and {}",
            keys_len, values_len
        )));
    }
    if keys_len < FALLBACK_THRESHOLD {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dataset too small for acceleration (got {}, need at least {}), use pandas directly",
            keys_len, FALLBACK_THRESHOLD
        )));
    }
    Ok(())
}

/// Computes parallel groupby sum for f64 values.
///
/// Releases the GIL during computation for true parallelism.
#[pyfunction]
fn groupby_sum_f64(
    py: Python<'_>,
    keys: PyReadonlyArray1<'_, i64>,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<GroupByResultF64> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    py.allow_threads(|| groupby::parallel_groupby_sum_f64(keys_slice, values_slice))
}

/// Computes parallel groupby mean for f64 values. Returns NaN for all-NaN groups.
#[pyfunction]
fn groupby_mean_f64(
    py: Python<'_>,
    keys: PyReadonlyArray1<'_, i64>,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<GroupByResultF64> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    py.allow_threads(|| groupby::parallel_groupby_mean_f64(keys_slice, values_slice))
}

/// Computes parallel groupby min for f64 values. Returns NaN for all-NaN groups.
#[pyfunction]
fn groupby_min_f64(
    py: Python<'_>,
    keys: PyReadonlyArray1<'_, i64>,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<GroupByResultF64> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    py.allow_threads(|| groupby::parallel_groupby_min_f64(keys_slice, values_slice))
}

/// Computes parallel groupby max for f64 values. Returns NaN for all-NaN groups.
#[pyfunction]
fn groupby_max_f64(
    py: Python<'_>,
    keys: PyReadonlyArray1<'_, i64>,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<GroupByResultF64> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    py.allow_threads(|| groupby::parallel_groupby_max_f64(keys_slice, values_slice))
}

/// Computes parallel groupby sum for i64 values. Returns f64 to match Pandas overflow behavior.
#[pyfunction]
fn groupby_sum_i64(
    py: Python<'_>,
    keys: PyReadonlyArray1<'_, i64>,
    values: PyReadonlyArray1<'_, i64>,
) -> PyResult<GroupByResultF64> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    py.allow_threads(|| groupby::parallel_groupby_sum_i64(keys_slice, values_slice))
}

/// Computes parallel groupby mean for i64 values.
#[pyfunction]
fn groupby_mean_i64(
    py: Python<'_>,
    keys: PyReadonlyArray1<'_, i64>,
    values: PyReadonlyArray1<'_, i64>,
) -> PyResult<GroupByResultF64> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    py.allow_threads(|| groupby::parallel_groupby_mean_i64(keys_slice, values_slice))
}

/// Computes parallel groupby min for i64 values.
#[pyfunction]
fn groupby_min_i64(
    py: Python<'_>,
    keys: PyReadonlyArray1<'_, i64>,
    values: PyReadonlyArray1<'_, i64>,
) -> PyResult<GroupByResultF64> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    py.allow_threads(|| groupby::parallel_groupby_min_i64(keys_slice, values_slice))
}

/// Computes parallel groupby max for i64 values.
#[pyfunction]
fn groupby_max_i64(
    py: Python<'_>,
    keys: PyReadonlyArray1<'_, i64>,
    values: PyReadonlyArray1<'_, i64>,
) -> PyResult<GroupByResultF64> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    py.allow_threads(|| groupby::parallel_groupby_max_i64(keys_slice, values_slice))
}

/// Returns the minimum dataset size threshold for Rust acceleration.
#[pyfunction]
fn get_fallback_threshold() -> usize {
    FALLBACK_THRESHOLD
}

/// Returns the number of threads used by the Rayon thread pool.
#[pyfunction]
fn get_thread_count() -> usize {
    rayon::current_num_threads()
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(groupby_sum_f64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_mean_f64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_min_f64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_max_f64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_sum_i64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_mean_i64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_min_i64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_max_i64, m)?)?;
    m.add_function(wrap_pyfunction!(get_fallback_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(get_thread_count, m)?)?;
    Ok(())
}
