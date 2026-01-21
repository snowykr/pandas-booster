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
pub mod groupby_multi;
pub mod radix_groupby;
pub mod zero_copy;

use groupby::GroupByResultF64;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, ToPyArray};

type MultiGroupByReturn<'py> = (Bound<'py, PyArray2<i64>>, Bound<'py, PyArray1<f64>>);

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

/// Computes parallel groupby count for f64 values. Counts non-NaN values.
#[pyfunction]
fn groupby_count_f64(
    py: Python<'_>,
    keys: PyReadonlyArray1<'_, i64>,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<GroupByResultF64> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    py.allow_threads(|| groupby::parallel_groupby_count_f64(keys_slice, values_slice))
}

/// Computes parallel groupby count for i64 values.
#[pyfunction]
fn groupby_count_i64(
    py: Python<'_>,
    keys: PyReadonlyArray1<'_, i64>,
    values: PyReadonlyArray1<'_, i64>,
) -> PyResult<GroupByResultF64> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    py.allow_threads(|| groupby::parallel_groupby_count_i64(keys_slice, values_slice))
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

// =============================================================================
// Multi-column groupby functions
// =============================================================================

/// Validates multi-key inputs: all columns same length, meets threshold.
fn validate_multi_inputs(key_lengths: &[usize], values_len: usize) -> PyResult<()> {
    if key_lengths.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "At least one key column is required",
        ));
    }
    if key_lengths.len() > 10 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Too many key columns (got {}, max 10)",
            key_lengths.len()
        )));
    }
    for (i, &len) in key_lengths.iter().enumerate() {
        if len != values_len {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Key column {} length {} != values length {}",
                i, len, values_len
            )));
        }
    }
    if values_len < FALLBACK_THRESHOLD {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dataset too small for acceleration (got {}, need at least {}), use pandas directly",
            values_len, FALLBACK_THRESHOLD
        )));
    }
    Ok(())
}

/// Multi-column groupby sum for f64 values.
/// Returns (keys_2d, values_1d) where keys_2d is shape (n_groups, n_keys).
#[pyfunction]
fn groupby_multi_sum_f64<'py>(
    py: Python<'py>,
    key_cols: Vec<PyReadonlyArray1<'py, i64>>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<MultiGroupByReturn<'py>> {
    let values_slice = zero_copy::get_slice_f64(&values)?;
    let key_slices: Vec<&[i64]> = key_cols
        .iter()
        .map(|col| zero_copy::get_slice_i64(col))
        .collect::<PyResult<Vec<_>>>()?;

    let key_lengths: Vec<usize> = key_slices.iter().map(|s| s.len()).collect();
    validate_multi_inputs(&key_lengths, values_slice.len())?;

    let result =
        py.allow_threads(|| groupby_multi::multi_groupby_sum_f64(&key_slices, values_slice))?;

    convert_multi_result(py, result)
}

/// Multi-column groupby mean for f64 values.
#[pyfunction]
fn groupby_multi_mean_f64<'py>(
    py: Python<'py>,
    key_cols: Vec<PyReadonlyArray1<'py, i64>>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<MultiGroupByReturn<'py>> {
    let values_slice = zero_copy::get_slice_f64(&values)?;
    let key_slices: Vec<&[i64]> = key_cols
        .iter()
        .map(|col| zero_copy::get_slice_i64(col))
        .collect::<PyResult<Vec<_>>>()?;

    let key_lengths: Vec<usize> = key_slices.iter().map(|s| s.len()).collect();
    validate_multi_inputs(&key_lengths, values_slice.len())?;

    let result =
        py.allow_threads(|| groupby_multi::multi_groupby_mean_f64(&key_slices, values_slice))?;

    convert_multi_result(py, result)
}

/// Multi-column groupby min for f64 values.
#[pyfunction]
fn groupby_multi_min_f64<'py>(
    py: Python<'py>,
    key_cols: Vec<PyReadonlyArray1<'py, i64>>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<MultiGroupByReturn<'py>> {
    let values_slice = zero_copy::get_slice_f64(&values)?;
    let key_slices: Vec<&[i64]> = key_cols
        .iter()
        .map(|col| zero_copy::get_slice_i64(col))
        .collect::<PyResult<Vec<_>>>()?;

    let key_lengths: Vec<usize> = key_slices.iter().map(|s| s.len()).collect();
    validate_multi_inputs(&key_lengths, values_slice.len())?;

    let result =
        py.allow_threads(|| groupby_multi::multi_groupby_min_f64(&key_slices, values_slice))?;

    convert_multi_result(py, result)
}

/// Multi-column groupby max for f64 values.
#[pyfunction]
fn groupby_multi_max_f64<'py>(
    py: Python<'py>,
    key_cols: Vec<PyReadonlyArray1<'py, i64>>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<MultiGroupByReturn<'py>> {
    let values_slice = zero_copy::get_slice_f64(&values)?;
    let key_slices: Vec<&[i64]> = key_cols
        .iter()
        .map(|col| zero_copy::get_slice_i64(col))
        .collect::<PyResult<Vec<_>>>()?;

    let key_lengths: Vec<usize> = key_slices.iter().map(|s| s.len()).collect();
    validate_multi_inputs(&key_lengths, values_slice.len())?;

    let result =
        py.allow_threads(|| groupby_multi::multi_groupby_max_f64(&key_slices, values_slice))?;

    convert_multi_result(py, result)
}

/// Multi-column groupby sum for i64 values.
#[pyfunction]
fn groupby_multi_sum_i64<'py>(
    py: Python<'py>,
    key_cols: Vec<PyReadonlyArray1<'py, i64>>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<MultiGroupByReturn<'py>> {
    let values_slice = zero_copy::get_slice_i64(&values)?;
    let key_slices: Vec<&[i64]> = key_cols
        .iter()
        .map(|col| zero_copy::get_slice_i64(col))
        .collect::<PyResult<Vec<_>>>()?;

    let key_lengths: Vec<usize> = key_slices.iter().map(|s| s.len()).collect();
    validate_multi_inputs(&key_lengths, values_slice.len())?;

    let result =
        py.allow_threads(|| groupby_multi::multi_groupby_sum_i64(&key_slices, values_slice))?;

    convert_multi_result(py, result)
}

/// Multi-column groupby mean for i64 values.
#[pyfunction]
fn groupby_multi_mean_i64<'py>(
    py: Python<'py>,
    key_cols: Vec<PyReadonlyArray1<'py, i64>>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<MultiGroupByReturn<'py>> {
    let values_slice = zero_copy::get_slice_i64(&values)?;
    let key_slices: Vec<&[i64]> = key_cols
        .iter()
        .map(|col| zero_copy::get_slice_i64(col))
        .collect::<PyResult<Vec<_>>>()?;

    let key_lengths: Vec<usize> = key_slices.iter().map(|s| s.len()).collect();
    validate_multi_inputs(&key_lengths, values_slice.len())?;

    let result =
        py.allow_threads(|| groupby_multi::multi_groupby_mean_i64(&key_slices, values_slice))?;

    convert_multi_result(py, result)
}

/// Multi-column groupby min for i64 values.
#[pyfunction]
fn groupby_multi_min_i64<'py>(
    py: Python<'py>,
    key_cols: Vec<PyReadonlyArray1<'py, i64>>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<MultiGroupByReturn<'py>> {
    let values_slice = zero_copy::get_slice_i64(&values)?;
    let key_slices: Vec<&[i64]> = key_cols
        .iter()
        .map(|col| zero_copy::get_slice_i64(col))
        .collect::<PyResult<Vec<_>>>()?;

    let key_lengths: Vec<usize> = key_slices.iter().map(|s| s.len()).collect();
    validate_multi_inputs(&key_lengths, values_slice.len())?;

    let result =
        py.allow_threads(|| groupby_multi::multi_groupby_min_i64(&key_slices, values_slice))?;

    convert_multi_result(py, result)
}

/// Multi-column groupby max for i64 values.
#[pyfunction]
fn groupby_multi_max_i64<'py>(
    py: Python<'py>,
    key_cols: Vec<PyReadonlyArray1<'py, i64>>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<MultiGroupByReturn<'py>> {
    let values_slice = zero_copy::get_slice_i64(&values)?;
    let key_slices: Vec<&[i64]> = key_cols
        .iter()
        .map(|col| zero_copy::get_slice_i64(col))
        .collect::<PyResult<Vec<_>>>()?;

    let key_lengths: Vec<usize> = key_slices.iter().map(|s| s.len()).collect();
    validate_multi_inputs(&key_lengths, values_slice.len())?;

    let result =
        py.allow_threads(|| groupby_multi::multi_groupby_max_i64(&key_slices, values_slice))?;

    convert_multi_result(py, result)
}

/// Multi-column groupby count for f64 values.
#[pyfunction]
fn groupby_multi_count_f64<'py>(
    py: Python<'py>,
    key_cols: Vec<PyReadonlyArray1<'py, i64>>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<MultiGroupByReturn<'py>> {
    let values_slice = zero_copy::get_slice_f64(&values)?;
    let key_slices: Vec<&[i64]> = key_cols
        .iter()
        .map(|col| zero_copy::get_slice_i64(col))
        .collect::<PyResult<Vec<_>>>()?;

    let key_lengths: Vec<usize> = key_slices.iter().map(|s| s.len()).collect();
    validate_multi_inputs(&key_lengths, values_slice.len())?;

    let result =
        py.allow_threads(|| groupby_multi::multi_groupby_count_f64(&key_slices, values_slice))?;

    convert_multi_result(py, result)
}

/// Multi-column groupby count for i64 values.
#[pyfunction]
fn groupby_multi_count_i64<'py>(
    py: Python<'py>,
    key_cols: Vec<PyReadonlyArray1<'py, i64>>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<MultiGroupByReturn<'py>> {
    let values_slice = zero_copy::get_slice_i64(&values)?;
    let key_slices: Vec<&[i64]> = key_cols
        .iter()
        .map(|col| zero_copy::get_slice_i64(col))
        .collect::<PyResult<Vec<_>>>()?;

    let key_lengths: Vec<usize> = key_slices.iter().map(|s| s.len()).collect();
    validate_multi_inputs(&key_lengths, values_slice.len())?;

    let result =
        py.allow_threads(|| groupby_multi::multi_groupby_count_i64(&key_slices, values_slice))?;

    convert_multi_result(py, result)
}

/// Convert GroupByMultiResult to (PyArray2<i64>, PyArray1<f64>).
fn convert_multi_result<'py>(
    py: Python<'py>,
    result: groupby_multi::GroupByMultiResult,
) -> PyResult<MultiGroupByReturn<'py>> {
    let n_groups = result.values.len();
    let n_keys = result.n_keys;

    // Create 2D array for keys: shape (n_groups, n_keys)
    let keys_2d = if n_groups > 0 {
        // Safety invariant: keys_flat must have exactly n_groups * n_keys elements.
        // This must be enforced in release as well, since we use unsafe raw copies below.
        let expected_len = n_groups.checked_mul(n_keys).ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("n_groups * n_keys overflow")
        })?;
        if result.keys_flat.len() != expected_len {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "keys_flat length {} does not match expected {} (n_groups={} * n_keys={})",
                result.keys_flat.len(),
                expected_len,
                n_groups,
                n_keys
            )));
        }

        // Reshape flat keys to 2D
        // SAFETY: We verified keys_flat.len() == n_groups * n_keys above.
        // The PyArray2 is created with shape [n_groups, n_keys] which has the same total
        // element count, so the copy is within bounds.
        unsafe {
            let arr = PyArray2::new_bound(py, [n_groups, n_keys], false);
            let ptr = arr.as_raw_array_mut().as_mut_ptr();
            std::ptr::copy_nonoverlapping(result.keys_flat.as_ptr(), ptr, result.keys_flat.len());
            arr
        }
    } else {
        // Empty result
        // SAFETY: the array is zero-length (no elements to initialize).
        unsafe { PyArray2::new_bound(py, [0, n_keys], false) }
    };

    let values_1d = result.values.to_pyarray_bound(py);

    Ok((keys_2d, values_1d))
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Single-key groupby
    m.add_function(wrap_pyfunction!(groupby_sum_f64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_mean_f64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_min_f64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_max_f64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_sum_i64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_mean_i64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_min_i64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_max_i64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_count_f64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_count_i64, m)?)?;
    // Multi-key groupby
    m.add_function(wrap_pyfunction!(groupby_multi_sum_f64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_mean_f64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_min_f64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_max_f64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_sum_i64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_mean_i64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_min_i64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_max_i64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_count_f64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_count_i64, m)?)?;
    // Utilities
    m.add_function(wrap_pyfunction!(get_fallback_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(get_thread_count, m)?)?;
    Ok(())
}
