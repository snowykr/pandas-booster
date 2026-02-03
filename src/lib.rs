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
pub mod radix_sort;
pub mod zero_copy;

use groupby::{GroupByResultF64, GroupByResultI64};
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
use rayon::prelude::*;

type MultiGroupByKeysReturn<'py> = Vec<Bound<'py, PyArray1<i64>>>;

type MultiGroupByReturnF64<'py> = (MultiGroupByKeysReturn<'py>, Bound<'py, PyArray1<f64>>);
type MultiGroupByReturnI64<'py> = (MultiGroupByKeysReturn<'py>, Bound<'py, PyArray1<i64>>);

// Backwards-friendly aliases: most kernels still return f64 values.
type MultiGroupByReturn<'py> = MultiGroupByReturnF64<'py>;

type SingleGroupByReturnF64<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>);
type SingleGroupByReturnI64<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>);

// Backwards-friendly alias: most kernels still return f64 values.
type SingleGroupByReturn<'py> = SingleGroupByReturnF64<'py>;

fn convert_single_result_f64<'py>(
    py: Python<'py>,
    result: GroupByResultF64,
) -> PyResult<SingleGroupByReturnF64<'py>> {
    let GroupByResultF64 { keys, values } = result;
    let keys_1d = keys.into_pyarray_bound(py);
    let values_1d = values.into_pyarray_bound(py);
    Ok((keys_1d, values_1d))
}

fn convert_single_result<'py>(
    py: Python<'py>,
    result: GroupByResultF64,
) -> PyResult<SingleGroupByReturn<'py>> {
    convert_single_result_f64(py, result)
}

fn convert_single_result_i64<'py>(
    py: Python<'py>,
    result: GroupByResultI64,
) -> PyResult<SingleGroupByReturnI64<'py>> {
    let GroupByResultI64 { keys, values } = result;
    let keys_1d = keys.into_pyarray_bound(py);
    let values_1d = values.into_pyarray_bound(py);
    Ok((keys_1d, values_1d))
}

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
fn groupby_sum_f64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result =
        py.allow_threads(|| groupby::parallel_groupby_sum_f64(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

/// Computes parallel groupby mean for f64 values. Returns NaN for all-NaN groups.
#[pyfunction]
fn groupby_mean_f64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result =
        py.allow_threads(|| groupby::parallel_groupby_mean_f64(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

/// Computes parallel groupby min for f64 values. Returns NaN for all-NaN groups.
#[pyfunction]
fn groupby_min_f64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result =
        py.allow_threads(|| groupby::parallel_groupby_min_f64(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

/// Computes parallel groupby max for f64 values. Returns NaN for all-NaN groups.
#[pyfunction]
fn groupby_max_f64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result =
        py.allow_threads(|| groupby::parallel_groupby_max_f64(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

/// Computes parallel groupby sum for i64 values. Returns f64 to match Pandas overflow behavior.
#[pyfunction]
fn groupby_sum_i64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result =
        py.allow_threads(|| groupby::parallel_groupby_sum_i64(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

/// Computes parallel groupby mean for i64 values.
#[pyfunction]
fn groupby_mean_i64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result =
        py.allow_threads(|| groupby::parallel_groupby_mean_i64(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

/// Computes parallel groupby min for i64 values.
#[pyfunction]
fn groupby_min_i64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result =
        py.allow_threads(|| groupby::parallel_groupby_min_i64(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

/// Computes parallel groupby max for i64 values.
#[pyfunction]
fn groupby_max_i64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result =
        py.allow_threads(|| groupby::parallel_groupby_max_i64(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

/// Computes parallel groupby count for f64 values. Counts non-NaN values.
#[pyfunction]
fn groupby_count_f64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturnI64<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result =
        py.allow_threads(|| groupby::parallel_groupby_count_f64(keys_slice, values_slice))?;
    convert_single_result_i64(py, result)
}

/// Computes parallel groupby count for i64 values.
#[pyfunction]
fn groupby_count_i64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturnI64<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result =
        py.allow_threads(|| groupby::parallel_groupby_count_i64(keys_slice, values_slice))?;
    convert_single_result_i64(py, result)
}

// =============================================================================
// Single-key groupby (sorted, for sort=True semantics)
// =============================================================================

#[pyfunction]
fn groupby_sum_f64_sorted<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result =
        py.allow_threads(|| groupby::parallel_groupby_sum_f64_sorted(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

#[pyfunction]
fn groupby_mean_f64_sorted<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result =
        py.allow_threads(|| groupby::parallel_groupby_mean_f64_sorted(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

#[pyfunction]
fn groupby_min_f64_sorted<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result =
        py.allow_threads(|| groupby::parallel_groupby_min_f64_sorted(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

#[pyfunction]
fn groupby_max_f64_sorted<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result =
        py.allow_threads(|| groupby::parallel_groupby_max_f64_sorted(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

#[pyfunction]
fn groupby_count_f64_sorted<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturnI64<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result =
        py.allow_threads(|| groupby::parallel_groupby_count_f64_sorted(keys_slice, values_slice))?;
    convert_single_result_i64(py, result)
}

#[pyfunction]
fn groupby_sum_i64_sorted<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result =
        py.allow_threads(|| groupby::parallel_groupby_sum_i64_sorted(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

#[pyfunction]
fn groupby_mean_i64_sorted<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result =
        py.allow_threads(|| groupby::parallel_groupby_mean_i64_sorted(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

#[pyfunction]
fn groupby_min_i64_sorted<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result =
        py.allow_threads(|| groupby::parallel_groupby_min_i64_sorted(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

#[pyfunction]
fn groupby_max_i64_sorted<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result =
        py.allow_threads(|| groupby::parallel_groupby_max_i64_sorted(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

#[pyfunction]
fn groupby_count_i64_sorted<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturnI64<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result =
        py.allow_threads(|| groupby::parallel_groupby_count_i64_sorted(keys_slice, values_slice))?;
    convert_single_result_i64(py, result)
}

// =============================================================================
// First-seen ordered groupby (sort=False semantics)
// =============================================================================

#[pyfunction]
fn groupby_sum_f64_firstseen_u32<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result = py.allow_threads(|| {
        groupby::parallel_groupby_sum_f64_firstseen_u32(keys_slice, values_slice)
    })?;
    convert_single_result(py, result)
}

#[pyfunction]
fn groupby_sum_f64_firstseen_u64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result = py.allow_threads(|| {
        groupby::parallel_groupby_sum_f64_firstseen_u64(keys_slice, values_slice)
    })?;
    convert_single_result(py, result)
}

#[pyfunction]
fn groupby_mean_f64_firstseen_u32<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result = py.allow_threads(|| {
        groupby::parallel_groupby_mean_f64_firstseen_u32(keys_slice, values_slice)
    })?;
    convert_single_result(py, result)
}

#[pyfunction]
fn groupby_mean_f64_firstseen_u64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result = py.allow_threads(|| {
        groupby::parallel_groupby_mean_f64_firstseen_u64(keys_slice, values_slice)
    })?;
    convert_single_result(py, result)
}

#[pyfunction]
fn groupby_min_f64_firstseen_u32<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result = py.allow_threads(|| {
        groupby::parallel_groupby_min_f64_firstseen_u32(keys_slice, values_slice)
    })?;
    convert_single_result(py, result)
}

#[pyfunction]
fn groupby_min_f64_firstseen_u64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result = py.allow_threads(|| {
        groupby::parallel_groupby_min_f64_firstseen_u64(keys_slice, values_slice)
    })?;
    convert_single_result(py, result)
}

#[pyfunction]
fn groupby_max_f64_firstseen_u32<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result = py.allow_threads(|| {
        groupby::parallel_groupby_max_f64_firstseen_u32(keys_slice, values_slice)
    })?;
    convert_single_result(py, result)
}

#[pyfunction]
fn groupby_max_f64_firstseen_u64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result = py.allow_threads(|| {
        groupby::parallel_groupby_max_f64_firstseen_u64(keys_slice, values_slice)
    })?;
    convert_single_result(py, result)
}

#[pyfunction]
fn groupby_count_f64_firstseen_u32<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturnI64<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result = py.allow_threads(|| {
        groupby::parallel_groupby_count_f64_firstseen_u32(keys_slice, values_slice)
    })?;
    convert_single_result_i64(py, result)
}

#[pyfunction]
fn groupby_count_f64_firstseen_u64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturnI64<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result = py.allow_threads(|| {
        groupby::parallel_groupby_count_f64_firstseen_u64(keys_slice, values_slice)
    })?;
    convert_single_result_i64(py, result)
}

#[pyfunction]
fn groupby_sum_i64_firstseen_u32<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result = py.allow_threads(|| {
        groupby::parallel_groupby_sum_i64_firstseen_u32(keys_slice, values_slice)
    })?;
    convert_single_result(py, result)
}

#[pyfunction]
fn groupby_sum_i64_firstseen_u64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result = py.allow_threads(|| {
        groupby::parallel_groupby_sum_i64_firstseen_u64(keys_slice, values_slice)
    })?;
    convert_single_result(py, result)
}

#[pyfunction]
fn groupby_mean_i64_firstseen_u32<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result = py.allow_threads(|| {
        groupby::parallel_groupby_mean_i64_firstseen_u32(keys_slice, values_slice)
    })?;
    convert_single_result(py, result)
}

#[pyfunction]
fn groupby_mean_i64_firstseen_u64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result = py.allow_threads(|| {
        groupby::parallel_groupby_mean_i64_firstseen_u64(keys_slice, values_slice)
    })?;
    convert_single_result(py, result)
}

#[pyfunction]
fn groupby_min_i64_firstseen_u32<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result = py.allow_threads(|| {
        groupby::parallel_groupby_min_i64_firstseen_u32(keys_slice, values_slice)
    })?;
    convert_single_result(py, result)
}

#[pyfunction]
fn groupby_min_i64_firstseen_u64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result = py.allow_threads(|| {
        groupby::parallel_groupby_min_i64_firstseen_u64(keys_slice, values_slice)
    })?;
    convert_single_result(py, result)
}

#[pyfunction]
fn groupby_max_i64_firstseen_u32<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result = py.allow_threads(|| {
        groupby::parallel_groupby_max_i64_firstseen_u32(keys_slice, values_slice)
    })?;
    convert_single_result(py, result)
}

#[pyfunction]
fn groupby_max_i64_firstseen_u64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result = py.allow_threads(|| {
        groupby::parallel_groupby_max_i64_firstseen_u64(keys_slice, values_slice)
    })?;
    convert_single_result(py, result)
}

#[pyfunction]
fn groupby_count_i64_firstseen_u32<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturnI64<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result = py.allow_threads(|| {
        groupby::parallel_groupby_count_i64_firstseen_u32(keys_slice, values_slice)
    })?;
    convert_single_result_i64(py, result)
}

#[pyfunction]
fn groupby_count_i64_firstseen_u64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturnI64<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result = py.allow_threads(|| {
        groupby::parallel_groupby_count_i64_firstseen_u64(keys_slice, values_slice)
    })?;
    convert_single_result_i64(py, result)
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
/// Returns (keys_cols, values_1d) where keys_cols is a list of n_keys 1D arrays
/// (each length n_groups). Older extensions returned a single 2D array.
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
) -> PyResult<MultiGroupByReturnI64<'py>> {
    let values_slice = zero_copy::get_slice_f64(&values)?;
    let key_slices: Vec<&[i64]> = key_cols
        .iter()
        .map(|col| zero_copy::get_slice_i64(col))
        .collect::<PyResult<Vec<_>>>()?;

    let key_lengths: Vec<usize> = key_slices.iter().map(|s| s.len()).collect();
    validate_multi_inputs(&key_lengths, values_slice.len())?;

    let result =
        py.allow_threads(|| groupby_multi::multi_groupby_count_f64(&key_slices, values_slice))?;

    convert_multi_result_i64(py, result)
}

/// Multi-column groupby count for i64 values.
#[pyfunction]
fn groupby_multi_count_i64<'py>(
    py: Python<'py>,
    key_cols: Vec<PyReadonlyArray1<'py, i64>>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<MultiGroupByReturnI64<'py>> {
    let values_slice = zero_copy::get_slice_i64(&values)?;
    let key_slices: Vec<&[i64]> = key_cols
        .iter()
        .map(|col| zero_copy::get_slice_i64(col))
        .collect::<PyResult<Vec<_>>>()?;

    let key_lengths: Vec<usize> = key_slices.iter().map(|s| s.len()).collect();
    validate_multi_inputs(&key_lengths, values_slice.len())?;

    let result =
        py.allow_threads(|| groupby_multi::multi_groupby_count_i64(&key_slices, values_slice))?;

    convert_multi_result_i64(py, result)
}

// =============================================================================
// Multi-key groupby (sorted, for sort=True semantics)
// =============================================================================

/// Multi-column groupby sum for f64 values (sorted by key tuple).
#[pyfunction]
fn groupby_multi_sum_f64_sorted<'py>(
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

    let result = py
        .allow_threads(|| groupby_multi::multi_groupby_sum_f64_sorted(&key_slices, values_slice))?;

    convert_multi_result(py, result)
}

/// Multi-column groupby mean for f64 values (sorted by key tuple).
#[pyfunction]
fn groupby_multi_mean_f64_sorted<'py>(
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

    let result = py.allow_threads(|| {
        groupby_multi::multi_groupby_mean_f64_sorted(&key_slices, values_slice)
    })?;

    convert_multi_result(py, result)
}

/// Multi-column groupby min for f64 values (sorted by key tuple).
#[pyfunction]
fn groupby_multi_min_f64_sorted<'py>(
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

    let result = py
        .allow_threads(|| groupby_multi::multi_groupby_min_f64_sorted(&key_slices, values_slice))?;

    convert_multi_result(py, result)
}

/// Multi-column groupby max for f64 values (sorted by key tuple).
#[pyfunction]
fn groupby_multi_max_f64_sorted<'py>(
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

    let result = py
        .allow_threads(|| groupby_multi::multi_groupby_max_f64_sorted(&key_slices, values_slice))?;

    convert_multi_result(py, result)
}

/// Multi-column groupby count for f64 values (sorted by key tuple).
#[pyfunction]
fn groupby_multi_count_f64_sorted<'py>(
    py: Python<'py>,
    key_cols: Vec<PyReadonlyArray1<'py, i64>>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<MultiGroupByReturnI64<'py>> {
    let values_slice = zero_copy::get_slice_f64(&values)?;
    let key_slices: Vec<&[i64]> = key_cols
        .iter()
        .map(|col| zero_copy::get_slice_i64(col))
        .collect::<PyResult<Vec<_>>>()?;

    let key_lengths: Vec<usize> = key_slices.iter().map(|s| s.len()).collect();
    validate_multi_inputs(&key_lengths, values_slice.len())?;

    let result = py.allow_threads(|| {
        groupby_multi::multi_groupby_count_f64_sorted(&key_slices, values_slice)
    })?;

    convert_multi_result_i64(py, result)
}

/// Multi-column groupby sum for i64 values (sorted by key tuple).
#[pyfunction]
fn groupby_multi_sum_i64_sorted<'py>(
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

    let result = py
        .allow_threads(|| groupby_multi::multi_groupby_sum_i64_sorted(&key_slices, values_slice))?;

    convert_multi_result(py, result)
}

/// Multi-column groupby mean for i64 values (sorted by key tuple).
#[pyfunction]
fn groupby_multi_mean_i64_sorted<'py>(
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

    let result = py.allow_threads(|| {
        groupby_multi::multi_groupby_mean_i64_sorted(&key_slices, values_slice)
    })?;

    convert_multi_result(py, result)
}

/// Multi-column groupby min for i64 values (sorted by key tuple).
#[pyfunction]
fn groupby_multi_min_i64_sorted<'py>(
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

    let result = py
        .allow_threads(|| groupby_multi::multi_groupby_min_i64_sorted(&key_slices, values_slice))?;

    convert_multi_result(py, result)
}

/// Multi-column groupby max for i64 values (sorted by key tuple).
#[pyfunction]
fn groupby_multi_max_i64_sorted<'py>(
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

    let result = py
        .allow_threads(|| groupby_multi::multi_groupby_max_i64_sorted(&key_slices, values_slice))?;

    convert_multi_result(py, result)
}

/// Multi-column groupby count for i64 values (sorted by key tuple).
#[pyfunction]
fn groupby_multi_count_i64_sorted<'py>(
    py: Python<'py>,
    key_cols: Vec<PyReadonlyArray1<'py, i64>>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<MultiGroupByReturnI64<'py>> {
    let values_slice = zero_copy::get_slice_i64(&values)?;
    let key_slices: Vec<&[i64]> = key_cols
        .iter()
        .map(|col| zero_copy::get_slice_i64(col))
        .collect::<PyResult<Vec<_>>>()?;

    let key_lengths: Vec<usize> = key_slices.iter().map(|s| s.len()).collect();
    validate_multi_inputs(&key_lengths, values_slice.len())?;

    let result = py.allow_threads(|| {
        groupby_multi::multi_groupby_count_i64_sorted(&key_slices, values_slice)
    })?;

    convert_multi_result_i64(py, result)
}

#[pyfunction]
fn groupby_multi_sum_f64_firstseen_u32<'py>(
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

    let result = py.allow_threads(|| {
        groupby_multi::multi_groupby_sum_f64_firstseen_u32(&key_slices, values_slice)
    })?;
    convert_multi_result(py, result)
}

#[pyfunction]
fn groupby_multi_sum_f64_firstseen_u64<'py>(
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

    let result = py.allow_threads(|| {
        groupby_multi::multi_groupby_sum_f64_firstseen_u64(&key_slices, values_slice)
    })?;
    convert_multi_result(py, result)
}

#[pyfunction]
fn groupby_multi_mean_f64_firstseen_u32<'py>(
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

    let result = py.allow_threads(|| {
        groupby_multi::multi_groupby_mean_f64_firstseen_u32(&key_slices, values_slice)
    })?;
    convert_multi_result(py, result)
}

#[pyfunction]
fn groupby_multi_mean_f64_firstseen_u64<'py>(
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

    let result = py.allow_threads(|| {
        groupby_multi::multi_groupby_mean_f64_firstseen_u64(&key_slices, values_slice)
    })?;
    convert_multi_result(py, result)
}

#[pyfunction]
fn groupby_multi_min_f64_firstseen_u32<'py>(
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

    let result = py.allow_threads(|| {
        groupby_multi::multi_groupby_min_f64_firstseen_u32(&key_slices, values_slice)
    })?;
    convert_multi_result(py, result)
}

#[pyfunction]
fn groupby_multi_min_f64_firstseen_u64<'py>(
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

    let result = py.allow_threads(|| {
        groupby_multi::multi_groupby_min_f64_firstseen_u64(&key_slices, values_slice)
    })?;
    convert_multi_result(py, result)
}

#[pyfunction]
fn groupby_multi_max_f64_firstseen_u32<'py>(
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

    let result = py.allow_threads(|| {
        groupby_multi::multi_groupby_max_f64_firstseen_u32(&key_slices, values_slice)
    })?;
    convert_multi_result(py, result)
}

#[pyfunction]
fn groupby_multi_max_f64_firstseen_u64<'py>(
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

    let result = py.allow_threads(|| {
        groupby_multi::multi_groupby_max_f64_firstseen_u64(&key_slices, values_slice)
    })?;
    convert_multi_result(py, result)
}

#[pyfunction]
fn groupby_multi_count_f64_firstseen_u32<'py>(
    py: Python<'py>,
    key_cols: Vec<PyReadonlyArray1<'py, i64>>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<MultiGroupByReturnI64<'py>> {
    let values_slice = zero_copy::get_slice_f64(&values)?;
    let key_slices: Vec<&[i64]> = key_cols
        .iter()
        .map(|col| zero_copy::get_slice_i64(col))
        .collect::<PyResult<Vec<_>>>()?;
    let key_lengths: Vec<usize> = key_slices.iter().map(|s| s.len()).collect();
    validate_multi_inputs(&key_lengths, values_slice.len())?;

    let result = py.allow_threads(|| {
        groupby_multi::multi_groupby_count_f64_firstseen_u32(&key_slices, values_slice)
    })?;
    convert_multi_result_i64(py, result)
}

#[pyfunction]
fn groupby_multi_count_f64_firstseen_u64<'py>(
    py: Python<'py>,
    key_cols: Vec<PyReadonlyArray1<'py, i64>>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<MultiGroupByReturnI64<'py>> {
    let values_slice = zero_copy::get_slice_f64(&values)?;
    let key_slices: Vec<&[i64]> = key_cols
        .iter()
        .map(|col| zero_copy::get_slice_i64(col))
        .collect::<PyResult<Vec<_>>>()?;
    let key_lengths: Vec<usize> = key_slices.iter().map(|s| s.len()).collect();
    validate_multi_inputs(&key_lengths, values_slice.len())?;

    let result = py.allow_threads(|| {
        groupby_multi::multi_groupby_count_f64_firstseen_u64(&key_slices, values_slice)
    })?;
    convert_multi_result_i64(py, result)
}

#[pyfunction]
fn groupby_multi_sum_i64_firstseen_u32<'py>(
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

    let result = py.allow_threads(|| {
        groupby_multi::multi_groupby_sum_i64_firstseen_u32(&key_slices, values_slice)
    })?;
    convert_multi_result(py, result)
}

#[pyfunction]
fn groupby_multi_sum_i64_firstseen_u64<'py>(
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

    let result = py.allow_threads(|| {
        groupby_multi::multi_groupby_sum_i64_firstseen_u64(&key_slices, values_slice)
    })?;
    convert_multi_result(py, result)
}

#[pyfunction]
fn groupby_multi_mean_i64_firstseen_u32<'py>(
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

    let result = py.allow_threads(|| {
        groupby_multi::multi_groupby_mean_i64_firstseen_u32(&key_slices, values_slice)
    })?;
    convert_multi_result(py, result)
}

#[pyfunction]
fn groupby_multi_mean_i64_firstseen_u64<'py>(
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

    let result = py.allow_threads(|| {
        groupby_multi::multi_groupby_mean_i64_firstseen_u64(&key_slices, values_slice)
    })?;
    convert_multi_result(py, result)
}

#[pyfunction]
fn groupby_multi_min_i64_firstseen_u32<'py>(
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

    let result = py.allow_threads(|| {
        groupby_multi::multi_groupby_min_i64_firstseen_u32(&key_slices, values_slice)
    })?;
    convert_multi_result(py, result)
}

#[pyfunction]
fn groupby_multi_min_i64_firstseen_u64<'py>(
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

    let result = py.allow_threads(|| {
        groupby_multi::multi_groupby_min_i64_firstseen_u64(&key_slices, values_slice)
    })?;
    convert_multi_result(py, result)
}

#[pyfunction]
fn groupby_multi_max_i64_firstseen_u32<'py>(
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

    let result = py.allow_threads(|| {
        groupby_multi::multi_groupby_max_i64_firstseen_u32(&key_slices, values_slice)
    })?;
    convert_multi_result(py, result)
}

#[pyfunction]
fn groupby_multi_max_i64_firstseen_u64<'py>(
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

    let result = py.allow_threads(|| {
        groupby_multi::multi_groupby_max_i64_firstseen_u64(&key_slices, values_slice)
    })?;
    convert_multi_result(py, result)
}

#[pyfunction]
fn groupby_multi_count_i64_firstseen_u32<'py>(
    py: Python<'py>,
    key_cols: Vec<PyReadonlyArray1<'py, i64>>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<MultiGroupByReturnI64<'py>> {
    let values_slice = zero_copy::get_slice_i64(&values)?;
    let key_slices: Vec<&[i64]> = key_cols
        .iter()
        .map(|col| zero_copy::get_slice_i64(col))
        .collect::<PyResult<Vec<_>>>()?;
    let key_lengths: Vec<usize> = key_slices.iter().map(|s| s.len()).collect();
    validate_multi_inputs(&key_lengths, values_slice.len())?;

    let result = py.allow_threads(|| {
        groupby_multi::multi_groupby_count_i64_firstseen_u32(&key_slices, values_slice)
    })?;
    convert_multi_result_i64(py, result)
}

#[pyfunction]
fn groupby_multi_count_i64_firstseen_u64<'py>(
    py: Python<'py>,
    key_cols: Vec<PyReadonlyArray1<'py, i64>>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<MultiGroupByReturnI64<'py>> {
    let values_slice = zero_copy::get_slice_i64(&values)?;
    let key_slices: Vec<&[i64]> = key_cols
        .iter()
        .map(|col| zero_copy::get_slice_i64(col))
        .collect::<PyResult<Vec<_>>>()?;
    let key_lengths: Vec<usize> = key_slices.iter().map(|s| s.len()).collect();
    validate_multi_inputs(&key_lengths, values_slice.len())?;
    let result = py.allow_threads(|| {
        groupby_multi::multi_groupby_count_i64_firstseen_u64(&key_slices, values_slice)
    })?;
    convert_multi_result_i64(py, result)
}

fn convert_multi_result_f64<'py>(
    py: Python<'py>,
    result: groupby_multi::GroupByMultiResultF64,
) -> PyResult<MultiGroupByReturnF64<'py>> {
    let n_groups = result.values.len();
    let n_keys = result.n_keys;
    debug_assert_eq!(result.keys_flat.len(), n_groups * n_keys);

    let perm = result.perm.as_deref();
    if let Some(p) = perm {
        debug_assert_eq!(p.len(), n_groups);
    }

    // Fast path: for small outputs, build Rust Vec columns and transfer ownership into NumPy.
    // This avoids per-element pointer writes and Rayon overhead.
    const SMALL_DIRECT_THRESHOLD: usize = 200_000; // measured in elements (n_groups * n_keys)
    if perm.is_none() && n_groups.saturating_mul(n_keys) <= SMALL_DIRECT_THRESHOLD {
        let mut key_cols: Vec<Vec<i64>> =
            (0..n_keys).map(|_| Vec::with_capacity(n_groups)).collect();
        for g in 0..n_groups {
            let base = g * n_keys;
            for (col, key_col) in key_cols.iter_mut().enumerate() {
                key_col.push(result.keys_flat[base + col]);
            }
        }

        let key_arrays: Vec<Bound<'py, PyArray1<i64>>> = key_cols
            .into_iter()
            .map(|col| col.into_pyarray_bound(py))
            .collect();
        let values_1d = result.values.into_pyarray_bound(py);
        return Ok((key_arrays, values_1d));
    }

    // Allocate output arrays under the GIL.
    // SAFETY: we will initialize all elements before returning to Python.
    let key_arrays: Vec<Bound<'py, PyArray1<i64>>> = (0..n_keys)
        .map(|_| unsafe { PyArray1::<i64>::new_bound(py, n_groups, false) })
        .collect();
    let values_out = unsafe { PyArray1::<f64>::new_bound(py, n_groups, false) };

    // Extract raw pointers under GIL; fill outside GIL.
    let key_ptrs: Vec<usize> = key_arrays.iter().map(|a| a.data() as usize).collect();
    let values_ptr: usize = values_out.data() as usize;

    let keys_flat = result.keys_flat;
    let values = result.values;

    py.allow_threads(|| {
        // Rayon overhead can dominate for small outputs.
        if n_groups < 50_000 {
            for out_g in 0..n_groups {
                let src_g = perm.map_or(out_g, |p| p[out_g]);
                let base = src_g * n_keys;
                unsafe {
                    *((values_ptr as *mut f64).add(out_g)) = values[src_g];
                    for col in 0..n_keys {
                        *((key_ptrs[col] as *mut i64).add(out_g)) = keys_flat[base + col];
                    }
                }
            }
            return;
        }

        let chunk = 1usize.max(n_groups / (rayon::current_num_threads() * 8));
        (0..n_groups)
            .into_par_iter()
            .with_min_len(chunk)
            .for_each(|out_g| {
                let src_g = perm.map_or(out_g, |p| p[out_g]);
                let base = src_g * n_keys;
                unsafe {
                    *((values_ptr as *mut f64).add(out_g)) = values[src_g];
                    for col in 0..n_keys {
                        *((key_ptrs[col] as *mut i64).add(out_g)) = keys_flat[base + col];
                    }
                }
            });
    });

    Ok((key_arrays, values_out))
}

fn convert_multi_result_i64<'py>(
    py: Python<'py>,
    result: groupby_multi::GroupByMultiResultI64,
) -> PyResult<MultiGroupByReturnI64<'py>> {
    let n_groups = result.values.len();
    let n_keys = result.n_keys;
    debug_assert_eq!(result.keys_flat.len(), n_groups * n_keys);

    let perm = result.perm.as_deref();
    if let Some(p) = perm {
        debug_assert_eq!(p.len(), n_groups);
    }

    const SMALL_DIRECT_THRESHOLD: usize = 200_000; // measured in elements (n_groups * n_keys)
    if perm.is_none() && n_groups.saturating_mul(n_keys) <= SMALL_DIRECT_THRESHOLD {
        let mut key_cols: Vec<Vec<i64>> =
            (0..n_keys).map(|_| Vec::with_capacity(n_groups)).collect();
        for g in 0..n_groups {
            let base = g * n_keys;
            for (col, key_col) in key_cols.iter_mut().enumerate() {
                key_col.push(result.keys_flat[base + col]);
            }
        }

        let key_arrays: Vec<Bound<'py, PyArray1<i64>>> = key_cols
            .into_iter()
            .map(|col| col.into_pyarray_bound(py))
            .collect();
        let values_1d = result.values.into_pyarray_bound(py);
        return Ok((key_arrays, values_1d));
    }

    let key_arrays: Vec<Bound<'py, PyArray1<i64>>> = (0..n_keys)
        .map(|_| unsafe { PyArray1::<i64>::new_bound(py, n_groups, false) })
        .collect();
    let values_out = unsafe { PyArray1::<i64>::new_bound(py, n_groups, false) };

    let key_ptrs: Vec<usize> = key_arrays.iter().map(|a| a.data() as usize).collect();
    let values_ptr: usize = values_out.data() as usize;

    let keys_flat = result.keys_flat;
    let values = result.values;

    py.allow_threads(|| {
        if n_groups < 50_000 {
            for out_g in 0..n_groups {
                let src_g = perm.map_or(out_g, |p| p[out_g]);
                let base = src_g * n_keys;
                unsafe {
                    *((values_ptr as *mut i64).add(out_g)) = values[src_g];
                    for col in 0..n_keys {
                        *((key_ptrs[col] as *mut i64).add(out_g)) = keys_flat[base + col];
                    }
                }
            }
            return;
        }

        let chunk = 1usize.max(n_groups / (rayon::current_num_threads() * 8));
        (0..n_groups)
            .into_par_iter()
            .with_min_len(chunk)
            .for_each(|out_g| {
                let src_g = perm.map_or(out_g, |p| p[out_g]);
                let base = src_g * n_keys;
                unsafe {
                    *((values_ptr as *mut i64).add(out_g)) = values[src_g];
                    for col in 0..n_keys {
                        *((key_ptrs[col] as *mut i64).add(out_g)) = keys_flat[base + col];
                    }
                }
            });
    });

    Ok((key_arrays, values_out))
}

fn convert_multi_result<'py>(
    py: Python<'py>,
    result: groupby_multi::GroupByMultiResultF64,
) -> PyResult<MultiGroupByReturn<'py>> {
    convert_multi_result_f64(py, result)
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

    // Single-key groupby (sorted, for sort=True)
    m.add_function(wrap_pyfunction!(groupby_sum_f64_sorted, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_mean_f64_sorted, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_min_f64_sorted, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_max_f64_sorted, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_count_f64_sorted, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_sum_i64_sorted, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_mean_i64_sorted, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_min_i64_sorted, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_max_i64_sorted, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_count_i64_sorted, m)?)?;

    // Single-key groupby (first-seen order, for sort=False)
    m.add_function(wrap_pyfunction!(groupby_sum_f64_firstseen_u32, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_sum_f64_firstseen_u64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_mean_f64_firstseen_u32, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_mean_f64_firstseen_u64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_min_f64_firstseen_u32, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_min_f64_firstseen_u64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_max_f64_firstseen_u32, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_max_f64_firstseen_u64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_count_f64_firstseen_u32, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_count_f64_firstseen_u64, m)?)?;

    m.add_function(wrap_pyfunction!(groupby_sum_i64_firstseen_u32, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_sum_i64_firstseen_u64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_mean_i64_firstseen_u32, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_mean_i64_firstseen_u64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_min_i64_firstseen_u32, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_min_i64_firstseen_u64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_max_i64_firstseen_u32, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_max_i64_firstseen_u64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_count_i64_firstseen_u32, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_count_i64_firstseen_u64, m)?)?;
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

    // Multi-key groupby (sorted, for sort=True)
    m.add_function(wrap_pyfunction!(groupby_multi_sum_f64_sorted, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_mean_f64_sorted, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_min_f64_sorted, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_max_f64_sorted, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_count_f64_sorted, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_sum_i64_sorted, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_mean_i64_sorted, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_min_i64_sorted, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_max_i64_sorted, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_count_i64_sorted, m)?)?;

    // Multi-key groupby (first-seen order, for sort=False)
    m.add_function(wrap_pyfunction!(groupby_multi_sum_f64_firstseen_u32, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_sum_f64_firstseen_u64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_mean_f64_firstseen_u32, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_mean_f64_firstseen_u64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_min_f64_firstseen_u32, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_min_f64_firstseen_u64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_max_f64_firstseen_u32, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_max_f64_firstseen_u64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_count_f64_firstseen_u32, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_count_f64_firstseen_u64, m)?)?;

    m.add_function(wrap_pyfunction!(groupby_multi_sum_i64_firstseen_u32, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_sum_i64_firstseen_u64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_mean_i64_firstseen_u32, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_mean_i64_firstseen_u64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_min_i64_firstseen_u32, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_min_i64_firstseen_u64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_max_i64_firstseen_u32, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_max_i64_firstseen_u64, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_count_i64_firstseen_u32, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_multi_count_i64_firstseen_u64, m)?)?;
    // Utilities
    m.add_function(wrap_pyfunction!(get_fallback_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(get_thread_count, m)?)?;
    Ok(())
}
