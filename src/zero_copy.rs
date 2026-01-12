//! Zero-copy utilities for accessing NumPy arrays as Rust slices.
//!
//! These functions safely borrow NumPy array data without copying,
//! but require the arrays to be C-contiguous (row-major order).

use numpy::PyReadonlyArray1;
use pyo3::PyResult;

/// Borrows a C-contiguous f64 NumPy array as a Rust slice.
///
/// # Errors
/// Returns `PyValueError` if the array is not C-contiguous.
pub fn get_slice_f64<'a>(arr: &'a PyReadonlyArray1<'_, f64>) -> PyResult<&'a [f64]> {
    arr.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(
            "Array must be C-contiguous. Use np.ascontiguousarray() first.",
        )
    })
}

/// Borrows a C-contiguous i64 NumPy array as a Rust slice.
///
/// # Errors
/// Returns `PyValueError` if the array is not C-contiguous.
pub fn get_slice_i64<'a>(arr: &'a PyReadonlyArray1<'_, i64>) -> PyResult<&'a [i64]> {
    arr.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(
            "Array must be C-contiguous. Use np.ascontiguousarray() first.",
        )
    })
}
