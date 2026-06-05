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
//!   for common groupby operations including sum, product, mean, median, variance,
//!   standard deviation, min, max, and count.
//! - [`groupby`]: Implements parallel map-reduce groupby using Rayon's `par_chunks`.
//! - [`zero_copy`]: Utilities for safely borrowing NumPy arrays as Rust slices.
//!
//! ## Performance Notes
//!
//! - **GIL Release**: All heavy computations use `py.detach()` to release the Python
//!   Global Interpreter Lock, enabling true parallelism.
//! - **Fallback Threshold**: Datasets smaller than [`FALLBACK_THRESHOLD`] (100,000 rows) should
//!   use native Pandas instead, as the overhead of Rust dispatch outweighs benefits.
//! - **Zero-Copy**: NumPy arrays are accessed directly without copying when C-contiguous.

#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;

pub mod aggregation;
pub mod groupby;
pub mod groupby_multi;
mod python_wrappers;
pub mod radix_groupby;
pub mod radix_sort;
pub mod zero_copy;

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python_wrappers::register::register(m)
}

#[cfg(test)]
mod tests {
    use crate::python_wrappers::shared::{
        validate_inputs, validate_inputs_len, validate_multi_inputs, validate_multi_inputs_len,
        FALLBACK_THRESHOLD,
    };

    #[test]
    fn test_single_key_std_var_boundary_skips_threshold_only_for_len_only_validator() {
        assert!(validate_inputs_len(6, 6).is_ok());
        assert!(validate_inputs_len(6, 5).is_err());

        assert!(validate_inputs(6, 6).is_err());
        assert!(validate_inputs(FALLBACK_THRESHOLD, FALLBACK_THRESHOLD).is_ok());
    }

    #[test]
    fn test_multi_key_std_var_boundary_skips_threshold_only_for_len_only_validator() {
        assert!(validate_multi_inputs_len(&[6, 6], 6).is_ok());
        assert!(validate_multi_inputs_len(&[6, 5], 6).is_err());

        assert!(validate_multi_inputs(&[6, 6], 6).is_err());
        assert!(validate_multi_inputs(&[FALLBACK_THRESHOLD], FALLBACK_THRESHOLD).is_ok());
    }
}
