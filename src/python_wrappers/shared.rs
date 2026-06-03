use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub(crate) type MultiGroupByKeysReturn<'py> = Vec<Bound<'py, PyArray1<i64>>>;

pub(crate) type MultiGroupByReturnF64<'py> =
    (MultiGroupByKeysReturn<'py>, Bound<'py, PyArray1<f64>>);
pub(crate) type MultiGroupByReturnI64<'py> =
    (MultiGroupByKeysReturn<'py>, Bound<'py, PyArray1<i64>>);

pub(crate) type MultiGroupByReturn<'py> = MultiGroupByReturnF64<'py>;

pub(crate) type SingleGroupByReturnF64<'py> =
    (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>);
pub(crate) type SingleGroupByReturnI64<'py> =
    (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>);
pub(crate) type SingleGroupByProfileReturnF64<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyDict>,
);

pub(crate) type SingleGroupByReturn<'py> = SingleGroupByReturnF64<'py>;

pub(crate) const FALLBACK_THRESHOLD: usize = 100_000;

pub(crate) fn validate_inputs_len(keys_len: usize, values_len: usize) -> PyResult<()> {
    if keys_len != values_len {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "keys and values must have the same length, got {} and {}",
            keys_len, values_len
        )));
    }
    Ok(())
}

pub(crate) fn validate_inputs(keys_len: usize, values_len: usize) -> PyResult<()> {
    validate_inputs_len(keys_len, values_len)?;
    if keys_len < FALLBACK_THRESHOLD {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dataset too small for acceleration (got {}, need at least {}), use pandas directly",
            keys_len, FALLBACK_THRESHOLD
        )));
    }
    Ok(())
}

pub(crate) fn validate_multi_inputs_len(key_lengths: &[usize], values_len: usize) -> PyResult<()> {
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
    Ok(())
}

pub(crate) fn validate_multi_inputs(key_lengths: &[usize], values_len: usize) -> PyResult<()> {
    validate_multi_inputs_len(key_lengths, values_len)?;
    if values_len < FALLBACK_THRESHOLD {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dataset too small for acceleration (got {}, need at least {}), use pandas directly",
            values_len, FALLBACK_THRESHOLD
        )));
    }
    Ok(())
}
