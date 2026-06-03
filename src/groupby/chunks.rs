use pyo3::prelude::*;

use super::order::FirstSeenRowIndex;

const DETERMINISTIC_TARGET_CHUNK_SIZE: usize = 131_072;
const DETERMINISTIC_MIN_CHUNKS: usize = 4;
const DETERMINISTIC_MAX_CHUNKS: usize = 2048;

pub(super) fn fixed_chunk_size(n_rows: usize) -> usize {
    if n_rows == 0 {
        return 1;
    }
    let n_chunks = n_rows
        .div_ceil(DETERMINISTIC_TARGET_CHUNK_SIZE)
        .clamp(DETERMINISTIC_MIN_CHUNKS, DETERMINISTIC_MAX_CHUNKS);
    n_rows.div_ceil(n_chunks)
}

pub(super) fn build_chunk_ranges(n_rows: usize, chunk_size: usize) -> Vec<(usize, usize)> {
    if n_rows == 0 {
        return Vec::new();
    }
    let mut ranges = Vec::with_capacity(n_rows.div_ceil(chunk_size));
    let mut start = 0usize;
    while start < n_rows {
        let end = (start + chunk_size).min(n_rows);
        ranges.push((start, end));
        start = end;
    }
    ranges
}

pub(super) fn validate_firstseen_deterministic_inputs<T, I>(
    keys: &[i64],
    values: &[T],
) -> PyResult<usize>
where
    I: FirstSeenRowIndex,
{
    let n_rows = keys.len();
    if values.len() != n_rows {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "keys and values must have same length",
        ));
    }
    I::validate_n_rows(n_rows)?;
    Ok(n_rows)
}

pub(super) fn prepare_firstseen_deterministic_ranges<T, I>(
    keys: &[i64],
    values: &[T],
) -> PyResult<Vec<(usize, usize)>>
where
    I: FirstSeenRowIndex,
{
    let n_rows = validate_firstseen_deterministic_inputs::<T, I>(keys, values)?;
    let chunk_size = fixed_chunk_size(n_rows);
    Ok(build_chunk_ranges(n_rows, chunk_size))
}
