use super::*;

/// Computes parallel groupby sum for f64 values.
///
/// Releases the GIL during computation for true parallelism.
#[pyfunction]
pub(crate) fn groupby_sum_f64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result = py.detach(|| groupby::parallel_groupby_sum_f64(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

/// Computes parallel groupby mean for f64 values. Returns NaN for all-NaN groups.
#[pyfunction]
pub(crate) fn groupby_mean_f64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result = py.detach(|| groupby::parallel_groupby_mean_f64(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

/// Computes parallel groupby sample variance for f64 values.
#[pyfunction]
pub(crate) fn groupby_var_f64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs_len(keys_slice.len(), values_slice.len())?;

    let result = py.detach(|| groupby::parallel_groupby_var_f64(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

/// Computes parallel groupby sample standard deviation for f64 values.
#[pyfunction]
pub(crate) fn groupby_std_f64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs_len(keys_slice.len(), values_slice.len())?;

    let result = py.detach(|| groupby::parallel_groupby_std_f64(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

/// Computes parallel groupby min for f64 values. Returns NaN for all-NaN groups.
#[pyfunction]
pub(crate) fn groupby_min_f64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result = py.detach(|| groupby::parallel_groupby_min_f64(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

/// Computes parallel groupby max for f64 values. Returns NaN for all-NaN groups.
#[pyfunction]
pub(crate) fn groupby_max_f64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result = py.detach(|| groupby::parallel_groupby_max_f64(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

/// Computes parallel groupby count for f64 values. Counts non-NaN values.
#[pyfunction]
pub(crate) fn groupby_count_f64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturnI64<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result = py.detach(|| groupby::parallel_groupby_count_f64(keys_slice, values_slice))?;
    convert_single_result_i64(py, result)
}

/// Computes parallel groupby sum for i64 values and returns i64 (Pandas-style wrap semantics).
#[pyfunction]
pub(crate) fn groupby_sum_i64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturnI64<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result = py.detach(|| groupby::parallel_groupby_sum_i64(keys_slice, values_slice))?;
    convert_single_result_i64(py, result)
}

/// Computes parallel groupby mean for i64 values.
#[pyfunction]
pub(crate) fn groupby_mean_i64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result = py.detach(|| groupby::parallel_groupby_mean_i64(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

/// Computes parallel groupby sample variance for i64 values.
#[pyfunction]
pub(crate) fn groupby_var_i64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs_len(keys_slice.len(), values_slice.len())?;

    let result = py.detach(|| groupby::parallel_groupby_var_i64(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

/// Computes parallel groupby sample standard deviation for i64 values.
#[pyfunction]
pub(crate) fn groupby_std_i64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs_len(keys_slice.len(), values_slice.len())?;

    let result = py.detach(|| groupby::parallel_groupby_std_i64(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

/// Computes parallel groupby min for i64 values.
#[pyfunction]
pub(crate) fn groupby_min_i64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturnI64<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result = py.detach(|| groupby::parallel_groupby_min_i64(keys_slice, values_slice))?;
    convert_single_result_i64(py, result)
}

/// Computes parallel groupby max for i64 values.
#[pyfunction]
pub(crate) fn groupby_max_i64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturnI64<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result = py.detach(|| groupby::parallel_groupby_max_i64(keys_slice, values_slice))?;
    convert_single_result_i64(py, result)
}

/// Computes parallel groupby count for i64 values.
#[pyfunction]
pub(crate) fn groupby_count_i64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturnI64<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result = py.detach(|| groupby::parallel_groupby_count_i64(keys_slice, values_slice))?;
    convert_single_result_i64(py, result)
}

// =============================================================================
// Single-key groupby (sorted, for sort=True semantics)
// =============================================================================
