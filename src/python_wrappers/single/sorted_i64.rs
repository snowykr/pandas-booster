use super::*;

#[pyfunction]
pub(crate) fn groupby_sum_i64_sorted<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturnI64<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result =
        py.detach(|| groupby::parallel_groupby_sum_i64_sorted(keys_slice, values_slice))?;
    convert_single_result_i64(py, result)
}

#[pyfunction]
pub(crate) fn groupby_mean_i64_sorted<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result =
        py.detach(|| groupby::parallel_groupby_mean_i64_sorted(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

#[pyfunction]
pub(crate) fn groupby_var_i64_sorted<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs_len(keys_slice.len(), values_slice.len())?;

    let result =
        py.detach(|| groupby::parallel_groupby_var_i64_sorted(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

#[pyfunction]
pub(crate) fn groupby_std_i64_sorted<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs_len(keys_slice.len(), values_slice.len())?;

    let result =
        py.detach(|| groupby::parallel_groupby_std_i64_sorted(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

#[pyfunction]
pub(crate) fn groupby_min_i64_sorted<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturnI64<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result =
        py.detach(|| groupby::parallel_groupby_min_i64_sorted(keys_slice, values_slice))?;
    convert_single_result_i64(py, result)
}

#[pyfunction]
pub(crate) fn groupby_max_i64_sorted<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturnI64<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result =
        py.detach(|| groupby::parallel_groupby_max_i64_sorted(keys_slice, values_slice))?;
    convert_single_result_i64(py, result)
}

#[pyfunction]
pub(crate) fn groupby_count_i64_sorted<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<SingleGroupByReturnI64<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_i64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result =
        py.detach(|| groupby::parallel_groupby_count_i64_sorted(keys_slice, values_slice))?;
    convert_single_result_i64(py, result)
}

// =============================================================================
// First-seen ordered groupby (sort=False semantics)
// =============================================================================
