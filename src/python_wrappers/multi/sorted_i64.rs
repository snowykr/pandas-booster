use super::*;

/// Multi-column groupby sum for i64 values (sorted by key tuple).
#[pyfunction]
pub(crate) fn groupby_multi_sum_i64_sorted<'py>(
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
        py.detach(|| groupby_multi::multi_groupby_sum_i64_sorted(&key_slices, values_slice))?;

    convert_multi_result_i64(py, result)
}

/// Multi-column groupby mean for i64 values (sorted by key tuple).
#[pyfunction]
pub(crate) fn groupby_multi_mean_i64_sorted<'py>(
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
        py.detach(|| groupby_multi::multi_groupby_mean_i64_sorted(&key_slices, values_slice))?;

    convert_multi_result(py, result)
}

/// Multi-column groupby variance for i64 values (sorted by key tuple).
#[pyfunction]
pub(crate) fn groupby_multi_var_i64_sorted<'py>(
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
    validate_multi_inputs_len(&key_lengths, values_slice.len())?;

    let result =
        py.detach(|| groupby_multi::multi_groupby_var_i64_sorted(&key_slices, values_slice))?;

    convert_multi_result(py, result)
}

/// Multi-column groupby standard deviation for i64 values (sorted by key tuple).
#[pyfunction]
pub(crate) fn groupby_multi_std_i64_sorted<'py>(
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
    validate_multi_inputs_len(&key_lengths, values_slice.len())?;

    let result =
        py.detach(|| groupby_multi::multi_groupby_std_i64_sorted(&key_slices, values_slice))?;

    convert_multi_result(py, result)
}

/// Multi-column groupby min for i64 values (sorted by key tuple).
#[pyfunction]
pub(crate) fn groupby_multi_min_i64_sorted<'py>(
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
        py.detach(|| groupby_multi::multi_groupby_min_i64_sorted(&key_slices, values_slice))?;

    convert_multi_result_i64(py, result)
}

/// Multi-column groupby max for i64 values (sorted by key tuple).
#[pyfunction]
pub(crate) fn groupby_multi_max_i64_sorted<'py>(
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
        py.detach(|| groupby_multi::multi_groupby_max_i64_sorted(&key_slices, values_slice))?;

    convert_multi_result_i64(py, result)
}

/// Multi-column groupby count for i64 values (sorted by key tuple).
#[pyfunction]
pub(crate) fn groupby_multi_count_i64_sorted<'py>(
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
        py.detach(|| groupby_multi::multi_groupby_count_i64_sorted(&key_slices, values_slice))?;

    convert_multi_result_i64(py, result)
}
