use super::*;

#[pyfunction]
pub(crate) fn groupby_multi_sum_f64_firstseen_u32<'py>(
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
        .detach(|| groupby_multi::multi_groupby_sum_f64_firstseen_u32(&key_slices, values_slice))?;
    convert_multi_result(py, result)
}

#[pyfunction]
pub(crate) fn groupby_multi_mean_f64_firstseen_u32<'py>(
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

    let result = py.detach(|| {
        groupby_multi::multi_groupby_mean_f64_firstseen_u32(&key_slices, values_slice)
    })?;
    convert_multi_result(py, result)
}

#[pyfunction]
pub(crate) fn groupby_multi_var_f64_firstseen_u32<'py>(
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
    validate_multi_inputs_len(&key_lengths, values_slice.len())?;

    let result = py
        .detach(|| groupby_multi::multi_groupby_var_f64_firstseen_u32(&key_slices, values_slice))?;
    convert_multi_result(py, result)
}

#[pyfunction]
pub(crate) fn groupby_multi_std_f64_firstseen_u32<'py>(
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
    validate_multi_inputs_len(&key_lengths, values_slice.len())?;

    let result = py
        .detach(|| groupby_multi::multi_groupby_std_f64_firstseen_u32(&key_slices, values_slice))?;
    convert_multi_result(py, result)
}

#[pyfunction]
pub(crate) fn groupby_multi_min_f64_firstseen_u32<'py>(
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
        .detach(|| groupby_multi::multi_groupby_min_f64_firstseen_u32(&key_slices, values_slice))?;
    convert_multi_result(py, result)
}

#[pyfunction]
pub(crate) fn groupby_multi_max_f64_firstseen_u32<'py>(
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
        .detach(|| groupby_multi::multi_groupby_max_f64_firstseen_u32(&key_slices, values_slice))?;
    convert_multi_result(py, result)
}

#[pyfunction]
pub(crate) fn groupby_multi_count_f64_firstseen_u32<'py>(
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

    let result = py.detach(|| {
        groupby_multi::multi_groupby_count_f64_firstseen_u32(&key_slices, values_slice)
    })?;
    convert_multi_result_i64(py, result)
}

#[pyfunction]
pub(crate) fn groupby_multi_sum_f64_firstseen_u64<'py>(
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
        .detach(|| groupby_multi::multi_groupby_sum_f64_firstseen_u64(&key_slices, values_slice))?;
    convert_multi_result(py, result)
}

#[pyfunction]
pub(crate) fn groupby_multi_mean_f64_firstseen_u64<'py>(
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

    let result = py.detach(|| {
        groupby_multi::multi_groupby_mean_f64_firstseen_u64(&key_slices, values_slice)
    })?;
    convert_multi_result(py, result)
}

#[pyfunction]
pub(crate) fn groupby_multi_var_f64_firstseen_u64<'py>(
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
    validate_multi_inputs_len(&key_lengths, values_slice.len())?;

    let result = py
        .detach(|| groupby_multi::multi_groupby_var_f64_firstseen_u64(&key_slices, values_slice))?;
    convert_multi_result(py, result)
}

#[pyfunction]
pub(crate) fn groupby_multi_std_f64_firstseen_u64<'py>(
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
    validate_multi_inputs_len(&key_lengths, values_slice.len())?;

    let result = py
        .detach(|| groupby_multi::multi_groupby_std_f64_firstseen_u64(&key_slices, values_slice))?;
    convert_multi_result(py, result)
}

#[pyfunction]
pub(crate) fn groupby_multi_min_f64_firstseen_u64<'py>(
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
        .detach(|| groupby_multi::multi_groupby_min_f64_firstseen_u64(&key_slices, values_slice))?;
    convert_multi_result(py, result)
}

#[pyfunction]
pub(crate) fn groupby_multi_max_f64_firstseen_u64<'py>(
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
        .detach(|| groupby_multi::multi_groupby_max_f64_firstseen_u64(&key_slices, values_slice))?;
    convert_multi_result(py, result)
}

#[pyfunction]
pub(crate) fn groupby_multi_count_f64_firstseen_u64<'py>(
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

    let result = py.detach(|| {
        groupby_multi::multi_groupby_count_f64_firstseen_u64(&key_slices, values_slice)
    })?;
    convert_multi_result_i64(py, result)
}
