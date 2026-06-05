use super::*;

#[pyfunction]
pub(crate) fn groupby_sum_f64_firstseen_u64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result =
        py.detach(|| groupby::parallel_groupby_sum_f64_firstseen_u64(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

#[pyfunction]
pub(crate) fn groupby_mean_f64_firstseen_u64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result =
        py.detach(|| groupby::parallel_groupby_mean_f64_firstseen_u64(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

#[pyfunction]
pub(crate) fn groupby_var_f64_firstseen_u64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs_len(keys_slice.len(), values_slice.len())?;

    let result =
        py.detach(|| groupby::parallel_groupby_var_f64_firstseen_u64(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

#[pyfunction]
pub(crate) fn profile_groupby_var_f64_firstseen_u64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByProfileReturnF64<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs_len(keys_slice.len(), values_slice.len())?;

    let profiled = py.detach(|| {
        groupby::profile_parallel_groupby_var_f64_firstseen_u64(keys_slice, values_slice)
    })?;
    let materialize_start = Instant::now();
    let (keys_1d, values_1d) = convert_single_result_f64(py, profiled.result)?;
    let profile_dict = build_single_profile_dict(
        py,
        &profiled.profile,
        materialize_start.elapsed().as_secs_f64(),
    )?;
    Ok((keys_1d, values_1d, profile_dict))
}

#[pyfunction]
pub(crate) fn groupby_std_f64_firstseen_u64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs_len(keys_slice.len(), values_slice.len())?;

    let result =
        py.detach(|| groupby::parallel_groupby_std_f64_firstseen_u64(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

#[pyfunction]
pub(crate) fn profile_groupby_std_f64_firstseen_u64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByProfileReturnF64<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs_len(keys_slice.len(), values_slice.len())?;

    let profiled = py.detach(|| {
        groupby::profile_parallel_groupby_std_f64_firstseen_u64(keys_slice, values_slice)
    })?;
    let materialize_start = Instant::now();
    let (keys_1d, values_1d) = convert_single_result_f64(py, profiled.result)?;
    let profile_dict = build_single_profile_dict(
        py,
        &profiled.profile,
        materialize_start.elapsed().as_secs_f64(),
    )?;
    Ok((keys_1d, values_1d, profile_dict))
}

#[pyfunction]
pub(crate) fn groupby_min_f64_firstseen_u64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result =
        py.detach(|| groupby::parallel_groupby_min_f64_firstseen_u64(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

#[pyfunction]
pub(crate) fn groupby_max_f64_firstseen_u64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturn<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result =
        py.detach(|| groupby::parallel_groupby_max_f64_firstseen_u64(keys_slice, values_slice))?;
    convert_single_result(py, result)
}

#[pyfunction]
pub(crate) fn groupby_count_f64_firstseen_u64<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SingleGroupByReturnI64<'py>> {
    let keys_slice = zero_copy::get_slice_i64(&keys)?;
    let values_slice = zero_copy::get_slice_f64(&values)?;
    validate_inputs(keys_slice.len(), values_slice.len())?;

    let result =
        py.detach(|| groupby::parallel_groupby_count_f64_firstseen_u64(keys_slice, values_slice))?;
    convert_single_result_i64(py, result)
}
