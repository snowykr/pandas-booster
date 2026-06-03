use super::*;

macro_rules! define_multi_prod_f64_wrapper {
    ($name:ident, $kernel:ident) => {
        #[pyfunction]
        pub(crate) fn $name<'py>(
            py: Python<'py>,
            key_cols: Vec<PyReadonlyArray1<'py, i64>>,
            values: PyReadonlyArray1<'py, f64>,
        ) -> PyResult<MultiGroupByReturnF64<'py>> {
            let values_slice = zero_copy::get_slice_f64(&values)?;
            let key_slices: Vec<&[i64]> = key_cols
                .iter()
                .map(|col| zero_copy::get_slice_i64(col))
                .collect::<PyResult<Vec<_>>>()?;

            let key_lengths: Vec<usize> = key_slices.iter().map(|s| s.len()).collect();
            validate_multi_inputs(&key_lengths, values_slice.len())?;

            let result = py.detach(|| groupby_multi::$kernel(&key_slices, values_slice))?;
            convert_multi_result_f64(py, result)
        }
    };
}

macro_rules! define_multi_prod_i64_wrapper {
    ($name:ident, $kernel:ident) => {
        #[pyfunction]
        pub(crate) fn $name<'py>(
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

            let result = py.detach(|| groupby_multi::$kernel(&key_slices, values_slice))?;
            convert_multi_result_i64(py, result)
        }
    };
}

define_multi_prod_f64_wrapper!(groupby_multi_prod_f64, multi_groupby_prod_f64);
define_multi_prod_f64_wrapper!(groupby_multi_prod_f64_sorted, multi_groupby_prod_f64_sorted);
define_multi_prod_f64_wrapper!(
    groupby_multi_prod_f64_firstseen_u32,
    multi_groupby_prod_f64_firstseen_u32
);
define_multi_prod_f64_wrapper!(
    groupby_multi_prod_f64_firstseen_u64,
    multi_groupby_prod_f64_firstseen_u64
);
define_multi_prod_i64_wrapper!(groupby_multi_prod_i64, multi_groupby_prod_i64);
define_multi_prod_i64_wrapper!(groupby_multi_prod_i64_sorted, multi_groupby_prod_i64_sorted);
define_multi_prod_i64_wrapper!(
    groupby_multi_prod_i64_firstseen_u32,
    multi_groupby_prod_i64_firstseen_u32
);
define_multi_prod_i64_wrapper!(
    groupby_multi_prod_i64_firstseen_u64,
    multi_groupby_prod_i64_firstseen_u64
);

macro_rules! define_multi_median_f64_wrapper {
    ($name:ident, $kernel:ident) => {
        #[pyfunction]
        pub(crate) fn $name<'py>(
            py: Python<'py>,
            key_cols: Vec<PyReadonlyArray1<'py, i64>>,
            values: PyReadonlyArray1<'py, f64>,
        ) -> PyResult<MultiGroupByReturnF64<'py>> {
            let values_slice = zero_copy::get_slice_f64(&values)?;
            let key_slices: Vec<&[i64]> = key_cols
                .iter()
                .map(|col| zero_copy::get_slice_i64(col))
                .collect::<PyResult<Vec<_>>>()?;

            let key_lengths: Vec<usize> = key_slices.iter().map(|s| s.len()).collect();
            validate_multi_inputs_len(&key_lengths, values_slice.len())?;

            let result = py.detach(|| groupby_multi::$kernel(&key_slices, values_slice))?;
            convert_multi_result_f64(py, result)
        }
    };
}

macro_rules! define_multi_median_i64_wrapper {
    ($name:ident, $kernel:ident) => {
        #[pyfunction]
        pub(crate) fn $name<'py>(
            py: Python<'py>,
            key_cols: Vec<PyReadonlyArray1<'py, i64>>,
            values: PyReadonlyArray1<'py, i64>,
        ) -> PyResult<MultiGroupByReturnF64<'py>> {
            let values_slice = zero_copy::get_slice_i64(&values)?;
            let key_slices: Vec<&[i64]> = key_cols
                .iter()
                .map(|col| zero_copy::get_slice_i64(col))
                .collect::<PyResult<Vec<_>>>()?;

            let key_lengths: Vec<usize> = key_slices.iter().map(|s| s.len()).collect();
            validate_multi_inputs_len(&key_lengths, values_slice.len())?;

            let result = py.detach(|| groupby_multi::$kernel(&key_slices, values_slice))?;
            convert_multi_result_f64(py, result)
        }
    };
}

define_multi_median_f64_wrapper!(groupby_multi_median_f64, multi_groupby_median_f64);
define_multi_median_f64_wrapper!(
    groupby_multi_median_f64_sorted,
    multi_groupby_median_f64_sorted
);
define_multi_median_f64_wrapper!(
    groupby_multi_median_f64_firstseen_u32,
    multi_groupby_median_f64_firstseen_u32
);
define_multi_median_f64_wrapper!(
    groupby_multi_median_f64_firstseen_u64,
    multi_groupby_median_f64_firstseen_u64
);
define_multi_median_i64_wrapper!(groupby_multi_median_i64, multi_groupby_median_i64);
define_multi_median_i64_wrapper!(
    groupby_multi_median_i64_sorted,
    multi_groupby_median_i64_sorted
);
define_multi_median_i64_wrapper!(
    groupby_multi_median_i64_firstseen_u32,
    multi_groupby_median_i64_firstseen_u32
);
define_multi_median_i64_wrapper!(
    groupby_multi_median_i64_firstseen_u64,
    multi_groupby_median_i64_firstseen_u64
);
