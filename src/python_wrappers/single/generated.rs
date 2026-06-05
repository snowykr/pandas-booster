use super::*;

macro_rules! define_single_prod_f64_wrapper {
    ($name:ident, $kernel:ident) => {
        #[pyfunction]
        pub(crate) fn $name<'py>(
            py: Python<'py>,
            keys: PyReadonlyArray1<'py, i64>,
            values: PyReadonlyArray1<'py, f64>,
        ) -> PyResult<SingleGroupByReturnF64<'py>> {
            let keys_slice = zero_copy::get_slice_i64(&keys)?;
            let values_slice = zero_copy::get_slice_f64(&values)?;
            validate_inputs(keys_slice.len(), values_slice.len())?;

            let result = py.detach(|| groupby::$kernel(keys_slice, values_slice))?;
            convert_single_result_f64(py, result)
        }
    };
}

macro_rules! define_single_prod_i64_wrapper {
    ($name:ident, $kernel:ident) => {
        #[pyfunction]
        pub(crate) fn $name<'py>(
            py: Python<'py>,
            keys: PyReadonlyArray1<'py, i64>,
            values: PyReadonlyArray1<'py, i64>,
        ) -> PyResult<SingleGroupByReturnI64<'py>> {
            let keys_slice = zero_copy::get_slice_i64(&keys)?;
            let values_slice = zero_copy::get_slice_i64(&values)?;
            validate_inputs(keys_slice.len(), values_slice.len())?;

            let result = py.detach(|| groupby::$kernel(keys_slice, values_slice))?;
            convert_single_result_i64(py, result)
        }
    };
}

define_single_prod_f64_wrapper!(groupby_prod_f64, parallel_groupby_prod_f64);
define_single_prod_f64_wrapper!(groupby_prod_f64_sorted, parallel_groupby_prod_f64_sorted);
define_single_prod_f64_wrapper!(
    groupby_prod_f64_firstseen_u32,
    parallel_groupby_prod_f64_firstseen_u32
);
define_single_prod_f64_wrapper!(
    groupby_prod_f64_firstseen_u64,
    parallel_groupby_prod_f64_firstseen_u64
);
define_single_prod_i64_wrapper!(groupby_prod_i64, parallel_groupby_prod_i64);
define_single_prod_i64_wrapper!(groupby_prod_i64_sorted, parallel_groupby_prod_i64_sorted);
define_single_prod_i64_wrapper!(
    groupby_prod_i64_firstseen_u32,
    parallel_groupby_prod_i64_firstseen_u32
);
define_single_prod_i64_wrapper!(
    groupby_prod_i64_firstseen_u64,
    parallel_groupby_prod_i64_firstseen_u64
);

macro_rules! define_single_median_f64_wrapper {
    ($name:ident, $kernel:ident) => {
        #[pyfunction]
        pub(crate) fn $name<'py>(
            py: Python<'py>,
            keys: PyReadonlyArray1<'py, i64>,
            values: PyReadonlyArray1<'py, f64>,
        ) -> PyResult<SingleGroupByReturnF64<'py>> {
            let keys_slice = zero_copy::get_slice_i64(&keys)?;
            let values_slice = zero_copy::get_slice_f64(&values)?;
            validate_inputs_len(keys_slice.len(), values_slice.len())?;

            let result = py.detach(|| groupby::$kernel(keys_slice, values_slice))?;
            convert_single_result_f64(py, result)
        }
    };
}

macro_rules! define_single_median_i64_wrapper {
    ($name:ident, $kernel:ident) => {
        #[pyfunction]
        pub(crate) fn $name<'py>(
            py: Python<'py>,
            keys: PyReadonlyArray1<'py, i64>,
            values: PyReadonlyArray1<'py, i64>,
        ) -> PyResult<SingleGroupByReturnF64<'py>> {
            let keys_slice = zero_copy::get_slice_i64(&keys)?;
            let values_slice = zero_copy::get_slice_i64(&values)?;
            validate_inputs_len(keys_slice.len(), values_slice.len())?;

            let result = py.detach(|| groupby::$kernel(keys_slice, values_slice))?;
            convert_single_result_f64(py, result)
        }
    };
}

define_single_median_f64_wrapper!(groupby_median_f64, parallel_groupby_median_f64);
define_single_median_f64_wrapper!(
    groupby_median_f64_sorted,
    parallel_groupby_median_f64_sorted
);
define_single_median_f64_wrapper!(
    groupby_median_f64_firstseen_u32,
    parallel_groupby_median_f64_firstseen_u32
);
define_single_median_f64_wrapper!(
    groupby_median_f64_firstseen_u64,
    parallel_groupby_median_f64_firstseen_u64
);
define_single_median_i64_wrapper!(groupby_median_i64, parallel_groupby_median_i64);
define_single_median_i64_wrapper!(
    groupby_median_i64_sorted,
    parallel_groupby_median_i64_sorted
);
define_single_median_i64_wrapper!(
    groupby_median_i64_firstseen_u32,
    parallel_groupby_median_i64_firstseen_u32
);
define_single_median_i64_wrapper!(
    groupby_median_i64_firstseen_u64,
    parallel_groupby_median_i64_firstseen_u64
);
