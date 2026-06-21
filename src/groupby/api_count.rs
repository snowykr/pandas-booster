use pyo3::prelude::*;

use crate::aggregation::{CountAggF64, CountAggI64};

use super::legacy::parallel_groupby;
use super::order::reorder_single_result_by_key;
use super::result::GroupByResultI64;
use super::scalar_firstseen::{
    parallel_groupby_firstseen_legacy_low_u32, parallel_groupby_firstseen_legacy_low_u64,
};

pub fn parallel_groupby_count_f64(keys: &[i64], values: &[f64]) -> PyResult<GroupByResultI64> {
    parallel_groupby::<f64, CountAggF64, i64>(keys, values)
}

pub fn parallel_groupby_count_f64_sorted(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultI64> {
    let mut result = parallel_groupby_count_f64(keys, values)?;
    reorder_single_result_by_key(&mut result);
    Ok(result)
}

pub fn parallel_groupby_count_f64_firstseen_u32(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultI64> {
    parallel_groupby_firstseen_legacy_low_u32::<f64, CountAggF64, i64>(keys, values)
}

pub fn parallel_groupby_count_f64_firstseen_u64(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultI64> {
    parallel_groupby_firstseen_legacy_low_u64::<f64, CountAggF64, i64>(keys, values)
}

pub fn parallel_groupby_count_i64(keys: &[i64], values: &[i64]) -> PyResult<GroupByResultI64> {
    parallel_groupby::<i64, CountAggI64, i64>(keys, values)
}

pub fn parallel_groupby_count_i64_sorted(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultI64> {
    let mut result = parallel_groupby_count_i64(keys, values)?;
    reorder_single_result_by_key(&mut result);
    Ok(result)
}

pub fn parallel_groupby_count_i64_firstseen_u32(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultI64> {
    parallel_groupby_firstseen_legacy_low_u32::<i64, CountAggI64, i64>(keys, values)
}

pub fn parallel_groupby_count_i64_firstseen_u64(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultI64> {
    parallel_groupby_firstseen_legacy_low_u64::<i64, CountAggI64, i64>(keys, values)
}
