use pyo3::prelude::*;

use crate::aggregation::{MeanAggF64, MedianAggF64, ProdAggF64, SumAggF64};

use super::deterministic::parallel_groupby_deterministic;
use super::engine::{
    parallel_groupby_firstseen_median_impl, parallel_groupby_firstseen_partitioned_impl,
    parallel_groupby_median_impl, parallel_groupby_prod_f64_ordered_impl,
};
use super::order::reorder_single_result_by_key;
use super::result::GroupByResultF64;
use super::scalar_firstseen::{
    parallel_groupby_firstseen_deterministic_low_u32,
    parallel_groupby_firstseen_deterministic_low_u64,
};

pub fn parallel_groupby_sum_f64(keys: &[i64], values: &[f64]) -> PyResult<GroupByResultF64> {
    parallel_groupby_deterministic::<f64, SumAggF64, f64>(keys, values)
}

pub fn parallel_groupby_sum_f64_sorted(keys: &[i64], values: &[f64]) -> PyResult<GroupByResultF64> {
    let mut result = parallel_groupby_sum_f64(keys, values)?;
    reorder_single_result_by_key(&mut result);
    Ok(result)
}

pub fn parallel_groupby_sum_f64_firstseen_u32(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_deterministic_low_u32::<f64, SumAggF64, f64>(keys, values)
}

pub fn parallel_groupby_sum_f64_firstseen_u64(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_deterministic_low_u64::<f64, SumAggF64, f64>(keys, values)
}

pub fn parallel_groupby_prod_f64(keys: &[i64], values: &[f64]) -> PyResult<GroupByResultF64> {
    parallel_groupby_prod_f64_ordered_impl(keys, values)
}

pub fn parallel_groupby_prod_f64_sorted(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    let mut result = parallel_groupby_prod_f64(keys, values)?;
    reorder_single_result_by_key(&mut result);
    Ok(result)
}

pub fn parallel_groupby_prod_f64_firstseen_u32(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_partitioned_impl::<f64, ProdAggF64, f64, u32>(keys, values)
}

pub fn parallel_groupby_prod_f64_firstseen_u64(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_partitioned_impl::<f64, ProdAggF64, f64, u64>(keys, values)
}

pub fn parallel_groupby_mean_f64(keys: &[i64], values: &[f64]) -> PyResult<GroupByResultF64> {
    parallel_groupby_deterministic::<f64, MeanAggF64, f64>(keys, values)
}

pub fn parallel_groupby_mean_f64_sorted(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    let mut result = parallel_groupby_mean_f64(keys, values)?;
    reorder_single_result_by_key(&mut result);
    Ok(result)
}

pub fn parallel_groupby_mean_f64_firstseen_u32(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_deterministic_low_u32::<f64, MeanAggF64, f64>(keys, values)
}

pub fn parallel_groupby_mean_f64_firstseen_u64(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_deterministic_low_u64::<f64, MeanAggF64, f64>(keys, values)
}

pub fn parallel_groupby_median_f64(keys: &[i64], values: &[f64]) -> PyResult<GroupByResultF64> {
    parallel_groupby_median_impl::<f64, MedianAggF64, f64>(keys, values)
}

pub fn parallel_groupby_median_f64_sorted(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    let mut result = parallel_groupby_median_f64(keys, values)?;
    reorder_single_result_by_key(&mut result);
    Ok(result)
}

pub fn parallel_groupby_median_f64_firstseen_u32(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_median_impl::<f64, MedianAggF64, f64, u32>(keys, values)
}

pub fn parallel_groupby_median_f64_firstseen_u64(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_median_impl::<f64, MedianAggF64, f64, u64>(keys, values)
}
