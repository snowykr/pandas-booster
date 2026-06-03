use pyo3::prelude::*;

use crate::aggregation::{
    MaxAggI64, MeanAggI64, MedianAggI64, MinAggI64, ProdAggI64, StdAggI64, SumAggI64, VarAggI64,
};

use super::engine::{
    parallel_groupby_firstseen_median_impl, parallel_groupby_firstseen_std_var_impl,
    parallel_groupby_median_impl, parallel_groupby_std_var_impl,
};
use super::legacy::{
    parallel_groupby, parallel_groupby_firstseen_u32, parallel_groupby_firstseen_u64,
};
use super::order::reorder_single_result_by_key;
use super::result::{GroupByResultF64, GroupByResultI64};

pub fn parallel_groupby_sum_i64(keys: &[i64], values: &[i64]) -> PyResult<GroupByResultI64> {
    parallel_groupby::<i64, SumAggI64, i64>(keys, values)
}

pub fn parallel_groupby_sum_i64_sorted(keys: &[i64], values: &[i64]) -> PyResult<GroupByResultI64> {
    let mut result = parallel_groupby_sum_i64(keys, values)?;
    reorder_single_result_by_key(&mut result);
    Ok(result)
}

pub fn parallel_groupby_sum_i64_firstseen_u32(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultI64> {
    parallel_groupby_firstseen_u32::<i64, SumAggI64, i64>(keys, values)
}

pub fn parallel_groupby_sum_i64_firstseen_u64(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultI64> {
    parallel_groupby_firstseen_u64::<i64, SumAggI64, i64>(keys, values)
}

pub fn parallel_groupby_prod_i64(keys: &[i64], values: &[i64]) -> PyResult<GroupByResultI64> {
    parallel_groupby::<i64, ProdAggI64, i64>(keys, values)
}

pub fn parallel_groupby_prod_i64_sorted(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultI64> {
    let mut result = parallel_groupby_prod_i64(keys, values)?;
    reorder_single_result_by_key(&mut result);
    Ok(result)
}

pub fn parallel_groupby_prod_i64_firstseen_u32(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultI64> {
    parallel_groupby_firstseen_u32::<i64, ProdAggI64, i64>(keys, values)
}

pub fn parallel_groupby_prod_i64_firstseen_u64(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultI64> {
    parallel_groupby_firstseen_u64::<i64, ProdAggI64, i64>(keys, values)
}

pub fn parallel_groupby_mean_i64(keys: &[i64], values: &[i64]) -> PyResult<GroupByResultF64> {
    parallel_groupby::<i64, MeanAggI64, f64>(keys, values)
}

pub fn parallel_groupby_mean_i64_sorted(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultF64> {
    let mut result = parallel_groupby_mean_i64(keys, values)?;
    reorder_single_result_by_key(&mut result);
    Ok(result)
}

pub fn parallel_groupby_mean_i64_firstseen_u32(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_u32::<i64, MeanAggI64, f64>(keys, values)
}

pub fn parallel_groupby_mean_i64_firstseen_u64(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_u64::<i64, MeanAggI64, f64>(keys, values)
}

pub fn parallel_groupby_median_i64(keys: &[i64], values: &[i64]) -> PyResult<GroupByResultF64> {
    parallel_groupby_median_impl::<i64, MedianAggI64, f64>(keys, values)
}

pub fn parallel_groupby_median_i64_sorted(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultF64> {
    let mut result = parallel_groupby_median_i64(keys, values)?;
    reorder_single_result_by_key(&mut result);
    Ok(result)
}

pub fn parallel_groupby_median_i64_firstseen_u32(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_median_impl::<i64, MedianAggI64, f64, u32>(keys, values)
}

pub fn parallel_groupby_median_i64_firstseen_u64(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_median_impl::<i64, MedianAggI64, f64, u64>(keys, values)
}

pub fn parallel_groupby_var_i64(keys: &[i64], values: &[i64]) -> PyResult<GroupByResultF64> {
    parallel_groupby_std_var_impl::<i64, VarAggI64, f64>(keys, values)
}

pub fn parallel_groupby_var_i64_sorted(keys: &[i64], values: &[i64]) -> PyResult<GroupByResultF64> {
    let mut result = parallel_groupby_var_i64(keys, values)?;
    reorder_single_result_by_key(&mut result);
    Ok(result)
}

pub fn parallel_groupby_var_i64_firstseen_u32(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_std_var_impl::<i64, VarAggI64, f64, u32>(keys, values)
}

pub fn parallel_groupby_var_i64_firstseen_u64(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_std_var_impl::<i64, VarAggI64, f64, u64>(keys, values)
}

pub fn parallel_groupby_std_i64(keys: &[i64], values: &[i64]) -> PyResult<GroupByResultF64> {
    parallel_groupby_std_var_impl::<i64, StdAggI64, f64>(keys, values)
}

pub fn parallel_groupby_std_i64_sorted(keys: &[i64], values: &[i64]) -> PyResult<GroupByResultF64> {
    let mut result = parallel_groupby_std_i64(keys, values)?;
    reorder_single_result_by_key(&mut result);
    Ok(result)
}

pub fn parallel_groupby_std_i64_firstseen_u32(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_std_var_impl::<i64, StdAggI64, f64, u32>(keys, values)
}

pub fn parallel_groupby_std_i64_firstseen_u64(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_std_var_impl::<i64, StdAggI64, f64, u64>(keys, values)
}

pub fn parallel_groupby_min_i64(keys: &[i64], values: &[i64]) -> PyResult<GroupByResultI64> {
    parallel_groupby::<i64, MinAggI64, i64>(keys, values)
}

pub fn parallel_groupby_min_i64_sorted(keys: &[i64], values: &[i64]) -> PyResult<GroupByResultI64> {
    let mut result = parallel_groupby_min_i64(keys, values)?;
    reorder_single_result_by_key(&mut result);
    Ok(result)
}

pub fn parallel_groupby_min_i64_firstseen_u32(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultI64> {
    parallel_groupby_firstseen_u32::<i64, MinAggI64, i64>(keys, values)
}

pub fn parallel_groupby_min_i64_firstseen_u64(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultI64> {
    parallel_groupby_firstseen_u64::<i64, MinAggI64, i64>(keys, values)
}

pub fn parallel_groupby_max_i64(keys: &[i64], values: &[i64]) -> PyResult<GroupByResultI64> {
    parallel_groupby::<i64, MaxAggI64, i64>(keys, values)
}

pub fn parallel_groupby_max_i64_sorted(keys: &[i64], values: &[i64]) -> PyResult<GroupByResultI64> {
    let mut result = parallel_groupby_max_i64(keys, values)?;
    reorder_single_result_by_key(&mut result);
    Ok(result)
}

pub fn parallel_groupby_max_i64_firstseen_u32(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultI64> {
    parallel_groupby_firstseen_u32::<i64, MaxAggI64, i64>(keys, values)
}

pub fn parallel_groupby_max_i64_firstseen_u64(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultI64> {
    parallel_groupby_firstseen_u64::<i64, MaxAggI64, i64>(keys, values)
}
