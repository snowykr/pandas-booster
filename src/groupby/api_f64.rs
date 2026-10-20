use pyo3::prelude::*;

use crate::aggregation::{MeanAggF64, MedianAggF64, SumAggF64};

use super::deterministic::parallel_groupby_deterministic;
use super::engine::{
    parallel_groupby_firstseen_median_impl, parallel_groupby_median_impl,
    parallel_groupby_partitioned_unordered_impl, parallel_groupby_prod_f64_firstseen_impl,
    parallel_groupby_prod_f64_ordered_impl,
};
use super::order::reorder_single_result_by_key;
use super::result::GroupByResultF64;
use super::routing::{sorted_median_route_decision, SortedMedianRouteKind, SortedMedianValueKind};
use super::scalar_firstseen::{
    parallel_groupby_firstseen_deterministic_low_u32,
    parallel_groupby_firstseen_deterministic_low_u64,
};
use super::sorted_median::groupby_median_f64_sorted_with_decision;

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
    parallel_groupby_prod_f64_firstseen_impl::<u32>(keys, values)
}

pub fn parallel_groupby_prod_f64_firstseen_u64(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_prod_f64_firstseen_impl::<u64>(keys, values)
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
    let decision = sorted_median_route_decision(keys, SortedMedianValueKind::F64);
    match decision.kind {
        SortedMedianRouteKind::DirectDense | SortedMedianRouteKind::DirectSparse => {
            groupby_median_f64_sorted_with_decision(keys, values, &decision)
        }
        SortedMedianRouteKind::PartitionedFallback => {
            let mut result = if keys.len() <= u32::MAX as usize {
                parallel_groupby_partitioned_unordered_impl::<f64, MedianAggF64, f64, u32>(
                    keys, values,
                )?
            } else {
                parallel_groupby_partitioned_unordered_impl::<f64, MedianAggF64, f64, u64>(
                    keys, values,
                )?
            };
            reorder_single_result_by_key(&mut result);
            Ok(result)
        }
        SortedMedianRouteKind::ExistingSortedFallback => {
            let mut result =
                parallel_groupby_deterministic::<f64, MedianAggF64, f64>(keys, values)?;
            reorder_single_result_by_key(&mut result);
            Ok(result)
        }
    }
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
