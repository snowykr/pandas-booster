use pyo3::prelude::*;

use crate::aggregation::{Aggregator, ProdAggF64};

use super::deterministic::{
    parallel_groupby_deterministic, parallel_groupby_firstseen_deterministic_impl,
};
use super::order::{finalize_deterministic_firstseen_result, FirstSeenRowIndex};
use super::partitioned::{
    build_partitioned_deterministic_firstseen_states,
    materialize_partitioned_deterministic_firstseen_states,
};
use super::reduce::PairwiseReduceValue;
use super::result::{GroupByResult, GroupByResultF64};
use super::routing::{should_use_partitioned_median_engine, should_use_partitioned_std_var_engine};

pub(super) fn parallel_groupby_firstseen_partitioned_impl<T, A, O, I>(
    keys: &[i64],
    values: &[T],
) -> PyResult<GroupByResult<O>>
where
    T: Copy + Send + Sync,
    O: Copy,
    A: Aggregator<T, O> + Send,
    I: FirstSeenRowIndex,
{
    let states = build_partitioned_deterministic_firstseen_states::<T, O, A, I>(keys, values)?;
    let materialized = materialize_partitioned_deterministic_firstseen_states::<T, O, A, I>(states);
    Ok(finalize_deterministic_firstseen_result(materialized))
}

pub(super) fn parallel_groupby_partitioned_unordered_impl<T, A, O, I>(
    keys: &[i64],
    values: &[T],
) -> PyResult<GroupByResult<O>>
where
    T: Copy + Send + Sync,
    O: Copy,
    A: Aggregator<T, O> + Send,
    I: FirstSeenRowIndex,
{
    let states = build_partitioned_deterministic_firstseen_states::<T, O, A, I>(keys, values)?;
    let materialized = materialize_partitioned_deterministic_firstseen_states::<T, O, A, I>(states);
    Ok(materialized.result)
}

pub(super) fn parallel_groupby_firstseen_std_var_impl<T, A, O, I>(
    keys: &[i64],
    values: &[T],
) -> PyResult<GroupByResult<O>>
where
    T: Copy + Send + Sync,
    O: Copy,
    A: Aggregator<T, O> + Clone + Default + Send,
    I: FirstSeenRowIndex,
    (A, I): PairwiseReduceValue<T, O, A>,
{
    if should_use_partitioned_std_var_engine(keys) {
        parallel_groupby_firstseen_partitioned_impl::<T, A, O, I>(keys, values)
    } else {
        parallel_groupby_firstseen_deterministic_impl::<T, A, O, I>(keys, values)
    }
}

pub(super) fn parallel_groupby_std_var_impl<T, A, O>(
    keys: &[i64],
    values: &[T],
) -> PyResult<GroupByResult<O>>
where
    T: Copy + Send + Sync,
    O: Copy,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    if should_use_partitioned_std_var_engine(keys) {
        if keys.len() <= u32::MAX as usize {
            parallel_groupby_partitioned_unordered_impl::<T, A, O, u32>(keys, values)
        } else {
            parallel_groupby_partitioned_unordered_impl::<T, A, O, u64>(keys, values)
        }
    } else {
        parallel_groupby_deterministic::<T, A, O>(keys, values)
    }
}

pub(super) fn parallel_groupby_firstseen_median_impl<T, A, O, I>(
    keys: &[i64],
    values: &[T],
) -> PyResult<GroupByResult<O>>
where
    T: Copy + Send + Sync,
    O: Copy,
    A: Aggregator<T, O> + Clone + Default + Send,
    I: FirstSeenRowIndex,
    (A, I): PairwiseReduceValue<T, O, A>,
{
    if should_use_partitioned_median_engine(keys) {
        parallel_groupby_firstseen_partitioned_impl::<T, A, O, I>(keys, values)
    } else {
        parallel_groupby_firstseen_deterministic_impl::<T, A, O, I>(keys, values)
    }
}

pub(super) fn parallel_groupby_median_impl<T, A, O>(
    keys: &[i64],
    values: &[T],
) -> PyResult<GroupByResult<O>>
where
    T: Copy + Send + Sync,
    O: Copy,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    if should_use_partitioned_median_engine(keys) {
        if keys.len() <= u32::MAX as usize {
            parallel_groupby_partitioned_unordered_impl::<T, A, O, u32>(keys, values)
        } else {
            parallel_groupby_partitioned_unordered_impl::<T, A, O, u64>(keys, values)
        }
    } else {
        parallel_groupby_deterministic::<T, A, O>(keys, values)
    }
}

pub(super) fn parallel_groupby_prod_f64_ordered_impl(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    if keys.len() <= u32::MAX as usize {
        parallel_groupby_partitioned_unordered_impl::<f64, ProdAggF64, f64, u32>(keys, values)
    } else {
        parallel_groupby_partitioned_unordered_impl::<f64, ProdAggF64, f64, u64>(keys, values)
    }
}
