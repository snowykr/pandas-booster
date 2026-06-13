use pyo3::prelude::*;

use crate::aggregation::Aggregator;

use super::deterministic::{
    parallel_groupby_firstseen_u32_deterministic, parallel_groupby_firstseen_u64_deterministic,
};
use super::engine::parallel_groupby_firstseen_partitioned_impl;
use super::legacy::{parallel_groupby_firstseen_u32, parallel_groupby_firstseen_u64};
use super::order::FirstSeenRowIndex;
use super::reduce::PairwiseReduceValue;
use super::result::GroupByResult;
use super::routing::should_use_partitioned_firstseen_engine;

#[cfg(test)]
thread_local! {
    static LAST_SCALAR_FIRSTSEEN_ROUTE: std::cell::Cell<Option<ScalarFirstseenRoute>> =
        const { std::cell::Cell::new(None) };
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ScalarFirstseenRoute {
    DeterministicLow,
    LegacyLow,
    #[cfg(test)]
    Partitioned,
}

#[cfg(test)]
pub(super) fn clear_scalar_firstseen_route_for_test() {
    LAST_SCALAR_FIRSTSEEN_ROUTE.with(|route| route.set(None));
}

#[cfg(test)]
fn record_scalar_firstseen_route_for_test(route: ScalarFirstseenRoute) {
    LAST_SCALAR_FIRSTSEEN_ROUTE.with(|last_route| last_route.set(Some(route)));
}

#[cfg(test)]
pub(super) fn take_scalar_firstseen_route_for_test() -> Option<ScalarFirstseenRoute> {
    LAST_SCALAR_FIRSTSEEN_ROUTE.with(|last_route| {
        let route = last_route.get();
        last_route.set(None);
        route
    })
}

fn route_firstseen<T, A, O, I>(
    keys: &[i64],
    values: &[T],
    #[cfg_attr(not(test), allow(unused_variables))] low_route: ScalarFirstseenRoute,
    low_engine: impl FnOnce(&[i64], &[T]) -> PyResult<GroupByResult<O>>,
) -> PyResult<GroupByResult<O>>
where
    T: Copy + Send + Sync,
    O: Copy,
    A: Aggregator<T, O> + Send,
    I: FirstSeenRowIndex,
{
    if should_use_partitioned_firstseen_engine(keys) {
        #[cfg(test)]
        record_scalar_firstseen_route_for_test(ScalarFirstseenRoute::Partitioned);
        parallel_groupby_firstseen_partitioned_impl::<T, A, O, I>(keys, values)
    } else {
        #[cfg(test)]
        record_scalar_firstseen_route_for_test(low_route);
        low_engine(keys, values)
    }
}

pub(super) fn parallel_groupby_firstseen_deterministic_low_u32<T, A, O>(
    keys: &[i64],
    values: &[T],
) -> PyResult<GroupByResult<O>>
where
    T: Copy + Send + Sync,
    O: Copy,
    A: Aggregator<T, O> + Clone + Default + Send,
    (A, u32): PairwiseReduceValue<T, O, A>,
{
    route_firstseen::<T, A, O, u32>(
        keys,
        values,
        ScalarFirstseenRoute::DeterministicLow,
        parallel_groupby_firstseen_u32_deterministic::<T, A, O>,
    )
}

pub(super) fn parallel_groupby_firstseen_deterministic_low_u64<T, A, O>(
    keys: &[i64],
    values: &[T],
) -> PyResult<GroupByResult<O>>
where
    T: Copy + Send + Sync,
    O: Copy,
    A: Aggregator<T, O> + Clone + Default + Send,
    (A, u64): PairwiseReduceValue<T, O, A>,
{
    route_firstseen::<T, A, O, u64>(
        keys,
        values,
        ScalarFirstseenRoute::DeterministicLow,
        parallel_groupby_firstseen_u64_deterministic::<T, A, O>,
    )
}

pub(super) fn parallel_groupby_firstseen_legacy_low_u32<T, A, O>(
    keys: &[i64],
    values: &[T],
) -> PyResult<GroupByResult<O>>
where
    T: Copy + Send + Sync,
    O: Copy,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    route_firstseen::<T, A, O, u32>(
        keys,
        values,
        ScalarFirstseenRoute::LegacyLow,
        parallel_groupby_firstseen_u32::<T, A, O>,
    )
}

pub(super) fn parallel_groupby_firstseen_legacy_low_u64<T, A, O>(
    keys: &[i64],
    values: &[T],
) -> PyResult<GroupByResult<O>>
where
    T: Copy + Send + Sync,
    O: Copy,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    route_firstseen::<T, A, O, u64>(
        keys,
        values,
        ScalarFirstseenRoute::LegacyLow,
        parallel_groupby_firstseen_u64::<T, A, O>,
    )
}
