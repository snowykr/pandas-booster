use crate::aggregation::Aggregator;

use super::dispatch::radix_groupby_dispatch;
use super::order::sort_groupby_result;
use super::result::GroupByMultiResult;
use super::sort_first::SortFirstDiagnostics;
use super::sort_first_routing::{
    choose_sort_first_route, SortFirstReducer, SortFirstRoute, SortFirstRoutingDecision,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SortedDispatchRoute {
    HashFirst,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SortedDispatchDiagnostics {
    pub route: SortedDispatchRoute,
    pub routing_decision: SortFirstRoutingDecision,
    pub hash_first_aggregation_count: usize,
    pub post_aggregation_sort_count: usize,
    pub sort_first: Option<SortFirstDiagnostics>,
}

pub(super) fn radix_groupby_sorted_with_diagnostics<T, A, O>(
    key_slices: &[&[i64]],
    values: &[T],
    reducer: SortFirstReducer,
) -> Result<(GroupByMultiResult<O>, SortedDispatchDiagnostics), String>
where
    T: Copy + Send + Sync,
    O: Copy + Send + Sync,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    let routing_decision = choose_sort_first_route(reducer, key_slices, values.len());

    match routing_decision.route {
        SortFirstRoute::HashFirst => {
            let mut result = radix_groupby_dispatch::<T, A, O>(key_slices, values)?;
            sort_groupby_result(&mut result);
            Ok((
                result,
                SortedDispatchDiagnostics {
                    route: SortedDispatchRoute::HashFirst,
                    routing_decision,
                    hash_first_aggregation_count: 1,
                    post_aggregation_sort_count: 1,
                    sort_first: None,
                },
            ))
        }
    }
}
