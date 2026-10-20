use std::time::Instant;

use crate::aggregation::Aggregator;

use super::dispatch::radix_groupby_dispatch_with_profile;
use super::order::{sort_groupby_result, sort_groupby_result_profiled};
use super::result::{GroupByMultiResult, MultiKeySortedPhaseProfile};
use super::sort_first::{production_sort_first_groupby, SortFirstDiagnostics};
use super::sort_first_routing::{
    try_choose_sort_first_route, SortFirstReducer, SortFirstRoute, SortFirstRoutingDecision,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SortedDispatchRoute {
    HashFirst,
    SortFirst,
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
    radix_groupby_sorted_with_optional_profile::<T, A, O>(key_slices, values, reducer, None)
}

// Ordinary and profiled exports must both enter here with explicit reducer identity.
pub(super) fn radix_groupby_sorted_with_optional_profile<T, A, O>(
    key_slices: &[&[i64]],
    values: &[T],
    reducer: SortFirstReducer,
    mut profile: Option<&mut MultiKeySortedPhaseProfile>,
) -> Result<(GroupByMultiResult<O>, SortedDispatchDiagnostics), String>
where
    T: Copy + Send + Sync,
    O: Copy + Send + Sync,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    let outer = Instant::now();
    let clock = Instant::now();
    let routing_decision = try_choose_sort_first_route(reducer, key_slices, values.len())?;
    let routing_s = clock.elapsed().as_secs_f64();
    let mut diagnostics = SortedDispatchDiagnostics {
        route: SortedDispatchRoute::HashFirst,
        routing_decision,
        hash_first_aggregation_count: 0,
        post_aggregation_sort_count: 0,
        sort_first: None,
    };
    if let Some(p) = profile.as_deref_mut() {
        p.routing_s = routing_s;
        p.route_reason = routing_decision.reason_label();
        p.distinct_lower_bound = routing_decision.cardinality_lower_bound;
        p.estimated_peak_bytes = routing_decision.modeled_peak_bytes;
        p.rayon_threads = rayon::current_num_threads();
        p.hash_first_aggregation_count = 0;
        p.post_aggregation_sort_count = 0;
    }
    let result = match routing_decision.route {
        SortFirstRoute::SortFirst => {
            let (result, engine_diagnostics, timings, proof) =
                production_sort_first_groupby::<T, A, O>(key_slices, values)?;
            diagnostics.route = SortedDispatchRoute::SortFirst;
            diagnostics.sort_first = Some(engine_diagnostics);
            if let Some(p) = profile.as_deref_mut() {
                p.route_label = "sort_first";
                p.sort_key_construction_s = timings.layout_s;
                p.radix_sort_s = timings.radix_s;
                p.sort_first_permutation_s = timings.layout_s + timings.radix_s;
                p.sort_first_boundary_s = timings.boundary_discovery_s;
                p.sort_first_setup_s = timings.starts_setup_s;
                p.sort_first_segment_scan_s = timings.reduction_emission_s;
                p.sort_first_segment_scan_count = engine_diagnostics.segment_scan_count;
                p.selected_sort_strategy = proof.strategy.label();
                p.sort_key_bit_widths = proof.bit_widths;
                p.final_group_count = result.values.len();
            }
            result
        }
        SortFirstRoute::HashFirst => {
            #[cfg(test)]
            super::sort_first_routing::observe_scratch("hash_fallback", 0);
            if let Some(p) = profile.as_deref_mut() {
                p.route_label = "hash_first";
                p.hash_first_aggregation_count += 1;
            }
            diagnostics.hash_first_aggregation_count += 1;
            let mut result = radix_groupby_dispatch_with_profile::<T, A, O>(
                key_slices,
                values,
                profile.as_deref_mut(),
            )?;
            // Empty output has no physical output sort.
            if !result.values.is_empty() {
                diagnostics.post_aggregation_sort_count += 1;
                if let Some(p) = profile.as_deref_mut() {
                    p.post_aggregation_sort_count += 1;
                    let sort = sort_groupby_result_profiled(&mut result);
                    p.sort_key_construction_s = sort.sort_key_construction_s;
                    p.radix_sort_s = sort.radix_sort_s;
                    p.sorted_materialization_s = sort.sorted_materialization_s;
                    p.selected_sort_strategy = sort.selected_sort_strategy;
                    p.sort_key_bit_widths = sort.sort_key_bit_widths;
                } else {
                    sort_groupby_result(&mut result);
                }
            }
            result
        }
    };
    if let Some(p) = profile {
        p.final_group_count = result.values.len();
        p.rust_elapsed_s = outer.elapsed().as_secs_f64();
    }
    Ok((result, diagnostics))
}
