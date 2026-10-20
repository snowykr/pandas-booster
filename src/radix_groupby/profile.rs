use crate::aggregation::Aggregator;

use super::dispatch_sorted::radix_groupby_sorted_with_optional_profile;
use super::result::{MultiKeySortedPhaseProfile, ProfiledGroupByMultiResult};
use super::sort_first_routing::SortFirstReducer;

pub(super) fn profile_radix_groupby_sorted<T, A, O>(
    key_slices: &[&[i64]],
    values: &[T],
    reducer: SortFirstReducer,
) -> Result<ProfiledGroupByMultiResult<O>, String>
where
    T: Copy + Send + Sync,
    O: Copy + Send + Sync,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    let mut profile = MultiKeySortedPhaseProfile {
        route_label: "hash_first",
        route_reason: "",
        routing_s: 0.0,
        sort_first_boundary_s: 0.0,
        sort_first_setup_s: 0.0,
        hash_first_aggregation_count: 0,
        post_aggregation_sort_count: 0,
        distinct_lower_bound: 0,
        estimated_peak_bytes: 0,
        rayon_threads: 0,
        rust_elapsed_s: 0.0,
        hash_build_s: 0.0,
        partition_scatter_s: 0.0,
        partition_aggregation_s: 0.0,
        flatten_s: 0.0,
        sort_key_construction_s: 0.0,
        radix_sort_s: 0.0,
        sorted_materialization_s: 0.0,
        selected_sort_strategy: "empty",
        sort_key_bit_widths: Vec::new(),
        sort_first_permutation_s: 0.0,
        sort_first_segment_scan_s: 0.0,
        sort_first_segment_scan_count: 0,
        partial_group_total: 0,
        final_group_count: 0,
    };
    let (result, _) = radix_groupby_sorted_with_optional_profile::<T, A, O>(
        key_slices,
        values,
        reducer,
        Some(&mut profile),
    )?;
    Ok(ProfiledGroupByMultiResult { result, profile })
}
