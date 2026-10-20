use crate::aggregation::Aggregator;

use super::dispatch::radix_groupby_dispatch_with_profile;
use super::order::sort_groupby_result_profiled;
use super::result::{MultiKeySortedPhaseProfile, ProfiledGroupByMultiResult};

pub(super) fn profile_radix_groupby_sorted<T, A, O>(
    key_slices: &[&[i64]],
    values: &[T],
) -> Result<ProfiledGroupByMultiResult<O>, String>
where
    T: Copy + Send + Sync,
    O: Copy + Send + Sync,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    let mut profile = MultiKeySortedPhaseProfile {
        route_label: "hash_first",
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
    let mut result =
        radix_groupby_dispatch_with_profile::<T, A, O>(key_slices, values, Some(&mut profile))?;
    if !values.is_empty() {
        let sort_profile = sort_groupby_result_profiled(&mut result);
        profile.sort_key_construction_s = sort_profile.sort_key_construction_s;
        profile.radix_sort_s = sort_profile.radix_sort_s;
        profile.sorted_materialization_s = sort_profile.sorted_materialization_s;
        profile.selected_sort_strategy = sort_profile.selected_sort_strategy;
        profile.sort_key_bit_widths = sort_profile.sort_key_bit_widths;
        profile.final_group_count = result.values.len();
    }
    Ok(ProfiledGroupByMultiResult { result, profile })
}
