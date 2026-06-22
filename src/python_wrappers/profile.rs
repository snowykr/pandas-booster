use crate::radix_groupby::MultiKeySortedPhaseProfile;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub(crate) fn build_multi_sorted_profile_dict<'py>(
    py: Python<'py>,
    profile: &MultiKeySortedPhaseProfile,
    conversion_s: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let profile_dict = PyDict::new(py);
    let rust_total_s = profile.hash_build_s
        + profile.partition_scatter_s
        + profile.partition_aggregation_s
        + profile.flatten_s
        + profile.sort_key_construction_s
        + profile.radix_sort_s
        + profile.sorted_materialization_s
        + profile.sort_first_permutation_s
        + profile.sort_first_segment_scan_s;
    let partial_to_final_ratio = if profile.final_group_count == 0 {
        0.0
    } else {
        profile.partial_group_total as f64 / profile.final_group_count as f64
    };

    profile_dict.set_item("profile_kind", "multi_key_sorted")?;
    profile_dict.set_item("route", profile.route_label)?;
    profile_dict.set_item("hash_build_s", profile.hash_build_s)?;
    profile_dict.set_item("partition_scatter_s", profile.partition_scatter_s)?;
    profile_dict.set_item("partition_aggregation_s", profile.partition_aggregation_s)?;
    profile_dict.set_item("flatten_s", profile.flatten_s)?;
    profile_dict.set_item("sort_key_construction_s", profile.sort_key_construction_s)?;
    profile_dict.set_item("radix_sort_s", profile.radix_sort_s)?;
    profile_dict.set_item("sorted_materialization_s", profile.sorted_materialization_s)?;
    profile_dict.set_item("selected_sort_strategy", profile.selected_sort_strategy)?;
    profile_dict.set_item("sort_key_bit_widths", profile.sort_key_bit_widths.clone())?;
    profile_dict.set_item("sort_first_permutation_s", profile.sort_first_permutation_s)?;
    profile_dict.set_item(
        "sort_first_segment_scan_s",
        profile.sort_first_segment_scan_s,
    )?;
    profile_dict.set_item(
        "sort_first_segment_scan_count",
        profile.sort_first_segment_scan_count,
    )?;
    profile_dict.set_item("conversion_s", conversion_s)?;
    profile_dict.set_item("pandas_index_construction_s", 0.0)?;
    profile_dict.set_item("rust_total_s", rust_total_s)?;
    profile_dict.set_item("partial_group_total", profile.partial_group_total)?;
    profile_dict.set_item("final_group_count", profile.final_group_count)?;
    profile_dict.set_item("partial_to_final_ratio", partial_to_final_ratio)?;
    Ok(profile_dict)
}
