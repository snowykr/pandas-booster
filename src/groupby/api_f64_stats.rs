use pyo3::prelude::*;
use std::time::Instant;

use crate::aggregation::{MaxAggF64, MedianAggF64, MinAggF64, StdAggF64, VarAggF64};

use super::engine::parallel_groupby_firstseen_std_var_impl;
use super::legacy::parallel_groupby;
use super::order::reorder_single_result_by_key;
use super::profile::{
    profile_parallel_groupby_deterministic, profile_parallel_groupby_firstseen_std_var_impl,
    profile_parallel_groupby_partitioned_unordered_impl, profile_parallel_groupby_std_var_impl,
};
use super::result::{GroupByResultF64, ProfiledGroupByResult, SingleKeyPhaseProfile};
use super::routing::{sorted_median_route_decision, SortedMedianRouteKind, SortedMedianValueKind};
use super::scalar_firstseen::{
    parallel_groupby_firstseen_legacy_low_u32, parallel_groupby_firstseen_legacy_low_u64,
};
use super::sorted_median::groupby_median_f64_sorted_with_decision_and_stats;

fn reorder_profiled_result(
    mut profiled: ProfiledGroupByResult<f64>,
) -> PyResult<ProfiledGroupByResult<f64>> {
    let reorder_start = Instant::now();
    reorder_single_result_by_key(&mut profiled.result);
    profiled.profile.reorder_s += reorder_start.elapsed().as_secs_f64();
    Ok(profiled)
}

pub fn parallel_groupby_var_f64(keys: &[i64], values: &[f64]) -> PyResult<GroupByResultF64> {
    super::engine::parallel_groupby_std_var_impl::<f64, VarAggF64, f64>(keys, values)
}

pub fn parallel_groupby_var_f64_sorted(keys: &[i64], values: &[f64]) -> PyResult<GroupByResultF64> {
    let mut result = parallel_groupby_var_f64(keys, values)?;
    reorder_single_result_by_key(&mut result);
    Ok(result)
}

pub fn profile_parallel_groupby_var_f64_sorted(
    keys: &[i64],
    values: &[f64],
) -> PyResult<ProfiledGroupByResult<f64>> {
    reorder_profiled_result(
        profile_parallel_groupby_std_var_impl::<f64, VarAggF64, f64>(keys, values)?,
    )
}

pub fn parallel_groupby_var_f64_firstseen_u32(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_std_var_impl::<f64, VarAggF64, f64, u32>(keys, values)
}

pub fn profile_parallel_groupby_var_f64_firstseen_u32(
    keys: &[i64],
    values: &[f64],
) -> PyResult<ProfiledGroupByResult<f64>> {
    profile_parallel_groupby_firstseen_std_var_impl::<f64, VarAggF64, f64, u32>(keys, values)
}

pub fn parallel_groupby_var_f64_firstseen_u64(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_std_var_impl::<f64, VarAggF64, f64, u64>(keys, values)
}

pub fn profile_parallel_groupby_var_f64_firstseen_u64(
    keys: &[i64],
    values: &[f64],
) -> PyResult<ProfiledGroupByResult<f64>> {
    profile_parallel_groupby_firstseen_std_var_impl::<f64, VarAggF64, f64, u64>(keys, values)
}

pub fn parallel_groupby_std_f64(keys: &[i64], values: &[f64]) -> PyResult<GroupByResultF64> {
    super::engine::parallel_groupby_std_var_impl::<f64, StdAggF64, f64>(keys, values)
}

pub fn parallel_groupby_std_f64_sorted(keys: &[i64], values: &[f64]) -> PyResult<GroupByResultF64> {
    let mut result = parallel_groupby_std_f64(keys, values)?;
    reorder_single_result_by_key(&mut result);
    Ok(result)
}

pub fn profile_parallel_groupby_std_f64_sorted(
    keys: &[i64],
    values: &[f64],
) -> PyResult<ProfiledGroupByResult<f64>> {
    reorder_profiled_result(
        profile_parallel_groupby_std_var_impl::<f64, StdAggF64, f64>(keys, values)?,
    )
}

pub fn parallel_groupby_std_f64_firstseen_u32(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_std_var_impl::<f64, StdAggF64, f64, u32>(keys, values)
}

pub fn profile_parallel_groupby_std_f64_firstseen_u32(
    keys: &[i64],
    values: &[f64],
) -> PyResult<ProfiledGroupByResult<f64>> {
    profile_parallel_groupby_firstseen_std_var_impl::<f64, StdAggF64, f64, u32>(keys, values)
}

pub fn parallel_groupby_std_f64_firstseen_u64(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_std_var_impl::<f64, StdAggF64, f64, u64>(keys, values)
}

pub fn profile_parallel_groupby_std_f64_firstseen_u64(
    keys: &[i64],
    values: &[f64],
) -> PyResult<ProfiledGroupByResult<f64>> {
    profile_parallel_groupby_firstseen_std_var_impl::<f64, StdAggF64, f64, u64>(keys, values)
}

pub fn profile_parallel_groupby_median_f64_sorted(
    keys: &[i64],
    values: &[f64],
) -> PyResult<ProfiledGroupByResult<f64>> {
    let decision = sorted_median_route_decision(keys, SortedMedianValueKind::F64);
    if matches!(
        decision.kind,
        SortedMedianRouteKind::DirectDense | SortedMedianRouteKind::DirectSparse
    ) {
        let profiled = groupby_median_f64_sorted_with_decision_and_stats(keys, values, &decision)?;

        return Ok(ProfiledGroupByResult {
            result: profiled.result,
            profile: SingleKeyPhaseProfile {
                local_build_s: 0.0,
                merge_s: 0.0,
                reorder_s: 0.0,
                materialize_s: 0.0,
                unique_build_s: profiled.stats.unique_build_s,
                key_sort_s: profiled.stats.key_sort_s,
                count_s: profiled.stats.count_s,
                buffer_setup_s: profiled.stats.buffer_setup_s,
                scatter_s: profiled.stats.scatter_s,
                median_select_s: profiled.stats.median_select_s,
                route_kind: Some(decision.kind.as_str()),
                route_reason: Some(decision.reason.as_str()),
                partial_group_total: profiled.stats.partial_group_total,
                final_group_count: profiled.stats.final_group_count,
            },
        });
    }

    let mut profiled = if matches!(decision.kind, SortedMedianRouteKind::PartitionedFallback) {
        if keys.len() <= u32::MAX as usize {
            reorder_profiled_result(profile_parallel_groupby_partitioned_unordered_impl::<
                f64,
                MedianAggF64,
                f64,
                u32,
            >(keys, values)?)?
        } else {
            reorder_profiled_result(profile_parallel_groupby_partitioned_unordered_impl::<
                f64,
                MedianAggF64,
                f64,
                u64,
            >(keys, values)?)?
        }
    } else {
        reorder_profiled_result(profile_parallel_groupby_deterministic::<
            f64,
            MedianAggF64,
            f64,
        >(keys, values)?)?
    };
    profiled.profile.route_kind = Some(decision.kind.as_str());
    profiled.profile.route_reason = Some(decision.reason.as_str());
    Ok(profiled)
}

pub fn parallel_groupby_min_f64(keys: &[i64], values: &[f64]) -> PyResult<GroupByResultF64> {
    parallel_groupby::<f64, MinAggF64, f64>(keys, values)
}

pub fn parallel_groupby_min_f64_sorted(keys: &[i64], values: &[f64]) -> PyResult<GroupByResultF64> {
    let mut result = parallel_groupby_min_f64(keys, values)?;
    reorder_single_result_by_key(&mut result);
    Ok(result)
}

pub fn parallel_groupby_min_f64_firstseen_u32(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_legacy_low_u32::<f64, MinAggF64, f64>(keys, values)
}

pub fn parallel_groupby_min_f64_firstseen_u64(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_legacy_low_u64::<f64, MinAggF64, f64>(keys, values)
}

pub fn parallel_groupby_max_f64(keys: &[i64], values: &[f64]) -> PyResult<GroupByResultF64> {
    parallel_groupby::<f64, MaxAggF64, f64>(keys, values)
}

pub fn parallel_groupby_max_f64_sorted(keys: &[i64], values: &[f64]) -> PyResult<GroupByResultF64> {
    let mut result = parallel_groupby_max_f64(keys, values)?;
    reorder_single_result_by_key(&mut result);
    Ok(result)
}

pub fn parallel_groupby_max_f64_firstseen_u32(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_legacy_low_u32::<f64, MaxAggF64, f64>(keys, values)
}

pub fn parallel_groupby_max_f64_firstseen_u64(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_legacy_low_u64::<f64, MaxAggF64, f64>(keys, values)
}
