use ahash::AHashMap;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::time::Instant;

use crate::aggregation::Aggregator;

use super::chunks::{build_chunk_ranges, fixed_chunk_size, prepare_firstseen_deterministic_ranges};
use super::deterministic::{
    build_deterministic_firstseen_partials, combine_deterministic_firstseen_partials,
    materialize_deterministic_firstseen_result,
};
use super::order::{finalize_deterministic_firstseen_result, FirstSeenRowIndex};
use super::partitioned::{
    build_partitioned_deterministic_firstseen_states,
    materialize_partitioned_deterministic_firstseen_states, SingleKeyPartitionState,
};
use super::reduce::{reduce_partial_maps_pairwise, PairwiseReduceValue};
use super::result::{GroupByResult, ProfiledGroupByResult, SingleKeyPhaseProfile};
use super::routing::should_use_partitioned_std_var_engine;

pub(super) fn profile_parallel_groupby_firstseen_partitioned_impl<T, A, O, I>(
    keys: &[i64],
    values: &[T],
) -> PyResult<ProfiledGroupByResult<O>>
where
    T: Copy + Send + Sync,
    O: Copy,
    A: Aggregator<T, O> + Send,
    I: FirstSeenRowIndex,
{
    let local_build_start = Instant::now();
    let states = build_partitioned_deterministic_firstseen_states::<T, O, A, I>(keys, values)?;
    let local_build_s = local_build_start.elapsed().as_secs_f64();
    let partial_group_total: usize = states.iter().map(SingleKeyPartitionState::len).sum();
    let final_group_count = partial_group_total;

    let materialize_start = Instant::now();
    let materialized = materialize_partitioned_deterministic_firstseen_states::<T, O, A, I>(states);
    let materialize_s = materialize_start.elapsed().as_secs_f64();

    let reorder_start = Instant::now();
    let result = finalize_deterministic_firstseen_result(materialized);
    let reorder_s = reorder_start.elapsed().as_secs_f64();

    Ok(ProfiledGroupByResult {
        result,
        profile: SingleKeyPhaseProfile {
            local_build_s,
            merge_s: 0.0,
            reorder_s,
            materialize_s,
            partial_group_total,
            final_group_count,
        },
    })
}

pub(super) fn profile_parallel_groupby_partitioned_unordered_impl<T, A, O, I>(
    keys: &[i64],
    values: &[T],
) -> PyResult<ProfiledGroupByResult<O>>
where
    T: Copy + Send + Sync,
    O: Copy,
    A: Aggregator<T, O> + Send,
    I: FirstSeenRowIndex,
{
    let local_build_start = Instant::now();
    let states = build_partitioned_deterministic_firstseen_states::<T, O, A, I>(keys, values)?;
    let local_build_s = local_build_start.elapsed().as_secs_f64();
    let partial_group_total: usize = states.iter().map(SingleKeyPartitionState::len).sum();
    let final_group_count = partial_group_total;

    let materialize_start = Instant::now();
    let materialized = materialize_partitioned_deterministic_firstseen_states::<T, O, A, I>(states);
    let materialize_s = materialize_start.elapsed().as_secs_f64();

    Ok(ProfiledGroupByResult {
        result: materialized.result,
        profile: SingleKeyPhaseProfile {
            local_build_s,
            merge_s: 0.0,
            reorder_s: 0.0,
            materialize_s,
            partial_group_total,
            final_group_count,
        },
    })
}

pub(super) fn profile_parallel_groupby_firstseen_deterministic_impl<T, A, O, I>(
    keys: &[i64],
    values: &[T],
) -> PyResult<ProfiledGroupByResult<O>>
where
    T: Copy + Send + Sync,
    O: Copy,
    A: Aggregator<T, O> + Clone + Default + Send,
    I: FirstSeenRowIndex,
    (A, I): PairwiseReduceValue<T, O, A>,
{
    let ranges = prepare_firstseen_deterministic_ranges::<T, I>(keys, values)?;

    let local_build_start = Instant::now();
    let partials = build_deterministic_firstseen_partials::<T, O, A, I>(keys, values, &ranges);
    let local_build_s = local_build_start.elapsed().as_secs_f64();
    let partial_group_total = partials.iter().map(|partial| partial.len()).sum();

    let merge_start = Instant::now();
    let merged = combine_deterministic_firstseen_partials::<T, O, A, I>(partials);
    let merge_s = merge_start.elapsed().as_secs_f64();

    let final_group_count = merged.len();
    let materialize_start = Instant::now();
    let materialized = materialize_deterministic_firstseen_result::<T, O, A, I>(merged);
    let materialize_s = materialize_start.elapsed().as_secs_f64();

    let reorder_start = Instant::now();
    let result = finalize_deterministic_firstseen_result(materialized);
    let reorder_s = reorder_start.elapsed().as_secs_f64();

    Ok(ProfiledGroupByResult {
        result,
        profile: SingleKeyPhaseProfile {
            local_build_s,
            merge_s,
            reorder_s,
            materialize_s,
            partial_group_total,
            final_group_count,
        },
    })
}

pub(super) fn profile_parallel_groupby_firstseen_std_var_impl<T, A, O, I>(
    keys: &[i64],
    values: &[T],
) -> PyResult<ProfiledGroupByResult<O>>
where
    T: Copy + Send + Sync,
    O: Copy,
    A: Aggregator<T, O> + Clone + Default + Send,
    I: FirstSeenRowIndex,
    (A, I): PairwiseReduceValue<T, O, A>,
{
    if should_use_partitioned_std_var_engine(keys) {
        profile_parallel_groupby_firstseen_partitioned_impl::<T, A, O, I>(keys, values)
    } else {
        profile_parallel_groupby_firstseen_deterministic_impl::<T, A, O, I>(keys, values)
    }
}

pub(super) fn profile_parallel_groupby_deterministic<T, A, O>(
    keys: &[i64],
    values: &[T],
) -> PyResult<ProfiledGroupByResult<O>>
where
    T: Copy + Send + Sync,
    O: Copy,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    if keys.len() != values.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "keys and values must have same length",
        ));
    }

    let n_rows = keys.len();
    let chunk_size = fixed_chunk_size(n_rows);
    let ranges = build_chunk_ranges(n_rows, chunk_size);

    let local_build_start = Instant::now();
    let partials: Vec<AHashMap<i64, A>> = ranges
        .par_iter()
        .map(|&(start, end)| {
            let k_chunk = &keys[start..end];
            let v_chunk = &values[start..end];
            let mut acc: AHashMap<i64, A> = AHashMap::default();
            for (&key, &val) in k_chunk.iter().zip(v_chunk.iter()) {
                acc.entry(key).or_insert_with(A::init).update(val);
            }
            acc
        })
        .collect();
    let local_build_s = local_build_start.elapsed().as_secs_f64();
    let partial_group_total = partials.iter().map(|partial| partial.len()).sum();

    let merge_start = Instant::now();
    let merged = reduce_partial_maps_pairwise::<T, O, A, A>(partials);
    let merge_s = merge_start.elapsed().as_secs_f64();

    let final_group_count = merged.len();
    let materialize_start = Instant::now();
    let mut result_keys = Vec::with_capacity(final_group_count);
    let mut result_values = Vec::with_capacity(final_group_count);

    for (k, agg) in merged {
        result_keys.push(k);
        result_values.push(agg.finalize_owned());
    }
    let materialize_s = materialize_start.elapsed().as_secs_f64();

    Ok(ProfiledGroupByResult {
        result: GroupByResult {
            keys: result_keys,
            values: result_values,
        },
        profile: SingleKeyPhaseProfile {
            local_build_s,
            merge_s,
            reorder_s: 0.0,
            materialize_s,
            partial_group_total,
            final_group_count,
        },
    })
}

pub(super) fn profile_parallel_groupby_std_var_impl<T, A, O>(
    keys: &[i64],
    values: &[T],
) -> PyResult<ProfiledGroupByResult<O>>
where
    T: Copy + Send + Sync,
    O: Copy,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    if should_use_partitioned_std_var_engine(keys) {
        if keys.len() <= u32::MAX as usize {
            profile_parallel_groupby_partitioned_unordered_impl::<T, A, O, u32>(keys, values)
        } else {
            profile_parallel_groupby_partitioned_unordered_impl::<T, A, O, u64>(keys, values)
        }
    } else {
        profile_parallel_groupby_deterministic::<T, A, O>(keys, values)
    }
}
