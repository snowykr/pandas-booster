use ahash::AHashMap;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::hash_map::Entry;

use crate::aggregation::Aggregator;

use super::chunks::{build_chunk_ranges, fixed_chunk_size, prepare_firstseen_deterministic_ranges};
use super::order::{
    finalize_deterministic_firstseen_result, FirstSeenMaterializedResult, FirstSeenRowIndex,
};
use super::reduce::{reduce_partial_maps_pairwise, PairwiseReduceValue};
use super::result::GroupByResult;

fn update_firstseen_local_accumulator<T, O, A, I>(
    acc: &mut AHashMap<i64, (A, I)>,
    key: i64,
    value: T,
    row: I,
) where
    A: Aggregator<T, O>,
    I: FirstSeenRowIndex,
{
    match acc.entry(key) {
        Entry::Occupied(mut e) => {
            let (agg, first) = e.get_mut();
            if row < *first {
                *first = row;
            }
            agg.update(value);
        }
        Entry::Vacant(e) => {
            let mut agg = A::init();
            agg.update(value);
            e.insert((agg, row));
        }
    }
}

fn build_firstseen_chunk_partial<T, O, A, I>(
    keys: &[i64],
    values: &[T],
    start_row: usize,
) -> AHashMap<i64, (A, I)>
where
    T: Copy,
    A: Aggregator<T, O>,
    I: FirstSeenRowIndex,
{
    let mut acc: AHashMap<i64, (A, I)> = AHashMap::default();

    for (offset, (&key, &value)) in keys.iter().zip(values.iter()).enumerate() {
        let row = I::from_row_index(start_row + offset);
        update_firstseen_local_accumulator::<T, O, A, I>(&mut acc, key, value, row);
    }

    acc
}

pub(super) fn build_deterministic_firstseen_partials<T, O, A, I>(
    keys: &[i64],
    values: &[T],
    ranges: &[(usize, usize)],
) -> Vec<AHashMap<i64, (A, I)>>
where
    T: Copy + Send + Sync,
    A: Aggregator<T, O> + Send,
    I: FirstSeenRowIndex,
{
    ranges
        .par_iter()
        .map(|&(start, end)| {
            build_firstseen_chunk_partial::<T, O, A, I>(
                &keys[start..end],
                &values[start..end],
                start,
            )
        })
        .collect()
}

pub(super) fn combine_deterministic_firstseen_partials<T, O, A, I>(
    partials: Vec<AHashMap<i64, (A, I)>>,
) -> AHashMap<i64, (A, I)>
where
    T: Copy + Send + Sync,
    O: Copy,
    A: Aggregator<T, O> + Send,
    I: FirstSeenRowIndex,
    (A, I): PairwiseReduceValue<T, O, A>,
{
    reduce_partial_maps_pairwise::<T, O, A, (A, I)>(partials)
}

pub(super) fn materialize_deterministic_firstseen_result<T, O, A, I>(
    merged: AHashMap<i64, (A, I)>,
) -> FirstSeenMaterializedResult<I, O>
where
    O: Copy,
    A: Aggregator<T, O>,
    I: FirstSeenRowIndex,
{
    let mut result_keys = Vec::with_capacity(merged.len());
    let mut result_values = Vec::with_capacity(merged.len());
    let mut first_seen = Vec::with_capacity(merged.len());

    for (key, (agg, first)) in merged {
        result_keys.push(key);
        result_values.push(agg.finalize_owned());
        first_seen.push(first);
    }

    FirstSeenMaterializedResult {
        result: GroupByResult {
            keys: result_keys,
            values: result_values,
        },
        first_seen,
    }
}

pub(super) fn parallel_groupby_firstseen_deterministic_impl<T, A, O, I>(
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
    let ranges = prepare_firstseen_deterministic_ranges::<T, I>(keys, values)?;
    let partials = build_deterministic_firstseen_partials::<T, O, A, I>(keys, values, &ranges);
    let merged = combine_deterministic_firstseen_partials::<T, O, A, I>(partials);
    let materialized = materialize_deterministic_firstseen_result::<T, O, A, I>(merged);
    Ok(finalize_deterministic_firstseen_result(materialized))
}

pub(super) fn parallel_groupby_deterministic<T, A, O>(
    keys: &[i64],
    values: &[T],
) -> PyResult<GroupByResult<O>>
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

    let merged = reduce_partial_maps_pairwise::<T, O, A, A>(partials);

    let mut result_keys = Vec::with_capacity(merged.len());
    let mut result_values = Vec::with_capacity(merged.len());

    for (k, agg) in merged {
        result_keys.push(k);
        result_values.push(agg.finalize_owned());
    }

    Ok(GroupByResult {
        keys: result_keys,
        values: result_values,
    })
}

pub(super) fn parallel_groupby_firstseen_u32_deterministic<T, A, O>(
    keys: &[i64],
    values: &[T],
) -> PyResult<GroupByResult<O>>
where
    T: Copy + Send + Sync,
    O: Copy,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    parallel_groupby_firstseen_deterministic_impl::<T, A, O, u32>(keys, values)
}

pub(super) fn parallel_groupby_firstseen_u64_deterministic<T, A, O>(
    keys: &[i64],
    values: &[T],
) -> PyResult<GroupByResult<O>>
where
    T: Copy + Send + Sync,
    O: Copy,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    parallel_groupby_firstseen_deterministic_impl::<T, A, O, u64>(keys, values)
}
