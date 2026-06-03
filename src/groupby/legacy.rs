use ahash::AHashMap;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::hash_map::Entry;

use crate::aggregation::Aggregator;

use super::order::{
    reorder_single_result_by_first_seen_u32, reorder_single_result_by_first_seen_u64,
};
use super::result::GroupByResult;

pub(super) fn parallel_groupby<T, A, O>(keys: &[i64], values: &[T]) -> PyResult<GroupByResult<O>>
where
    T: Copy + Send + Sync,
    O: Copy,
    A: Aggregator<T, O>,
{
    let chunk_size = (keys.len() / rayon::current_num_threads()).max(10_000);

    let merged: AHashMap<i64, A> = keys
        .par_chunks(chunk_size)
        .zip(values.par_chunks(chunk_size))
        .fold(
            AHashMap::default,
            |mut acc: AHashMap<i64, A>, (k_chunk, v_chunk)| {
                for (&key, &val) in k_chunk.iter().zip(v_chunk.iter()) {
                    acc.entry(key).or_insert_with(A::init).update(val);
                }
                acc
            },
        )
        .reduce(AHashMap::default, |mut map1, map2| {
            for (k, v) in map2 {
                match map1.entry(k) {
                    Entry::Occupied(mut e) => {
                        e.get_mut().merge(v);
                    }
                    Entry::Vacant(e) => {
                        e.insert(v);
                    }
                }
            }
            map1
        });

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

pub(super) fn parallel_groupby_firstseen_u32<T, A, O>(
    keys: &[i64],
    values: &[T],
) -> PyResult<GroupByResult<O>>
where
    T: Copy + Send + Sync,
    O: Copy,
    A: Aggregator<T, O> + Clone + Default,
{
    let n_rows = keys.len();
    if values.len() != n_rows {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "keys and values must have same length",
        ));
    }
    if n_rows > (u32::MAX as usize) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "idx32 kernel requires n_rows <= u32::MAX",
        ));
    }

    let chunk_size = (n_rows / rayon::current_num_threads()).max(10_000);

    let merged: AHashMap<i64, (A, u32)> = keys
        .par_chunks(chunk_size)
        .zip(values.par_chunks(chunk_size))
        .enumerate()
        .fold(
            AHashMap::default,
            |mut acc: AHashMap<i64, (A, u32)>, (chunk_idx, (k_chunk, v_chunk))| {
                // `par_chunks(...).enumerate()` is an IndexedParallelIterator, so
                // `chunk_idx` corresponds to the logical chunk order in the original slice.
                let base = (chunk_idx * chunk_size) as u32;
                for (i, (&key, &val)) in k_chunk.iter().zip(v_chunk.iter()).enumerate() {
                    let row = base + (i as u32);
                    match acc.entry(key) {
                        Entry::Occupied(mut e) => {
                            let (agg, first) = e.get_mut();
                            if row < *first {
                                *first = row;
                            }
                            agg.update(val);
                        }
                        Entry::Vacant(e) => {
                            let mut agg = A::init();
                            agg.update(val);
                            e.insert((agg, row));
                        }
                    }
                }
                acc
            },
        )
        .reduce(AHashMap::default, |mut map1, map2| {
            for (k, (agg2, first2)) in map2 {
                match map1.entry(k) {
                    Entry::Occupied(mut e) => {
                        let (agg1, first1) = e.get_mut();
                        *first1 = (*first1).min(first2);
                        agg1.merge(agg2);
                    }
                    Entry::Vacant(e) => {
                        e.insert((agg2, first2));
                    }
                }
            }
            map1
        });

    let mut result_keys = Vec::with_capacity(merged.len());
    let mut result_values = Vec::with_capacity(merged.len());
    let mut first_seen = Vec::with_capacity(merged.len());

    for (k, (agg, first)) in merged {
        result_keys.push(k);
        result_values.push(agg.finalize_owned());
        first_seen.push(first);
    }

    let mut result = GroupByResult {
        keys: result_keys,
        values: result_values,
    };
    reorder_single_result_by_first_seen_u32(&mut result, &first_seen);
    Ok(result)
}

pub(super) fn parallel_groupby_firstseen_u64<T, A, O>(
    keys: &[i64],
    values: &[T],
) -> PyResult<GroupByResult<O>>
where
    T: Copy + Send + Sync,
    O: Copy,
    A: Aggregator<T, O> + Clone + Default,
{
    let n_rows = keys.len();
    if values.len() != n_rows {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "keys and values must have same length",
        ));
    }

    let chunk_size = (n_rows / rayon::current_num_threads()).max(10_000);

    let merged: AHashMap<i64, (A, u64)> = keys
        .par_chunks(chunk_size)
        .zip(values.par_chunks(chunk_size))
        .enumerate()
        .fold(
            AHashMap::default,
            |mut acc: AHashMap<i64, (A, u64)>, (chunk_idx, (k_chunk, v_chunk))| {
                // `par_chunks(...).enumerate()` is an IndexedParallelIterator, so
                // `chunk_idx` corresponds to the logical chunk order in the original slice.
                let base = (chunk_idx * chunk_size) as u64;
                for (i, (&key, &val)) in k_chunk.iter().zip(v_chunk.iter()).enumerate() {
                    let row = base + (i as u64);
                    match acc.entry(key) {
                        Entry::Occupied(mut e) => {
                            let (agg, first) = e.get_mut();
                            if row < *first {
                                *first = row;
                            }
                            agg.update(val);
                        }
                        Entry::Vacant(e) => {
                            let mut agg = A::init();
                            agg.update(val);
                            e.insert((agg, row));
                        }
                    }
                }
                acc
            },
        )
        .reduce(AHashMap::default, |mut map1, map2| {
            for (k, (agg2, first2)) in map2 {
                match map1.entry(k) {
                    Entry::Occupied(mut e) => {
                        let (agg1, first1) = e.get_mut();
                        *first1 = (*first1).min(first2);
                        agg1.merge(agg2);
                    }
                    Entry::Vacant(e) => {
                        e.insert((agg2, first2));
                    }
                }
            }
            map1
        });

    let mut result_keys = Vec::with_capacity(merged.len());
    let mut result_values = Vec::with_capacity(merged.len());
    let mut first_seen = Vec::with_capacity(merged.len());

    for (k, (agg, first)) in merged {
        result_keys.push(k);
        result_values.push(agg.finalize_owned());
        first_seen.push(first);
    }

    let mut result = GroupByResult {
        keys: result_keys,
        values: result_values,
    };
    reorder_single_result_by_first_seen_u64(&mut result, &first_seen);
    Ok(result)
}
