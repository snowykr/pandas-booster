//! Parallel groupby implementation using Rayon's map-reduce pattern.
//!
//! This module provides high-performance groupby operations by:
//! 1. Chunking input arrays across CPU cores
//! 2. Building per-thread hash maps with partial aggregations
//! 3. Merging partial results via the [`Aggregator::merge`](crate::aggregation::Aggregator::merge) method
//!
//! Result ordering is **not guaranteed** for the default kernels as it depends on hash map
//! iteration order. The `*_firstseen_*` kernels reorder groups to preserve Pandas appearance
//! order (first-seen group order).

use ahash::AHashMap;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::hash_map::Entry;

use crate::radix_sort::{radix_sort_perm_by_u32, radix_sort_perm_by_u64};

use crate::aggregation::{
    Aggregator, CountAggF64, CountAggI64, MaxAggF64, MaxAggI64, MeanAggF64, MeanAggI64, MinAggF64,
    MinAggI64, SumAggF64, SumAggI64,
};

/// Result container for groupby operations, holding key-value pairs.
#[derive(Debug)]
pub struct GroupByResultF64 {
    pub keys: Vec<i64>,
    pub values: Vec<f64>,
}

fn reorder_single_result_by_first_seen_u32(result: &mut GroupByResultF64, first_seen: &[u32]) {
    if result.values.is_empty() {
        return;
    }
    debug_assert_eq!(result.values.len(), first_seen.len());
    debug_assert_eq!(result.keys.len(), first_seen.len());

    let perm = radix_sort_perm_by_u32(first_seen);

    let keys = &result.keys;
    let values = &result.values;
    let mut sorted_keys = Vec::with_capacity(keys.len());
    let mut sorted_values = Vec::with_capacity(values.len());
    for &g in &perm {
        sorted_keys.push(keys[g]);
        sorted_values.push(values[g]);
    }
    result.keys = sorted_keys;
    result.values = sorted_values;
}

fn reorder_single_result_by_first_seen_u64(result: &mut GroupByResultF64, first_seen: &[u64]) {
    if result.values.is_empty() {
        return;
    }
    debug_assert_eq!(result.values.len(), first_seen.len());
    debug_assert_eq!(result.keys.len(), first_seen.len());

    let perm = radix_sort_perm_by_u64(first_seen);

    let keys = &result.keys;
    let values = &result.values;
    let mut sorted_keys = Vec::with_capacity(keys.len());
    let mut sorted_values = Vec::with_capacity(values.len());
    for &g in &perm {
        sorted_keys.push(keys[g]);
        sorted_values.push(values[g]);
    }
    result.keys = sorted_keys;
    result.values = sorted_values;
}

fn parallel_groupby_f64<T, A>(keys: &[i64], values: &[T]) -> PyResult<GroupByResultF64>
where
    T: Copy + Send + Sync,
    A: Aggregator<T, f64> + Clone + Default,
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
                map1.entry(k)
                    .and_modify(|existing| existing.merge(v.clone()))
                    .or_insert(v);
            }
            map1
        });

    let mut result_keys = Vec::with_capacity(merged.len());
    let mut result_values = Vec::with_capacity(merged.len());

    for (k, agg) in merged {
        result_keys.push(k);
        result_values.push(agg.finalize());
    }

    Ok(GroupByResultF64 {
        keys: result_keys,
        values: result_values,
    })
}

fn parallel_groupby_firstseen_u32<T, A>(keys: &[i64], values: &[T]) -> PyResult<GroupByResultF64>
where
    T: Copy + Send + Sync,
    A: Aggregator<T, f64> + Clone + Default,
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
        result_values.push(agg.finalize());
        first_seen.push(first);
    }

    let mut result = GroupByResultF64 {
        keys: result_keys,
        values: result_values,
    };
    reorder_single_result_by_first_seen_u32(&mut result, &first_seen);
    Ok(result)
}

fn parallel_groupby_firstseen_u64<T, A>(keys: &[i64], values: &[T]) -> PyResult<GroupByResultF64>
where
    T: Copy + Send + Sync,
    A: Aggregator<T, f64> + Clone + Default,
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
        result_values.push(agg.finalize());
        first_seen.push(first);
    }

    let mut result = GroupByResultF64 {
        keys: result_keys,
        values: result_values,
    };
    reorder_single_result_by_first_seen_u64(&mut result, &first_seen);
    Ok(result)
}

pub fn parallel_groupby_sum_f64(keys: &[i64], values: &[f64]) -> PyResult<GroupByResultF64> {
    parallel_groupby_f64::<f64, SumAggF64>(keys, values)
}

pub fn parallel_groupby_sum_f64_firstseen_u32(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_u32::<f64, SumAggF64>(keys, values)
}

pub fn parallel_groupby_sum_f64_firstseen_u64(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_u64::<f64, SumAggF64>(keys, values)
}

pub fn parallel_groupby_mean_f64(keys: &[i64], values: &[f64]) -> PyResult<GroupByResultF64> {
    parallel_groupby_f64::<f64, MeanAggF64>(keys, values)
}

pub fn parallel_groupby_mean_f64_firstseen_u32(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_u32::<f64, MeanAggF64>(keys, values)
}

pub fn parallel_groupby_mean_f64_firstseen_u64(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_u64::<f64, MeanAggF64>(keys, values)
}

pub fn parallel_groupby_min_f64(keys: &[i64], values: &[f64]) -> PyResult<GroupByResultF64> {
    parallel_groupby_f64::<f64, MinAggF64>(keys, values)
}

pub fn parallel_groupby_min_f64_firstseen_u32(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_u32::<f64, MinAggF64>(keys, values)
}

pub fn parallel_groupby_min_f64_firstseen_u64(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_u64::<f64, MinAggF64>(keys, values)
}

pub fn parallel_groupby_max_f64(keys: &[i64], values: &[f64]) -> PyResult<GroupByResultF64> {
    parallel_groupby_f64::<f64, MaxAggF64>(keys, values)
}

pub fn parallel_groupby_max_f64_firstseen_u32(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_u32::<f64, MaxAggF64>(keys, values)
}

pub fn parallel_groupby_max_f64_firstseen_u64(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_u64::<f64, MaxAggF64>(keys, values)
}

pub fn parallel_groupby_sum_i64(keys: &[i64], values: &[i64]) -> PyResult<GroupByResultF64> {
    parallel_groupby_f64::<i64, SumAggI64>(keys, values)
}

pub fn parallel_groupby_sum_i64_firstseen_u32(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_u32::<i64, SumAggI64>(keys, values)
}

pub fn parallel_groupby_sum_i64_firstseen_u64(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_u64::<i64, SumAggI64>(keys, values)
}

pub fn parallel_groupby_mean_i64(keys: &[i64], values: &[i64]) -> PyResult<GroupByResultF64> {
    parallel_groupby_f64::<i64, MeanAggI64>(keys, values)
}

pub fn parallel_groupby_mean_i64_firstseen_u32(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_u32::<i64, MeanAggI64>(keys, values)
}

pub fn parallel_groupby_mean_i64_firstseen_u64(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_u64::<i64, MeanAggI64>(keys, values)
}

pub fn parallel_groupby_min_i64(keys: &[i64], values: &[i64]) -> PyResult<GroupByResultF64> {
    parallel_groupby_f64::<i64, MinAggI64>(keys, values)
}

pub fn parallel_groupby_min_i64_firstseen_u32(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_u32::<i64, MinAggI64>(keys, values)
}

pub fn parallel_groupby_min_i64_firstseen_u64(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_u64::<i64, MinAggI64>(keys, values)
}

pub fn parallel_groupby_max_i64(keys: &[i64], values: &[i64]) -> PyResult<GroupByResultF64> {
    parallel_groupby_f64::<i64, MaxAggI64>(keys, values)
}

pub fn parallel_groupby_max_i64_firstseen_u32(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_u32::<i64, MaxAggI64>(keys, values)
}

pub fn parallel_groupby_max_i64_firstseen_u64(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_u64::<i64, MaxAggI64>(keys, values)
}

pub fn parallel_groupby_count_f64(keys: &[i64], values: &[f64]) -> PyResult<GroupByResultF64> {
    parallel_groupby_f64::<f64, CountAggF64>(keys, values)
}

pub fn parallel_groupby_count_f64_firstseen_u32(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_u32::<f64, CountAggF64>(keys, values)
}

pub fn parallel_groupby_count_f64_firstseen_u64(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_u64::<f64, CountAggF64>(keys, values)
}

pub fn parallel_groupby_count_i64(keys: &[i64], values: &[i64]) -> PyResult<GroupByResultF64> {
    parallel_groupby_f64::<i64, CountAggI64>(keys, values)
}

pub fn parallel_groupby_count_i64_firstseen_u32(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_u32::<i64, CountAggI64>(keys, values)
}

pub fn parallel_groupby_count_i64_firstseen_u64(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_u64::<i64, CountAggI64>(keys, values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::ThreadPoolBuilder;

    #[test]
    fn test_groupby_sum_f64() {
        let keys = vec![1, 2, 1, 2, 1];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = parallel_groupby_sum_f64(&keys, &values).unwrap();

        let mut map: AHashMap<i64, f64> = AHashMap::new();
        for (k, v) in result.keys.iter().zip(result.values.iter()) {
            map.insert(*k, *v);
        }

        assert!((map[&1] - 9.0).abs() < 1e-10);
        assert!((map[&2] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_groupby_mean_f64() {
        let keys = vec![1, 2, 1, 2, 1];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = parallel_groupby_mean_f64(&keys, &values).unwrap();

        let mut map: AHashMap<i64, f64> = AHashMap::new();
        for (k, v) in result.keys.iter().zip(result.values.iter()) {
            map.insert(*k, *v);
        }

        assert!((map[&1] - 3.0).abs() < 1e-10);
        assert!((map[&2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_groupby_with_nan() {
        let keys = vec![1, 1, 1];
        let values = vec![1.0, f64::NAN, 2.0];
        let result = parallel_groupby_sum_f64(&keys, &values).unwrap();

        assert!((result.values[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_groupby_all_nan_group() {
        let keys = vec![1, 1, 2, 2];
        let values = vec![f64::NAN, f64::NAN, 1.0, 2.0];
        let result = parallel_groupby_sum_f64(&keys, &values).unwrap();

        let mut map: AHashMap<i64, f64> = AHashMap::new();
        for (k, v) in result.keys.iter().zip(result.values.iter()) {
            map.insert(*k, *v);
        }

        // All-NaN group returns 0.0, matching pandas behavior: df.groupby('k')['v'].sum()
        assert!((map[&1] - 0.0).abs() < 1e-10);
        assert!((map[&2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_groupby_mean_all_nan_group() {
        let keys = vec![1, 1, 2, 2];
        let values = vec![f64::NAN, f64::NAN, 1.0, 2.0];
        let result = parallel_groupby_mean_f64(&keys, &values).unwrap();

        let mut map: AHashMap<i64, f64> = AHashMap::new();
        for (k, v) in result.keys.iter().zip(result.values.iter()) {
            map.insert(*k, *v);
        }

        assert!(map[&1].is_nan());
        assert!((map[&2] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_groupby_min_all_nan_group() {
        let keys = vec![1, 1, 2, 2];
        let values = vec![f64::NAN, f64::NAN, 1.0, 2.0];
        let result = parallel_groupby_min_f64(&keys, &values).unwrap();

        let mut map: AHashMap<i64, f64> = AHashMap::new();
        for (k, v) in result.keys.iter().zip(result.values.iter()) {
            map.insert(*k, *v);
        }

        assert!(map[&1].is_nan());
        assert!((map[&2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_groupby_max_all_nan_group() {
        let keys = vec![1, 1, 2, 2];
        let values = vec![f64::NAN, f64::NAN, 1.0, 2.0];
        let result = parallel_groupby_max_f64(&keys, &values).unwrap();

        let mut map: AHashMap<i64, f64> = AHashMap::new();
        for (k, v) in result.keys.iter().zip(result.values.iter()) {
            map.insert(*k, *v);
        }

        assert!(map[&1].is_nan());
        assert!((map[&2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_groupby_sum_i64() {
        let keys = vec![1, 2, 1, 2, 1];
        let values: Vec<i64> = vec![1, 2, 3, 4, 5];
        let result = parallel_groupby_sum_i64(&keys, &values).unwrap();

        let mut map: AHashMap<i64, f64> = AHashMap::new();
        for (k, v) in result.keys.iter().zip(result.values.iter()) {
            map.insert(*k, *v);
        }

        assert!((map[&1] - 9.0).abs() < 1e-10);
        assert!((map[&2] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_groupby_min_i64() {
        let keys = vec![1, 2, 1, 2, 1];
        let values: Vec<i64> = vec![5, 2, 3, 4, 1];
        let result = parallel_groupby_min_i64(&keys, &values).unwrap();

        let mut map: AHashMap<i64, f64> = AHashMap::new();
        for (k, v) in result.keys.iter().zip(result.values.iter()) {
            map.insert(*k, *v);
        }

        assert!((map[&1] - 1.0).abs() < 1e-10);
        assert!((map[&2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_groupby_max_i64() {
        let keys = vec![1, 2, 1, 2, 1];
        let values: Vec<i64> = vec![5, 2, 3, 4, 1];
        let result = parallel_groupby_max_i64(&keys, &values).unwrap();

        let mut map: AHashMap<i64, f64> = AHashMap::new();
        for (k, v) in result.keys.iter().zip(result.values.iter()) {
            map.insert(*k, *v);
        }

        assert!((map[&1] - 5.0).abs() < 1e-10);
        assert!((map[&2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_groupby_count_f64() {
        let keys = vec![1, 2, 1, 2, 1];
        let values = vec![1.0, 2.0, f64::NAN, 4.0, 5.0];
        let result = parallel_groupby_count_f64(&keys, &values).unwrap();

        let mut map: AHashMap<i64, f64> = AHashMap::new();
        for (k, v) in result.keys.iter().zip(result.values.iter()) {
            map.insert(*k, *v);
        }

        assert!((map[&1] - 2.0).abs() < 1e-10);
        assert!((map[&2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_groupby_count_i64() {
        let keys = vec![1, 2, 1, 2, 1];
        let values: Vec<i64> = vec![1, 2, 3, 4, 5];
        let result = parallel_groupby_count_i64(&keys, &values).unwrap();

        let mut map: AHashMap<i64, f64> = AHashMap::new();
        for (k, v) in result.keys.iter().zip(result.values.iter()) {
            map.insert(*k, *v);
        }

        assert!((map[&1] - 3.0).abs() < 1e-10);
        assert!((map[&2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_groupby_firstseen_order_u32_across_chunks() {
        // Force multiple Rayon threads so chunking happens.
        let pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        pool.install(|| {
            let n = 120_000;
            let mut keys = vec![0i64; n];
            // Ensure first-seen occurrences are far apart.
            keys[0] = 10;
            keys[50_000] = 20;
            keys[100_000] = 30;

            // Fill the rest with repetitions (no new groups).
            for (i, k) in keys.iter_mut().enumerate().skip(1) {
                if i == 50_000 || i == 100_000 {
                    continue;
                }
                *k = match i % 3 {
                    0 => 10,
                    1 => 20,
                    _ => 30,
                };
            }
            let values = vec![1.0f64; n];

            let result = parallel_groupby_sum_f64_firstseen_u32(&keys, &values).unwrap();
            assert_eq!(result.keys, vec![10, 20, 30]);
        });
    }

    #[test]
    fn test_groupby_firstseen_order_u64_across_chunks() {
        // Same as the u32 test, but exercising the u64 first-seen path.
        let pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        pool.install(|| {
            let n = 120_000;
            let mut keys = vec![0i64; n];
            keys[0] = 10;
            keys[50_000] = 20;
            keys[100_000] = 30;

            for (i, k) in keys.iter_mut().enumerate().skip(1) {
                if i == 50_000 || i == 100_000 {
                    continue;
                }
                *k = match i % 3 {
                    0 => 10,
                    1 => 20,
                    _ => 30,
                };
            }
            let values = vec![1.0f64; n];

            let result = parallel_groupby_sum_f64_firstseen_u64(&keys, &values).unwrap();
            assert_eq!(result.keys, vec![10, 20, 30]);
        });
    }
}
