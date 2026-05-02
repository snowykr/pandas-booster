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

use ahash::{AHashMap, AHashSet};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::hash_map::Entry;
use std::hash::{Hash, Hasher};
use std::time::Instant;

use crate::radix_groupby::stable_scatter_by_partition;
use crate::radix_sort::{
    radix_sort_perm_by_i64_par, radix_sort_perm_by_u32, radix_sort_perm_by_u64,
};

const RADIX_SORT_THRESHOLD: usize = 2048;
const DETERMINISTIC_TARGET_CHUNK_SIZE: usize = 131_072;
const DETERMINISTIC_MIN_CHUNKS: usize = 4;
const DETERMINISTIC_MAX_CHUNKS: usize = 2048;
const STD_VAR_ENGINE_SAMPLE_SIZE: usize = 16_384;
const STD_VAR_ENGINE_MIN_SAMPLE_UNIQUES: usize = 4_096;

use crate::aggregation::{
    Aggregator, CountAggF64, CountAggI64, MaxAggF64, MaxAggI64, MeanAggF64, MeanAggI64, MinAggF64,
    MinAggI64, ProdAggF64, ProdAggI64, StdAggF64, StdAggI64, SumAggF64, SumAggI64, VarAggF64,
    VarAggI64,
};

/// Result container for groupby operations, holding key-value pairs.
#[derive(Debug)]
pub struct GroupByResult<V> {
    pub keys: Vec<i64>,
    pub values: Vec<V>,
}

#[derive(Debug, Clone)]
pub struct SingleKeyPhaseProfile {
    pub local_build_s: f64,
    pub merge_s: f64,
    pub reorder_s: f64,
    pub materialize_s: f64,
    pub partial_group_total: usize,
    pub final_group_count: usize,
}

#[derive(Debug)]
pub struct ProfiledGroupByResult<V> {
    pub result: GroupByResult<V>,
    pub profile: SingleKeyPhaseProfile,
}

pub type GroupByResultF64 = GroupByResult<f64>;
pub type GroupByResultI64 = GroupByResult<i64>;

fn reorder_single_result_by_first_seen_u32<V: Copy>(
    result: &mut GroupByResult<V>,
    first_seen: &[u32],
) {
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

fn reorder_single_result_by_first_seen_u64<V: Copy>(
    result: &mut GroupByResult<V>,
    first_seen: &[u64],
) {
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

fn reorder_single_result_by_key<V: Copy>(result: &mut GroupByResult<V>) {
    if result.values.is_empty() {
        return;
    }
    debug_assert_eq!(result.keys.len(), result.values.len());

    let keys = &result.keys;
    let values = &result.values;

    let perm = if keys.len() < RADIX_SORT_THRESHOLD {
        let mut perm: Vec<usize> = (0..keys.len()).collect();
        perm.sort_unstable_by(|&i, &j| keys[i].cmp(&keys[j]).then(i.cmp(&j)));
        perm
    } else {
        radix_sort_perm_by_i64_par(keys)
    };

    let mut sorted_keys = Vec::with_capacity(keys.len());
    let mut sorted_values = Vec::with_capacity(values.len());
    for &idx in &perm {
        sorted_keys.push(keys[idx]);
        sorted_values.push(values[idx]);
    }

    result.keys = sorted_keys;
    result.values = sorted_values;
}

fn fixed_chunk_size(n_rows: usize) -> usize {
    if n_rows == 0 {
        return 1;
    }
    let n_chunks = n_rows
        .div_ceil(DETERMINISTIC_TARGET_CHUNK_SIZE)
        .clamp(DETERMINISTIC_MIN_CHUNKS, DETERMINISTIC_MAX_CHUNKS);
    n_rows.div_ceil(n_chunks)
}

fn build_chunk_ranges(n_rows: usize, chunk_size: usize) -> Vec<(usize, usize)> {
    if n_rows == 0 {
        return Vec::new();
    }
    let mut ranges = Vec::with_capacity(n_rows.div_ceil(chunk_size));
    let mut start = 0usize;
    while start < n_rows {
        let end = (start + chunk_size).min(n_rows);
        ranges.push((start, end));
        start = end;
    }
    ranges
}

trait FirstSeenRowIndex: Copy + Ord + Send + Sync {
    fn validate_n_rows(_n_rows: usize) -> PyResult<()> {
        Ok(())
    }

    fn from_row_index(row: usize) -> Self;

    fn reorder_result<V: Copy>(result: &mut GroupByResult<V>, first_seen: &[Self]);
}

impl FirstSeenRowIndex for u32 {
    fn validate_n_rows(n_rows: usize) -> PyResult<()> {
        if n_rows > (u32::MAX as usize) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "idx32 kernel requires n_rows <= u32::MAX",
            ));
        }
        Ok(())
    }

    fn from_row_index(row: usize) -> Self {
        row as u32
    }

    fn reorder_result<V: Copy>(result: &mut GroupByResult<V>, first_seen: &[Self]) {
        reorder_single_result_by_first_seen_u32(result, first_seen);
    }
}

impl FirstSeenRowIndex for u64 {
    fn from_row_index(row: usize) -> Self {
        row as u64
    }

    fn reorder_result<V: Copy>(result: &mut GroupByResult<V>, first_seen: &[Self]) {
        reorder_single_result_by_first_seen_u64(result, first_seen);
    }
}

struct FirstSeenMaterializedResult<I, V> {
    result: GroupByResult<V>,
    first_seen: Vec<I>,
}

struct SingleKeyPartitionState<A, I> {
    gid_map: AHashMap<i64, usize>,
    aggs: Vec<A>,
    first_seen: Vec<I>,
}

impl<A, I> SingleKeyPartitionState<A, I> {
    #[inline]
    fn len(&self) -> usize {
        self.aggs.len()
    }
}

trait PairwiseReduceValue<T, O, A>: Send
where
    A: Aggregator<T, O>,
{
    fn merge_value(left: &mut Self, right: Self);
}

impl<T, O, A> PairwiseReduceValue<T, O, A> for A
where
    A: Aggregator<T, O> + Send,
{
    #[inline]
    fn merge_value(left: &mut Self, right: Self) {
        left.merge(right);
    }
}

impl<T, O, A> PairwiseReduceValue<T, O, A> for (A, u32)
where
    A: Aggregator<T, O> + Send,
{
    #[inline]
    fn merge_value(left: &mut Self, right: Self) {
        left.1 = left.1.min(right.1);
        left.0.merge(right.0);
    }
}

impl<T, O, A> PairwiseReduceValue<T, O, A> for (A, u64)
where
    A: Aggregator<T, O> + Send,
{
    #[inline]
    fn merge_value(left: &mut Self, right: Self) {
        left.1 = left.1.min(right.1);
        left.0.merge(right.0);
    }
}

fn reduce_partial_maps_pairwise<T, O, A, V>(mut partials: Vec<AHashMap<i64, V>>) -> AHashMap<i64, V>
where
    T: Copy + Send + Sync,
    O: Copy,
    A: Aggregator<T, O> + Send,
    V: PairwiseReduceValue<T, O, A>,
{
    if partials.is_empty() {
        return AHashMap::default();
    }

    while partials.len() > 1 {
        let carry = if partials.len() % 2 == 1 {
            partials.pop()
        } else {
            None
        };

        let mut iter = partials.into_iter();
        let mut pairs = Vec::with_capacity(iter.len() / 2);
        while let Some(left) = iter.next() {
            let right = iter.next().expect("pairwise merge requires right map");
            pairs.push((left, right));
        }

        let mut next: Vec<AHashMap<i64, V>> = pairs
            .into_par_iter()
            .map(|(mut left, right)| {
                for (k, v) in right {
                    match left.entry(k) {
                        Entry::Occupied(mut e) => {
                            V::merge_value(e.get_mut(), v);
                        }
                        Entry::Vacant(e) => {
                            e.insert(v);
                        }
                    }
                }
                left
            })
            .collect();

        if let Some(map) = carry {
            next.push(map);
        }
        partials = next;
    }

    partials
        .pop()
        .expect("non-empty partial maps must produce one map")
}

fn validate_firstseen_deterministic_inputs<T, I>(keys: &[i64], values: &[T]) -> PyResult<usize>
where
    I: FirstSeenRowIndex,
{
    let n_rows = keys.len();
    if values.len() != n_rows {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "keys and values must have same length",
        ));
    }
    I::validate_n_rows(n_rows)?;
    Ok(n_rows)
}

fn prepare_firstseen_deterministic_ranges<T, I>(
    keys: &[i64],
    values: &[T],
) -> PyResult<Vec<(usize, usize)>>
where
    I: FirstSeenRowIndex,
{
    let n_rows = validate_firstseen_deterministic_inputs::<T, I>(keys, values)?;
    let chunk_size = fixed_chunk_size(n_rows);
    Ok(build_chunk_ranges(n_rows, chunk_size))
}

#[inline]
fn hash_single_key(key: i64) -> u64 {
    use ahash::AHasher;

    let mut hasher = AHasher::default();
    key.hash(&mut hasher);
    hasher.finish()
}

fn compute_single_key_hashes(keys: &[i64]) -> Vec<u64> {
    let mut hashes = vec![0u64; keys.len()];
    hashes
        .par_iter_mut()
        .enumerate()
        .for_each(|(row, hash_out)| {
            *hash_out = hash_single_key(keys[row]);
        });
    hashes
}

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

fn build_deterministic_firstseen_partials<T, O, A, I>(
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

fn combine_deterministic_firstseen_partials<T, O, A, I>(
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

fn materialize_deterministic_firstseen_result<T, O, A, I>(
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
        result_values.push(agg.finalize());
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

fn finalize_deterministic_firstseen_result<I, V: Copy>(
    mut materialized: FirstSeenMaterializedResult<I, V>,
) -> GroupByResult<V>
where
    I: FirstSeenRowIndex,
{
    I::reorder_result(&mut materialized.result, &materialized.first_seen);
    materialized.result
}

fn estimate_sample_unique_keys(keys: &[i64]) -> usize {
    let sample_size = keys.len().min(STD_VAR_ENGINE_SAMPLE_SIZE);
    if sample_size == 0 {
        return 0;
    }

    let stride = keys.len().div_ceil(sample_size);
    let mut seen = AHashSet::with_capacity(sample_size);
    let mut row = 0usize;
    let mut sampled = 0usize;

    while row < keys.len() && sampled < sample_size {
        seen.insert(keys[row]);
        row += stride;
        sampled += 1;
    }

    seen.len()
}

#[inline]
fn should_use_partitioned_std_var_engine(keys: &[i64]) -> bool {
    let sample_size = keys.len().min(STD_VAR_ENGINE_SAMPLE_SIZE);
    sample_size > STD_VAR_ENGINE_MIN_SAMPLE_UNIQUES
        && estimate_sample_unique_keys(keys) >= STD_VAR_ENGINE_MIN_SAMPLE_UNIQUES
}

fn build_partitioned_firstseen_state<T, O, A, I>(
    perm_rows: &[usize],
    keys: &[i64],
    values: &[T],
) -> SingleKeyPartitionState<A, I>
where
    T: Copy,
    A: Aggregator<T, O>,
    I: FirstSeenRowIndex,
{
    let mut gid_map: AHashMap<i64, usize> = AHashMap::default();
    let mut aggs: Vec<A> = Vec::new();
    let mut first_seen: Vec<I> = Vec::new();

    for &row in perm_rows {
        let key = keys[row];
        let value = values[row];
        let row_index = I::from_row_index(row);

        if let Some(&gid) = gid_map.get(&key) {
            if row_index < first_seen[gid] {
                first_seen[gid] = row_index;
            }
            aggs[gid].update(value);
        } else {
            let gid = aggs.len();
            gid_map.insert(key, gid);

            let mut agg = A::init();
            agg.update(value);
            aggs.push(agg);
            first_seen.push(row_index);
        }
    }

    SingleKeyPartitionState {
        gid_map,
        aggs,
        first_seen,
    }
}

fn build_partitioned_deterministic_firstseen_states<T, O, A, I>(
    keys: &[i64],
    values: &[T],
) -> PyResult<Vec<SingleKeyPartitionState<A, I>>>
where
    T: Copy + Send + Sync,
    A: Aggregator<T, O> + Send,
    I: FirstSeenRowIndex,
{
    let n_rows = validate_firstseen_deterministic_inputs::<T, I>(keys, values)?;
    if n_rows == 0 {
        return Ok(Vec::new());
    }

    let hashes = compute_single_key_hashes(keys);
    let (perm, offsets) = stable_scatter_by_partition(&hashes);

    Ok((0..offsets.len() - 1)
        .into_par_iter()
        .map(|partition| {
            let start = offsets[partition];
            let end = offsets[partition + 1];
            build_partitioned_firstseen_state::<T, O, A, I>(&perm[start..end], keys, values)
        })
        .collect())
}

fn materialize_partitioned_deterministic_firstseen_states<T, O, A, I>(
    states: Vec<SingleKeyPartitionState<A, I>>,
) -> FirstSeenMaterializedResult<I, O>
where
    O: Copy,
    A: Aggregator<T, O>,
    I: FirstSeenRowIndex,
{
    let total_groups: usize = states.iter().map(SingleKeyPartitionState::len).sum();
    let mut result_keys = Vec::with_capacity(total_groups);
    let mut result_values = Vec::with_capacity(total_groups);
    let mut first_seen = Vec::with_capacity(total_groups);

    for state in states {
        let SingleKeyPartitionState {
            gid_map,
            aggs,
            first_seen: state_first_seen,
        } = state;

        debug_assert_eq!(gid_map.len(), aggs.len());
        debug_assert_eq!(gid_map.len(), state_first_seen.len());

        for (key, gid) in gid_map {
            result_keys.push(key);
            result_values.push(aggs[gid].finalize());
            first_seen.push(state_first_seen[gid]);
        }
    }

    FirstSeenMaterializedResult {
        result: GroupByResult {
            keys: result_keys,
            values: result_values,
        },
        first_seen,
    }
}

fn parallel_groupby_firstseen_partitioned_impl<T, A, O, I>(
    keys: &[i64],
    values: &[T],
) -> PyResult<GroupByResult<O>>
where
    T: Copy + Send + Sync,
    O: Copy,
    A: Aggregator<T, O> + Send,
    I: FirstSeenRowIndex,
{
    let states = build_partitioned_deterministic_firstseen_states::<T, O, A, I>(keys, values)?;
    let materialized = materialize_partitioned_deterministic_firstseen_states::<T, O, A, I>(states);
    Ok(finalize_deterministic_firstseen_result(materialized))
}

fn parallel_groupby_partitioned_unordered_impl<T, A, O, I>(
    keys: &[i64],
    values: &[T],
) -> PyResult<GroupByResult<O>>
where
    T: Copy + Send + Sync,
    O: Copy,
    A: Aggregator<T, O> + Send,
    I: FirstSeenRowIndex,
{
    let states = build_partitioned_deterministic_firstseen_states::<T, O, A, I>(keys, values)?;
    let materialized = materialize_partitioned_deterministic_firstseen_states::<T, O, A, I>(states);
    Ok(materialized.result)
}

fn profile_parallel_groupby_firstseen_partitioned_impl<T, A, O, I>(
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

fn profile_parallel_groupby_partitioned_unordered_impl<T, A, O, I>(
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

fn parallel_groupby_firstseen_deterministic_impl<T, A, O, I>(
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

fn profile_parallel_groupby_firstseen_deterministic_impl<T, A, O, I>(
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

fn parallel_groupby_firstseen_std_var_impl<T, A, O, I>(
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
    if should_use_partitioned_std_var_engine(keys) {
        parallel_groupby_firstseen_partitioned_impl::<T, A, O, I>(keys, values)
    } else {
        parallel_groupby_firstseen_deterministic_impl::<T, A, O, I>(keys, values)
    }
}

fn profile_parallel_groupby_firstseen_std_var_impl<T, A, O, I>(
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

fn parallel_groupby_std_var_impl<T, A, O>(keys: &[i64], values: &[T]) -> PyResult<GroupByResult<O>>
where
    T: Copy + Send + Sync,
    O: Copy,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    if should_use_partitioned_std_var_engine(keys) {
        if keys.len() <= u32::MAX as usize {
            parallel_groupby_partitioned_unordered_impl::<T, A, O, u32>(keys, values)
        } else {
            parallel_groupby_partitioned_unordered_impl::<T, A, O, u64>(keys, values)
        }
    } else {
        parallel_groupby_deterministic::<T, A, O>(keys, values)
    }
}

fn profile_parallel_groupby_std_var_impl<T, A, O>(
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

fn parallel_groupby_deterministic<T, A, O>(keys: &[i64], values: &[T]) -> PyResult<GroupByResult<O>>
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
        result_values.push(agg.finalize());
    }

    Ok(GroupByResult {
        keys: result_keys,
        values: result_values,
    })
}

fn profile_parallel_groupby_deterministic<T, A, O>(
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
        result_values.push(agg.finalize());
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

fn parallel_groupby_firstseen_u32_deterministic<T, A, O>(
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

fn parallel_groupby_firstseen_u64_deterministic<T, A, O>(
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

fn parallel_groupby<T, A, O>(keys: &[i64], values: &[T]) -> PyResult<GroupByResult<O>>
where
    T: Copy + Send + Sync,
    O: Copy,
    A: Aggregator<T, O> + Clone + Default,
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

    Ok(GroupByResult {
        keys: result_keys,
        values: result_values,
    })
}

fn parallel_groupby_firstseen_u32<T, A, O>(keys: &[i64], values: &[T]) -> PyResult<GroupByResult<O>>
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
        result_values.push(agg.finalize());
        first_seen.push(first);
    }

    let mut result = GroupByResult {
        keys: result_keys,
        values: result_values,
    };
    reorder_single_result_by_first_seen_u32(&mut result, &first_seen);
    Ok(result)
}

fn parallel_groupby_firstseen_u64<T, A, O>(keys: &[i64], values: &[T]) -> PyResult<GroupByResult<O>>
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
        result_values.push(agg.finalize());
        first_seen.push(first);
    }

    let mut result = GroupByResult {
        keys: result_keys,
        values: result_values,
    };
    reorder_single_result_by_first_seen_u64(&mut result, &first_seen);
    Ok(result)
}

pub fn parallel_groupby_sum_f64(keys: &[i64], values: &[f64]) -> PyResult<GroupByResultF64> {
    parallel_groupby_deterministic::<f64, SumAggF64, f64>(keys, values)
}

pub fn parallel_groupby_sum_f64_sorted(keys: &[i64], values: &[f64]) -> PyResult<GroupByResultF64> {
    let mut result = parallel_groupby_sum_f64(keys, values)?;
    reorder_single_result_by_key(&mut result);
    Ok(result)
}

pub fn parallel_groupby_sum_f64_firstseen_u32(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_u32_deterministic::<f64, SumAggF64, f64>(keys, values)
}

pub fn parallel_groupby_sum_f64_firstseen_u64(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_u64_deterministic::<f64, SumAggF64, f64>(keys, values)
}

pub fn parallel_groupby_prod_f64(keys: &[i64], values: &[f64]) -> PyResult<GroupByResultF64> {
    parallel_groupby_deterministic::<f64, ProdAggF64, f64>(keys, values)
}

pub fn parallel_groupby_prod_f64_sorted(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    let mut result = parallel_groupby_prod_f64(keys, values)?;
    reorder_single_result_by_key(&mut result);
    Ok(result)
}

pub fn parallel_groupby_prod_f64_firstseen_u32(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_u32_deterministic::<f64, ProdAggF64, f64>(keys, values)
}

pub fn parallel_groupby_prod_f64_firstseen_u64(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_u64_deterministic::<f64, ProdAggF64, f64>(keys, values)
}

pub fn parallel_groupby_mean_f64(keys: &[i64], values: &[f64]) -> PyResult<GroupByResultF64> {
    parallel_groupby_deterministic::<f64, MeanAggF64, f64>(keys, values)
}

pub fn parallel_groupby_mean_f64_sorted(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    let mut result = parallel_groupby_mean_f64(keys, values)?;
    reorder_single_result_by_key(&mut result);
    Ok(result)
}

pub fn parallel_groupby_mean_f64_firstseen_u32(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_u32_deterministic::<f64, MeanAggF64, f64>(keys, values)
}

pub fn parallel_groupby_mean_f64_firstseen_u64(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_u64_deterministic::<f64, MeanAggF64, f64>(keys, values)
}

pub fn parallel_groupby_var_f64(keys: &[i64], values: &[f64]) -> PyResult<GroupByResultF64> {
    parallel_groupby_std_var_impl::<f64, VarAggF64, f64>(keys, values)
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
    let mut profiled = profile_parallel_groupby_std_var_impl::<f64, VarAggF64, f64>(keys, values)?;
    let reorder_start = Instant::now();
    reorder_single_result_by_key(&mut profiled.result);
    profiled.profile.reorder_s += reorder_start.elapsed().as_secs_f64();
    Ok(profiled)
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
    parallel_groupby_std_var_impl::<f64, StdAggF64, f64>(keys, values)
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
    let mut profiled = profile_parallel_groupby_std_var_impl::<f64, StdAggF64, f64>(keys, values)?;
    let reorder_start = Instant::now();
    reorder_single_result_by_key(&mut profiled.result);
    profiled.profile.reorder_s += reorder_start.elapsed().as_secs_f64();
    Ok(profiled)
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
    parallel_groupby_firstseen_u32::<f64, MinAggF64, f64>(keys, values)
}

pub fn parallel_groupby_min_f64_firstseen_u64(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_u64::<f64, MinAggF64, f64>(keys, values)
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
    parallel_groupby_firstseen_u32::<f64, MaxAggF64, f64>(keys, values)
}

pub fn parallel_groupby_max_f64_firstseen_u64(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_u64::<f64, MaxAggF64, f64>(keys, values)
}

pub fn parallel_groupby_sum_i64(keys: &[i64], values: &[i64]) -> PyResult<GroupByResultI64> {
    parallel_groupby::<i64, SumAggI64, i64>(keys, values)
}

pub fn parallel_groupby_sum_i64_sorted(keys: &[i64], values: &[i64]) -> PyResult<GroupByResultI64> {
    let mut result = parallel_groupby_sum_i64(keys, values)?;
    reorder_single_result_by_key(&mut result);
    Ok(result)
}

pub fn parallel_groupby_sum_i64_firstseen_u32(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultI64> {
    parallel_groupby_firstseen_u32::<i64, SumAggI64, i64>(keys, values)
}

pub fn parallel_groupby_sum_i64_firstseen_u64(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultI64> {
    parallel_groupby_firstseen_u64::<i64, SumAggI64, i64>(keys, values)
}

pub fn parallel_groupby_prod_i64(keys: &[i64], values: &[i64]) -> PyResult<GroupByResultI64> {
    parallel_groupby::<i64, ProdAggI64, i64>(keys, values)
}

pub fn parallel_groupby_prod_i64_sorted(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultI64> {
    let mut result = parallel_groupby_prod_i64(keys, values)?;
    reorder_single_result_by_key(&mut result);
    Ok(result)
}

pub fn parallel_groupby_prod_i64_firstseen_u32(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultI64> {
    parallel_groupby_firstseen_u32::<i64, ProdAggI64, i64>(keys, values)
}

pub fn parallel_groupby_prod_i64_firstseen_u64(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultI64> {
    parallel_groupby_firstseen_u64::<i64, ProdAggI64, i64>(keys, values)
}

pub fn parallel_groupby_mean_i64(keys: &[i64], values: &[i64]) -> PyResult<GroupByResultF64> {
    parallel_groupby::<i64, MeanAggI64, f64>(keys, values)
}

pub fn parallel_groupby_mean_i64_sorted(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultF64> {
    let mut result = parallel_groupby_mean_i64(keys, values)?;
    reorder_single_result_by_key(&mut result);
    Ok(result)
}

pub fn parallel_groupby_mean_i64_firstseen_u32(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_u32::<i64, MeanAggI64, f64>(keys, values)
}

pub fn parallel_groupby_mean_i64_firstseen_u64(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_u64::<i64, MeanAggI64, f64>(keys, values)
}

pub fn parallel_groupby_var_i64(keys: &[i64], values: &[i64]) -> PyResult<GroupByResultF64> {
    parallel_groupby_std_var_impl::<i64, VarAggI64, f64>(keys, values)
}

pub fn parallel_groupby_var_i64_sorted(keys: &[i64], values: &[i64]) -> PyResult<GroupByResultF64> {
    let mut result = parallel_groupby_var_i64(keys, values)?;
    reorder_single_result_by_key(&mut result);
    Ok(result)
}

pub fn parallel_groupby_var_i64_firstseen_u32(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_std_var_impl::<i64, VarAggI64, f64, u32>(keys, values)
}

pub fn parallel_groupby_var_i64_firstseen_u64(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_std_var_impl::<i64, VarAggI64, f64, u64>(keys, values)
}

pub fn parallel_groupby_std_i64(keys: &[i64], values: &[i64]) -> PyResult<GroupByResultF64> {
    parallel_groupby_std_var_impl::<i64, StdAggI64, f64>(keys, values)
}

pub fn parallel_groupby_std_i64_sorted(keys: &[i64], values: &[i64]) -> PyResult<GroupByResultF64> {
    let mut result = parallel_groupby_std_i64(keys, values)?;
    reorder_single_result_by_key(&mut result);
    Ok(result)
}

pub fn parallel_groupby_std_i64_firstseen_u32(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_std_var_impl::<i64, StdAggI64, f64, u32>(keys, values)
}

pub fn parallel_groupby_std_i64_firstseen_u64(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultF64> {
    parallel_groupby_firstseen_std_var_impl::<i64, StdAggI64, f64, u64>(keys, values)
}

pub fn parallel_groupby_min_i64(keys: &[i64], values: &[i64]) -> PyResult<GroupByResultI64> {
    parallel_groupby::<i64, MinAggI64, i64>(keys, values)
}

pub fn parallel_groupby_min_i64_sorted(keys: &[i64], values: &[i64]) -> PyResult<GroupByResultI64> {
    let mut result = parallel_groupby_min_i64(keys, values)?;
    reorder_single_result_by_key(&mut result);
    Ok(result)
}

pub fn parallel_groupby_min_i64_firstseen_u32(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultI64> {
    parallel_groupby_firstseen_u32::<i64, MinAggI64, i64>(keys, values)
}

pub fn parallel_groupby_min_i64_firstseen_u64(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultI64> {
    parallel_groupby_firstseen_u64::<i64, MinAggI64, i64>(keys, values)
}

pub fn parallel_groupby_max_i64(keys: &[i64], values: &[i64]) -> PyResult<GroupByResultI64> {
    parallel_groupby::<i64, MaxAggI64, i64>(keys, values)
}

pub fn parallel_groupby_max_i64_sorted(keys: &[i64], values: &[i64]) -> PyResult<GroupByResultI64> {
    let mut result = parallel_groupby_max_i64(keys, values)?;
    reorder_single_result_by_key(&mut result);
    Ok(result)
}

pub fn parallel_groupby_max_i64_firstseen_u32(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultI64> {
    parallel_groupby_firstseen_u32::<i64, MaxAggI64, i64>(keys, values)
}

pub fn parallel_groupby_max_i64_firstseen_u64(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultI64> {
    parallel_groupby_firstseen_u64::<i64, MaxAggI64, i64>(keys, values)
}

pub fn parallel_groupby_count_f64(keys: &[i64], values: &[f64]) -> PyResult<GroupByResultI64> {
    parallel_groupby::<f64, CountAggF64, i64>(keys, values)
}

pub fn parallel_groupby_count_f64_sorted(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultI64> {
    let mut result = parallel_groupby_count_f64(keys, values)?;
    reorder_single_result_by_key(&mut result);
    Ok(result)
}

pub fn parallel_groupby_count_f64_firstseen_u32(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultI64> {
    parallel_groupby_firstseen_u32::<f64, CountAggF64, i64>(keys, values)
}

pub fn parallel_groupby_count_f64_firstseen_u64(
    keys: &[i64],
    values: &[f64],
) -> PyResult<GroupByResultI64> {
    parallel_groupby_firstseen_u64::<f64, CountAggF64, i64>(keys, values)
}

pub fn parallel_groupby_count_i64(keys: &[i64], values: &[i64]) -> PyResult<GroupByResultI64> {
    parallel_groupby::<i64, CountAggI64, i64>(keys, values)
}

pub fn parallel_groupby_count_i64_sorted(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultI64> {
    let mut result = parallel_groupby_count_i64(keys, values)?;
    reorder_single_result_by_key(&mut result);
    Ok(result)
}

pub fn parallel_groupby_count_i64_firstseen_u32(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultI64> {
    parallel_groupby_firstseen_u32::<i64, CountAggI64, i64>(keys, values)
}

pub fn parallel_groupby_count_i64_firstseen_u64(
    keys: &[i64],
    values: &[i64],
) -> PyResult<GroupByResultI64> {
    parallel_groupby_firstseen_u64::<i64, CountAggI64, i64>(keys, values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::ThreadPoolBuilder;

    fn make_sensitive_single_key_float_data() -> (Vec<i64>, Vec<f64>) {
        let n = 260_000usize;
        let mut keys = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n);
        for i in 0..n {
            keys.push((i % 257) as i64);
            let mut v = match i % 4 {
                0 => 1e16,
                1 => 1.0,
                2 => -1e16,
                _ => 0.25,
            };
            if i % 997 == 0 {
                v = f64::NAN;
            }
            if i % 991 == 0 {
                v = -0.0;
            }
            values.push(v);
        }
        (keys, values)
    }

    fn make_partitioned_single_key_float_data() -> (Vec<i64>, Vec<f64>) {
        let n_groups = 10_000usize;
        let mut keys = Vec::with_capacity(n_groups * 2);
        let mut values = Vec::with_capacity(n_groups * 2);

        for i in 0..n_groups {
            let key = ((i * 8_191) % n_groups) as i64;
            let base = i as f64;
            keys.push(key);
            keys.push(key);
            values.push(base);
            values.push(base + 0.5);
        }

        (keys, values)
    }

    fn fingerprint_by_key(result: &GroupByResultF64) -> Vec<(i64, u64)> {
        let mut out: Vec<(i64, u64)> = result
            .keys
            .iter()
            .zip(result.values.iter())
            .map(|(&k, &v)| (k, v.to_bits()))
            .collect();
        out.sort_unstable_by_key(|(k, _)| *k);
        out
    }

    fn assert_float_kernel_bitwise_deterministic(
        kernel: fn(&[i64], &[f64]) -> PyResult<GroupByResultF64>,
        keys: &[i64],
        values: &[f64],
    ) {
        let baseline = {
            let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
            let result = pool.install(|| kernel(keys, values).unwrap());
            fingerprint_by_key(&result)
        };

        for &threads in &[2usize, 4, 8, 16] {
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let result = pool.install(|| kernel(keys, values).unwrap());
            assert_eq!(
                fingerprint_by_key(&result),
                baseline,
                "bitwise mismatch at thread count {threads}"
            );
        }
    }

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
    fn test_groupby_prod_f64_nan_and_arithmetic_nan_semantics() {
        let keys = vec![1, 1, 1, 2, 2, 3, 3];
        let values = vec![2.0, f64::NAN, 3.0, f64::NAN, f64::NAN, f64::INFINITY, 0.0];
        let result = parallel_groupby_prod_f64(&keys, &values).unwrap();

        let mut map: AHashMap<i64, f64> = AHashMap::new();
        for (k, v) in result.keys.iter().zip(result.values.iter()) {
            map.insert(*k, *v);
        }

        assert_eq!(map[&1], 6.0);
        assert_eq!(map[&2], 1.0);
        assert!(map[&3].is_nan());
    }

    #[test]
    fn test_groupby_prod_f64_sorted_and_firstseen() {
        let keys = vec![3, 1, 2, 1, 3];
        let values = vec![2.0, 10.0, 100.0, 0.5, 4.0];

        let sorted = parallel_groupby_prod_f64_sorted(&keys, &values).unwrap();
        assert_eq!(sorted.keys, vec![1, 2, 3]);
        assert_eq!(sorted.values, vec![5.0, 100.0, 8.0]);

        let firstseen = parallel_groupby_prod_f64_firstseen_u32(&keys, &values).unwrap();
        assert_eq!(firstseen.keys, vec![3, 1, 2]);
        assert_eq!(firstseen.values, vec![8.0, 5.0, 100.0]);
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
    fn test_groupby_sorted_orders_by_key() {
        let keys = vec![3, 1, 2, 1, 3];
        let values = vec![1.0, 10.0, 100.0, 1.0, 2.0];
        let result = parallel_groupby_sum_f64_sorted(&keys, &values).unwrap();

        assert_eq!(result.keys, vec![1, 2, 3]);
        assert_eq!(result.values.len(), 3);
        assert!((result.values[0] - 11.0).abs() < 1e-10);
        assert!((result.values[1] - 100.0).abs() < 1e-10);
        assert!((result.values[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_groupby_sorted_orders_negative_keys() {
        let keys = vec![0, -1, 2, -3, -1];
        let values = vec![1.0, 1.0, 1.0, 1.0, 2.0];
        let result = parallel_groupby_sum_f64_sorted(&keys, &values).unwrap();

        assert_eq!(result.keys, vec![-3, -1, 0, 2]);
        assert_eq!(result.values.len(), 4);
        assert!((result.values[0] - 1.0).abs() < 1e-10);
        assert!((result.values[1] - 3.0).abs() < 1e-10);
        assert!((result.values[2] - 1.0).abs() < 1e-10);
        assert!((result.values[3] - 1.0).abs() < 1e-10);
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

        let mut map: AHashMap<i64, i64> = AHashMap::new();
        for (k, v) in result.keys.iter().zip(result.values.iter()) {
            map.insert(*k, *v);
        }

        assert_eq!(map[&1], 9);
        assert_eq!(map[&2], 6);
    }

    #[test]
    fn test_groupby_prod_i64_wraps_and_firstseen_u64() {
        let keys = vec![9, 1, 9, 1, 2];
        let values = vec![i64::MAX, 3, 2, 4, 5];
        let result = parallel_groupby_prod_i64_firstseen_u64(&keys, &values).unwrap();

        assert_eq!(result.keys, vec![9, 1, 2]);
        assert_eq!(result.values[0], i64::MAX.wrapping_mul(2));
        assert_eq!(result.values[1], 12);
        assert_eq!(result.values[2], 5);

        let sorted = parallel_groupby_prod_i64_sorted(&keys, &values).unwrap();
        assert_eq!(sorted.keys, vec![1, 2, 9]);
        assert_eq!(sorted.values, vec![12, 5, i64::MAX.wrapping_mul(2)]);
    }

    #[test]
    fn test_groupby_min_i64() {
        let keys = vec![1, 2, 1, 2, 1];
        let values: Vec<i64> = vec![5, 2, 3, 4, 1];
        let result = parallel_groupby_min_i64(&keys, &values).unwrap();

        let mut map: AHashMap<i64, i64> = AHashMap::new();
        for (k, v) in result.keys.iter().zip(result.values.iter()) {
            map.insert(*k, *v);
        }

        assert_eq!(map[&1], 1);
        assert_eq!(map[&2], 2);
    }

    #[test]
    fn test_groupby_max_i64() {
        let keys = vec![1, 2, 1, 2, 1];
        let values: Vec<i64> = vec![5, 2, 3, 4, 1];
        let result = parallel_groupby_max_i64(&keys, &values).unwrap();

        let mut map: AHashMap<i64, i64> = AHashMap::new();
        for (k, v) in result.keys.iter().zip(result.values.iter()) {
            map.insert(*k, *v);
        }

        assert_eq!(map[&1], 5);
        assert_eq!(map[&2], 4);
    }

    #[test]
    fn test_groupby_count_f64() {
        let keys = vec![1, 2, 1, 2, 1];
        let values = vec![1.0, 2.0, f64::NAN, 4.0, 5.0];
        let result = parallel_groupby_count_f64(&keys, &values).unwrap();

        let mut map: AHashMap<i64, i64> = AHashMap::new();
        for (k, v) in result.keys.iter().zip(result.values.iter()) {
            map.insert(*k, *v);
        }

        assert_eq!(map[&1], 2);
        assert_eq!(map[&2], 2);
    }

    #[test]
    fn test_groupby_count_i64() {
        let keys = vec![1, 2, 1, 2, 1];
        let values: Vec<i64> = vec![1, 2, 3, 4, 5];
        let result = parallel_groupby_count_i64(&keys, &values).unwrap();

        let mut map: AHashMap<i64, i64> = AHashMap::new();
        for (k, v) in result.keys.iter().zip(result.values.iter()) {
            map.insert(*k, *v);
        }

        assert_eq!(map[&1], 3);
        assert_eq!(map[&2], 2);
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

    #[test]
    fn test_sum_f64_sorted_bitwise_deterministic_across_threads() {
        let (keys, values) = make_sensitive_single_key_float_data();
        assert_float_kernel_bitwise_deterministic(parallel_groupby_sum_f64_sorted, &keys, &values);
    }

    #[test]
    fn test_mean_f64_sorted_bitwise_deterministic_across_threads() {
        let (keys, values) = make_sensitive_single_key_float_data();
        assert_float_kernel_bitwise_deterministic(parallel_groupby_mean_f64_sorted, &keys, &values);
    }

    #[test]
    fn test_sum_f64_firstseen_u32_bitwise_deterministic_across_threads() {
        let (keys, values) = make_sensitive_single_key_float_data();
        assert_float_kernel_bitwise_deterministic(
            parallel_groupby_sum_f64_firstseen_u32,
            &keys,
            &values,
        );
    }

    #[test]
    fn test_mean_f64_firstseen_u32_bitwise_deterministic_across_threads() {
        let (keys, values) = make_sensitive_single_key_float_data();
        assert_float_kernel_bitwise_deterministic(
            parallel_groupby_mean_f64_firstseen_u32,
            &keys,
            &values,
        );
    }

    #[test]
    fn test_sum_f64_firstseen_u64_bitwise_deterministic_across_threads() {
        let (keys, values) = make_sensitive_single_key_float_data();
        assert_float_kernel_bitwise_deterministic(
            parallel_groupby_sum_f64_firstseen_u64,
            &keys,
            &values,
        );
    }

    #[test]
    fn test_mean_f64_firstseen_u64_bitwise_deterministic_across_threads() {
        let (keys, values) = make_sensitive_single_key_float_data();
        assert_float_kernel_bitwise_deterministic(
            parallel_groupby_mean_f64_firstseen_u64,
            &keys,
            &values,
        );
    }

    #[test]
    fn test_var_f64_firstseen_u32_preserves_first_seen_order_and_nan_semantics() {
        let keys = vec![5, 1, 5, 9, 1, 7, 7];
        let values = vec![10.0, 2.0, 14.0, f64::NAN, 4.0, 3.0, 3.0];

        let result = parallel_groupby_var_f64_firstseen_u32(&keys, &values).unwrap();

        assert_eq!(result.keys, vec![5, 1, 9, 7]);
        assert!((result.values[0] - 8.0).abs() < 1e-10);
        assert!((result.values[1] - 2.0).abs() < 1e-10);
        assert!(result.values[2].is_nan());
        assert_eq!(result.values[3].to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn test_std_i64_firstseen_u64_preserves_first_seen_order_and_singleton_nan() {
        let keys = vec![5, 1, 5, 9, 1];
        let values = vec![10_i64, 2, 14, 7, 4];

        let result = parallel_groupby_std_i64_firstseen_u64(&keys, &values).unwrap();

        assert_eq!(result.keys, vec![5, 1, 9]);
        assert!((result.values[0] - (8.0f64).sqrt()).abs() < 1e-10);
        assert!((result.values[1] - (2.0f64).sqrt()).abs() < 1e-10);
        assert!(result.values[2].is_nan());
    }

    #[test]
    fn test_std_var_routing_prefers_legacy_engine_for_low_cardinality_samples() {
        let n = 20_000usize;
        let keys: Vec<i64> = (0..n).map(|i| (i % 1_000) as i64).collect();
        let values: Vec<f64> = (0..n).map(|i| i as f64).collect();

        let profiled = profile_parallel_groupby_std_f64_firstseen_u32(&keys, &values).unwrap();

        assert!(profiled.profile.merge_s > 0.0);
        assert!(profiled.profile.partial_group_total > profiled.profile.final_group_count);
    }

    #[test]
    fn test_std_var_routing_prefers_partitioned_engine_for_high_uniqueness_samples() {
        let n = 20_000usize;
        let keys: Vec<i64> = (0..n).map(|i| i as i64).collect();
        let values: Vec<f64> = (0..n).map(|i| i as f64).collect();

        let profiled = profile_parallel_groupby_std_f64_firstseen_u32(&keys, &values).unwrap();

        assert_eq!(profiled.profile.merge_s, 0.0);
        assert_eq!(
            profiled.profile.partial_group_total,
            profiled.profile.final_group_count
        );
    }

    #[test]
    fn test_sorted_std_var_routing_prefers_legacy_engine_for_low_cardinality_samples() {
        let n = 20_000usize;
        let keys: Vec<i64> = (0..n).map(|i| (i % 1_000) as i64).collect();
        let values: Vec<f64> = (0..n).map(|i| i as f64).collect();

        let profiled = profile_parallel_groupby_std_f64_sorted(&keys, &values).unwrap();

        assert!(profiled.profile.merge_s > 0.0);
        assert!(profiled.profile.partial_group_total > profiled.profile.final_group_count);
    }

    #[test]
    fn test_sorted_std_var_routing_prefers_partitioned_engine_for_high_uniqueness_samples() {
        let n = 20_000usize;
        let keys: Vec<i64> = (0..n).map(|i| i as i64).collect();
        let values: Vec<f64> = (0..n).map(|i| i as f64).collect();

        let profiled = profile_parallel_groupby_std_f64_sorted(&keys, &values).unwrap();

        assert_eq!(profiled.profile.merge_s, 0.0);
        assert_eq!(
            profiled.profile.partial_group_total,
            profiled.profile.final_group_count
        );
    }

    #[test]
    fn test_std_f64_partitioned_sorted_bitwise_deterministic_across_threads() {
        let (keys, values) = make_partitioned_single_key_float_data();
        assert_float_kernel_bitwise_deterministic(parallel_groupby_std_f64_sorted, &keys, &values);
    }

    #[test]
    fn test_var_f64_partitioned_firstseen_u32_bitwise_deterministic_across_threads() {
        let (keys, values) = make_partitioned_single_key_float_data();
        assert_float_kernel_bitwise_deterministic(
            parallel_groupby_var_f64_firstseen_u32,
            &keys,
            &values,
        );
    }

    #[test]
    fn test_var_f64_firstseen_u32_bitwise_deterministic_across_threads() {
        let (keys, values) = make_sensitive_single_key_float_data();
        assert_float_kernel_bitwise_deterministic(
            parallel_groupby_var_f64_firstseen_u32,
            &keys,
            &values,
        );
    }

    #[test]
    fn test_std_f64_firstseen_u32_bitwise_deterministic_across_threads() {
        let (keys, values) = make_sensitive_single_key_float_data();
        assert_float_kernel_bitwise_deterministic(
            parallel_groupby_std_f64_firstseen_u32,
            &keys,
            &values,
        );
    }

    #[test]
    fn test_var_f64_firstseen_u64_bitwise_deterministic_across_threads() {
        let (keys, values) = make_sensitive_single_key_float_data();
        assert_float_kernel_bitwise_deterministic(
            parallel_groupby_var_f64_firstseen_u64,
            &keys,
            &values,
        );
    }

    #[test]
    fn test_std_f64_firstseen_u64_bitwise_deterministic_across_threads() {
        let (keys, values) = make_sensitive_single_key_float_data();
        assert_float_kernel_bitwise_deterministic(
            parallel_groupby_std_f64_firstseen_u64,
            &keys,
            &values,
        );
    }
}
