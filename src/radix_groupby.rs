//! Radix Partitioning Engine for high-performance multi-key groupby.
//!
//! This module implements a merge-free, shared-nothing parallel groupby using radix partitioning.
//! The key insight is that by partitioning data by hash prefix BEFORE aggregation, each partition
//! can be processed independently without any cross-thread synchronization or merge overhead.
//!
//! ## Algorithm Overview
//!
//! 1. **Histogram**: Count rows per partition (parallel prefix count)
//! 2. **Prefix Sum**: Compute write offsets for each partition
//! 3. **Scatter**: Reorder data by partition (parallel scatter with atomic offsets)
//! 4. **Aggregate**: Process each partition independently (no merge needed!)
//!
//! ## Why Radix Beats Fold/Reduce
//!
//! The current fold/reduce approach has O(G × log(T)) merge cost where G = groups, T = threads.
//! With high cardinality (G >> 10M), merge dominates. Radix eliminates merge entirely.

use ahash::AHashMap;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};

use crate::aggregation::{
    Aggregator, CountAggF64, CountAggI64, MaxAggF64, MaxAggI64, MeanAggF64, MeanAggI64, MinAggF64,
    MinAggI64, SumAggF64, SumAggI64,
};
use crate::radix_sort::{
    i64_to_sortable_u64, radix_sort_perm_by_u32, radix_sort_perm_by_u64,
    radix_sort_perm_by_u64_for_indices_par,
};

const RADIX_SORT_THRESHOLD: usize = 2048;
pub(crate) const SMALL_DIRECT_THRESHOLD_ELEMS: usize = 200_000;

// CompositeKey is a fallback path. For the supported Python API (<= 10 key columns),
// we specialize on FixedKey<N> for N=1..=10.
const INLINE_KEYS: usize = 10;
const NUM_PARTITIONS_BITS: usize = 10;
const NUM_PARTITIONS: usize = 1 << NUM_PARTITIONS_BITS;

type PartitionResults<K, I, V> = Vec<(Vec<(K, V)>, Vec<I>)>;

#[derive(Clone, Copy)]
struct PermPtr(*mut usize);

unsafe impl Send for PermPtr {}
unsafe impl Sync for PermPtr {}

impl PermPtr {
    #[inline]
    unsafe fn write(self, pos: usize, row: usize) {
        // SAFETY: callers must ensure `pos < perm.len()` and that each `pos` is written
        // exactly once (no concurrent writes to the same location).
        *self.0.add(pos) = row;
    }
}

#[derive(Clone, Debug)]
pub struct CompositeKey(SmallVec<[i64; INLINE_KEYS]>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FixedKey<const N: usize>(pub [i64; N]);

impl CompositeKey {
    #[inline]
    fn with_capacity(cap: usize) -> Self {
        Self(SmallVec::with_capacity(cap))
    }

    #[inline]
    fn push(&mut self, val: i64) {
        self.0.push(val);
    }

    #[inline]
    fn iter(&self) -> impl Iterator<Item = &i64> {
        self.0.iter()
    }
}

impl PartialEq for CompositeKey {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for CompositeKey {}

impl Hash for CompositeKey {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        for &v in &self.0 {
            v.hash(state);
        }
    }
}

impl PartialOrd for CompositeKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CompositeKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.iter().cmp(other.0.iter())
    }
}

#[derive(Debug)]
pub struct GroupByMultiResult<V> {
    /// Row-major flat keys: keys for group g occupy `keys_flat[g*n_keys..(g+1)*n_keys]`.
    pub keys_flat: Vec<i64>,
    pub n_keys: usize,
    pub values: Vec<V>,
    /// Output ordering permutation.
    ///
    /// If present, output group at position `out_g` is sourced from group `perm[out_g]`.
    /// If None, `keys_flat`/`values` are already materialized in output order,
    /// so identity mapping is implied and consumers must not apply any additional permutation.
    pub perm: Option<Vec<usize>>,
}

// -----------------------------------------------------------------------------
// First-seen ordering utilities
// -----------------------------------------------------------------------------

fn reorder_result_by_first_seen_u32<V: Copy>(
    result: &mut GroupByMultiResult<V>,
    first_seen: &[u32],
) {
    if result.values.is_empty() {
        return;
    }
    debug_assert_eq!(result.values.len(), first_seen.len());

    let n_keys = result.n_keys;
    let n_groups = result.values.len();
    debug_assert_eq!(result.keys_flat.len(), n_groups * n_keys);

    let perm = radix_sort_perm_by_u32(first_seen);

    let keys_flat = &result.keys_flat;
    let values = &result.values;

    if n_groups.saturating_mul(n_keys) > SMALL_DIRECT_THRESHOLD_ELEMS {
        result.perm = Some(perm);
        return;
    }

    let mut sorted_keys = Vec::with_capacity(keys_flat.len());
    let mut sorted_values = Vec::with_capacity(values.len());

    for &g in &perm {
        sorted_keys.extend_from_slice(&keys_flat[g * n_keys..(g + 1) * n_keys]);
        sorted_values.push(values[g]);
    }

    result.keys_flat = sorted_keys;
    result.values = sorted_values;
    result.perm = None;
}

fn reorder_result_by_first_seen_u64<V: Copy>(
    result: &mut GroupByMultiResult<V>,
    first_seen: &[u64],
) {
    if result.values.is_empty() {
        return;
    }
    debug_assert_eq!(result.values.len(), first_seen.len());

    let n_keys = result.n_keys;
    let n_groups = result.values.len();
    debug_assert_eq!(result.keys_flat.len(), n_groups * n_keys);

    let perm = radix_sort_perm_by_u64(first_seen);

    let keys_flat = &result.keys_flat;
    let values = &result.values;

    if n_groups.saturating_mul(n_keys) > SMALL_DIRECT_THRESHOLD_ELEMS {
        result.perm = Some(perm);
        return;
    }

    let mut sorted_keys = Vec::with_capacity(keys_flat.len());
    let mut sorted_values = Vec::with_capacity(values.len());

    for &g in &perm {
        sorted_keys.extend_from_slice(&keys_flat[g * n_keys..(g + 1) * n_keys]);
        sorted_values.push(values[g]);
    }

    result.keys_flat = sorted_keys;
    result.values = sorted_values;
    result.perm = None;
}

#[inline]
fn compute_hash(key_slices: &[&[i64]], row: usize) -> u64 {
    use ahash::AHasher;
    let mut hasher = AHasher::default();
    for col in key_slices {
        col[row].hash(&mut hasher);
    }
    hasher.finish()
}

#[inline]
fn hash_to_partition(hash: u64) -> usize {
    (hash as usize) & (NUM_PARTITIONS - 1)
}

fn stable_scatter_by_partition(hashes: &[u64]) -> (Vec<usize>, Vec<usize>) {
    let n_rows = hashes.len();
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_size = (n_rows / n_threads).max(1024);

    let counts: Vec<[usize; NUM_PARTITIONS]> = hashes
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut local = [0usize; NUM_PARTITIONS];
            for &h in chunk {
                local[hash_to_partition(h)] += 1;
            }
            local
        })
        .collect();

    let mut offsets = vec![0usize; NUM_PARTITIONS + 1];
    for local in &counts {
        for p in 0..NUM_PARTITIONS {
            offsets[p + 1] += local[p];
        }
    }
    for p in 0..NUM_PARTITIONS {
        offsets[p + 1] += offsets[p];
    }

    let mut thread_offsets: Vec<[usize; NUM_PARTITIONS]> =
        vec![[0usize; NUM_PARTITIONS]; counts.len()];
    let mut running = [0usize; NUM_PARTITIONS];
    running[..NUM_PARTITIONS].copy_from_slice(&offsets[..NUM_PARTITIONS]);
    for (chunk_idx, local) in counts.iter().enumerate() {
        thread_offsets[chunk_idx] = running;
        for p in 0..NUM_PARTITIONS {
            running[p] += local[p];
        }
    }

    let mut perm = vec![0usize; n_rows];
    #[cfg(debug_assertions)]
    perm.fill(usize::MAX);
    let perm_ptr = PermPtr(perm.as_mut_ptr());

    hashes
        .par_chunks(chunk_size)
        .enumerate()
        .zip(thread_offsets.into_par_iter())
        .for_each(|((chunk_idx, chunk), mut write)| {
            let base = chunk_idx * chunk_size;
            for (i, &h) in chunk.iter().enumerate() {
                let row = base + i;
                let p = hash_to_partition(h);
                let pos = write[p];
                write[p] = pos + 1;
                // SAFETY:
                // - `pos` is unique within this chunk due to local `write[p]` increments.
                // - Chunk starting offsets are disjoint across chunks.
                unsafe { perm_ptr.write(pos, row) };
            }
        });

    #[cfg(debug_assertions)]
    {
        debug_assert!(perm.iter().all(|&r| r != usize::MAX));
        debug_assert!(perm.iter().all(|&r| r < n_rows));
    }

    (perm, offsets)
}

#[inline]
fn extract_key(key_slices: &[&[i64]], row: usize) -> CompositeKey {
    let mut key = CompositeKey::with_capacity(key_slices.len());
    for col in key_slices {
        key.push(col[row]);
    }
    key
}

#[inline]
fn compute_hash_fixed<const N: usize>(key_slices: &[&[i64]], row: usize) -> u64 {
    use ahash::AHasher;
    let mut hasher = AHasher::default();
    for col in &key_slices[..N] {
        col[row].hash(&mut hasher);
    }
    hasher.finish()
}

#[inline]
fn extract_key_fixed<const N: usize>(key_slices: &[&[i64]], row: usize) -> FixedKey<N> {
    let mut arr = [0i64; N];
    for (dst, col) in arr.iter_mut().zip(&key_slices[..N]) {
        *dst = col[row];
    }
    FixedKey(arr)
}

// -----------------------------------------------------------------------------
// Key Operations Trait & Implementations
// -----------------------------------------------------------------------------

trait RadixKeyOps: Send + Sync + Copy + 'static {
    type Key: Eq + Hash + Clone + Send + Sync + Debug;

    fn new(n_keys: usize) -> Self;
    fn n_keys(&self) -> usize;
    fn compute_hash(&self, key_slices: &[&[i64]], row: usize) -> u64;
    fn extract_key(&self, key_slices: &[&[i64]], row: usize) -> Self::Key;
    fn push_flat(keys_flat: &mut Vec<i64>, key: &Self::Key);
}

#[derive(Clone, Copy)]
struct FixedKeyOps<const N: usize>;

impl<const N: usize> RadixKeyOps for FixedKeyOps<N> {
    type Key = FixedKey<N>;

    #[inline(always)]
    fn new(_: usize) -> Self {
        Self
    }

    #[inline(always)]
    fn n_keys(&self) -> usize {
        N
    }

    #[inline(always)]
    fn compute_hash(&self, key_slices: &[&[i64]], row: usize) -> u64 {
        compute_hash_fixed::<N>(key_slices, row)
    }

    #[inline(always)]
    fn extract_key(&self, key_slices: &[&[i64]], row: usize) -> Self::Key {
        extract_key_fixed::<N>(key_slices, row)
    }

    #[inline(always)]
    fn push_flat(keys_flat: &mut Vec<i64>, key: &Self::Key) {
        keys_flat.extend_from_slice(&key.0);
    }
}

#[derive(Clone, Copy)]
struct CompositeKeyOps {
    n_keys: usize,
}

impl RadixKeyOps for CompositeKeyOps {
    type Key = CompositeKey;

    #[inline(always)]
    fn new(n: usize) -> Self {
        Self { n_keys: n }
    }

    #[inline(always)]
    fn n_keys(&self) -> usize {
        self.n_keys
    }

    #[inline(always)]
    fn compute_hash(&self, key_slices: &[&[i64]], row: usize) -> u64 {
        compute_hash(key_slices, row)
    }

    #[inline(always)]
    fn extract_key(&self, key_slices: &[&[i64]], row: usize) -> Self::Key {
        extract_key(key_slices, row)
    }

    #[inline(always)]
    fn push_flat(keys_flat: &mut Vec<i64>, key: &Self::Key) {
        keys_flat.extend(key.iter().copied());
    }
}

// -----------------------------------------------------------------------------
// Unified Engine
// -----------------------------------------------------------------------------

fn radix_groupby_engine<Ops, T, A, O>(
    ops: Ops,
    key_slices: &[&[i64]],
    values: &[T],
) -> Result<GroupByMultiResult<O>, String>
where
    Ops: RadixKeyOps,
    T: Copy + Send + Sync,
    O: Copy + Send + Sync,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    let n_rows = values.len();

    // Safety check: ensure all key columns have the same length as values
    for (i, col) in key_slices.iter().enumerate() {
        if col.len() != n_rows {
            return Err(format!(
                "Key column {} has length {}, expected {}",
                i,
                col.len(),
                n_rows
            ));
        }
    }

    if n_rows == 0 {
        let n_keys = ops.n_keys();
        return Ok(GroupByMultiResult {
            keys_flat: Vec::new(),
            n_keys,
            values: Vec::new(),
            perm: None,
        });
    }

    // Phase 1: compute and cache hashes
    let mut hashes = vec![0u64; n_rows];
    hashes.par_iter_mut().enumerate().for_each(|(row, h_out)| {
        let h = ops.compute_hash(key_slices, row);
        *h_out = h;
    });

    // Phase 2/3: deterministic stable scatter by partition.
    let (perm, offsets) = stable_scatter_by_partition(&hashes);

    // Phase 4: Per-partition aggregation (no merge!)
    let partition_results: Vec<Vec<(Ops::Key, O)>> = (0..NUM_PARTITIONS)
        .into_par_iter()
        .map(|p| {
            let start = offsets[p];
            let end = offsets[p + 1];
            if start == end {
                return Vec::new();
            }

            let mut local_map: AHashMap<Ops::Key, A> = AHashMap::new();
            for &row in &perm[start..end] {
                let key = ops.extract_key(key_slices, row);
                let val = values[row];
                local_map.entry(key).or_insert_with(A::init).update(val);
            }

            local_map
                .into_iter()
                .map(|(k, agg)| (k, agg.finalize()))
                .collect()
        })
        .collect();

    // Flatten results
    let total_groups: usize = partition_results.iter().map(|v| v.len()).sum();
    let n_keys = ops.n_keys();
    let mut keys_flat = Vec::with_capacity(total_groups * n_keys);
    let mut out_values: Vec<O> = Vec::with_capacity(total_groups);

    for partition in partition_results {
        for (key, val) in partition {
            Ops::push_flat(&mut keys_flat, &key);
            out_values.push(val);
        }
    }

    Ok(GroupByMultiResult {
        keys_flat,
        n_keys,
        values: out_values,
        perm: None,
    })
}

// -----------------------------------------------------------------------------
// First-seen radix engines (idx32 / idx64)
// -----------------------------------------------------------------------------

fn radix_groupby_engine_firstseen_u32<Ops, T, A, O>(
    ops: Ops,
    key_slices: &[&[i64]],
    values: &[T],
) -> Result<GroupByMultiResult<O>, String>
where
    Ops: RadixKeyOps,
    T: Copy + Send + Sync,
    O: Copy + Send + Sync,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    let n_rows = values.len();

    for (i, col) in key_slices.iter().enumerate() {
        if col.len() != n_rows {
            return Err(format!(
                "Key column {} has length {}, expected {}",
                i,
                col.len(),
                n_rows
            ));
        }
    }

    if n_rows == 0 {
        let n_keys = ops.n_keys();
        return Ok(GroupByMultiResult {
            keys_flat: Vec::new(),
            n_keys,
            values: Vec::new(),
            perm: None,
        });
    }

    if n_rows > (u32::MAX as usize) {
        return Err("idx32 kernel requires n_rows <= u32::MAX".to_string());
    }

    // Phase 1: compute and cache hashes
    let mut hashes = vec![0u64; n_rows];

    hashes.par_iter_mut().enumerate().for_each(|(row, h_out)| {
        let h = ops.compute_hash(key_slices, row);
        *h_out = h;
    });

    // Phase 2/3: deterministic stable scatter by partition.
    let (perm, offsets) = stable_scatter_by_partition(&hashes);

    // Phase 4: Per-partition aggregation (gid map + SoA state)
    let partition_results: PartitionResults<Ops::Key, u32, O> = (0..NUM_PARTITIONS)
        .into_par_iter()
        .map(|p| {
            let start = offsets[p];
            let end = offsets[p + 1];
            if start == end {
                return (Vec::new(), Vec::new());
            }

            let mut gid_map: AHashMap<Ops::Key, u32> = AHashMap::new();
            let mut aggs: Vec<A> = Vec::new();
            let mut first_seen: Vec<u32> = Vec::new();

            for &row in &perm[start..end] {
                let key = ops.extract_key(key_slices, row);
                let row_u32 = row as u32;
                let val = values[row];

                if let Some(&gid) = gid_map.get(&key) {
                    let g = gid as usize;
                    if row_u32 < first_seen[g] {
                        first_seen[g] = row_u32;
                    }
                    aggs[g].update(val);
                } else {
                    let gid = aggs.len() as u32;
                    gid_map.insert(key, gid);
                    let mut agg = A::init();
                    agg.update(val);
                    aggs.push(agg);
                    first_seen.push(row_u32);
                }
            }

            let mut out_pairs: Vec<(Ops::Key, O)> = Vec::with_capacity(aggs.len());
            let mut out_first_seen: Vec<u32> = Vec::with_capacity(aggs.len());
            for (k, gid) in gid_map {
                let g = gid as usize;
                out_pairs.push((k, aggs[g].finalize()));
                out_first_seen.push(first_seen[g]);
            }

            (out_pairs, out_first_seen)
        })
        .collect();

    let total_groups: usize = partition_results.iter().map(|(pairs, _)| pairs.len()).sum();
    let n_keys = ops.n_keys();
    let mut keys_flat = Vec::with_capacity(total_groups * n_keys);
    let mut out_values: Vec<O> = Vec::with_capacity(total_groups);
    let mut out_first_seen = Vec::with_capacity(total_groups);

    for (pairs, firsts) in partition_results {
        debug_assert_eq!(pairs.len(), firsts.len());
        for ((key, val), first) in pairs.into_iter().zip(firsts.into_iter()) {
            Ops::push_flat(&mut keys_flat, &key);
            out_values.push(val);
            out_first_seen.push(first);
        }
    }

    let mut result = GroupByMultiResult {
        keys_flat,
        n_keys,
        values: out_values,
        perm: None,
    };
    reorder_result_by_first_seen_u32(&mut result, &out_first_seen);
    Ok(result)
}

fn radix_groupby_engine_firstseen_u64<Ops, T, A, O>(
    ops: Ops,
    key_slices: &[&[i64]],
    values: &[T],
) -> Result<GroupByMultiResult<O>, String>
where
    Ops: RadixKeyOps,
    T: Copy + Send + Sync,
    O: Copy + Send + Sync,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    let n_rows = values.len();

    for (i, col) in key_slices.iter().enumerate() {
        if col.len() != n_rows {
            return Err(format!(
                "Key column {} has length {}, expected {}",
                i,
                col.len(),
                n_rows
            ));
        }
    }

    if n_rows == 0 {
        let n_keys = ops.n_keys();
        return Ok(GroupByMultiResult {
            keys_flat: Vec::new(),
            n_keys,
            values: Vec::new(),
            perm: None,
        });
    }

    // Phase 1: compute and cache hashes
    let mut hashes = vec![0u64; n_rows];

    hashes.par_iter_mut().enumerate().for_each(|(row, h_out)| {
        let h = ops.compute_hash(key_slices, row);
        *h_out = h;
    });

    // Phase 2/3: deterministic stable scatter by partition.
    let (perm, offsets) = stable_scatter_by_partition(&hashes);

    // Phase 4: Per-partition aggregation (gid map + SoA state)
    let partition_results: PartitionResults<Ops::Key, u64, O> = (0..NUM_PARTITIONS)
        .into_par_iter()
        .map(|p| {
            let start = offsets[p];
            let end = offsets[p + 1];
            if start == end {
                return (Vec::new(), Vec::new());
            }

            let mut gid_map: AHashMap<Ops::Key, u32> = AHashMap::new();
            let mut aggs: Vec<A> = Vec::new();
            let mut first_seen: Vec<u64> = Vec::new();

            for &row in &perm[start..end] {
                let key = ops.extract_key(key_slices, row);
                let row_u64 = row as u64;
                let val = values[row];

                if let Some(&gid) = gid_map.get(&key) {
                    let g = gid as usize;
                    if row_u64 < first_seen[g] {
                        first_seen[g] = row_u64;
                    }
                    aggs[g].update(val);
                } else {
                    let gid = aggs.len() as u32;
                    gid_map.insert(key, gid);
                    let mut agg = A::init();
                    agg.update(val);
                    aggs.push(agg);
                    first_seen.push(row_u64);
                }
            }

            let mut out_pairs: Vec<(Ops::Key, O)> = Vec::with_capacity(aggs.len());
            let mut out_first_seen: Vec<u64> = Vec::with_capacity(aggs.len());
            for (k, gid) in gid_map {
                let g = gid as usize;
                out_pairs.push((k, aggs[g].finalize()));
                out_first_seen.push(first_seen[g]);
            }

            (out_pairs, out_first_seen)
        })
        .collect();

    let total_groups: usize = partition_results.iter().map(|(pairs, _)| pairs.len()).sum();
    let n_keys = ops.n_keys();
    let mut keys_flat = Vec::with_capacity(total_groups * n_keys);
    let mut out_values: Vec<O> = Vec::with_capacity(total_groups);
    let mut out_first_seen = Vec::with_capacity(total_groups);

    for (pairs, firsts) in partition_results {
        debug_assert_eq!(pairs.len(), firsts.len());
        for ((key, val), first) in pairs.into_iter().zip(firsts.into_iter()) {
            Ops::push_flat(&mut keys_flat, &key);
            out_values.push(val);
            out_first_seen.push(first);
        }
    }

    let mut result = GroupByMultiResult {
        keys_flat,
        n_keys,
        values: out_values,
        perm: None,
    };
    reorder_result_by_first_seen_u64(&mut result, &out_first_seen);
    Ok(result)
}

// -----------------------------------------------------------------------------
// Public Wrappers (Internal Dispatch)
// -----------------------------------------------------------------------------

fn radix_groupby_fixed<const N: usize, T, A, O>(
    key_slices: &[&[i64]],
    values: &[T],
) -> Result<GroupByMultiResult<O>, String>
where
    T: Copy + Send + Sync,
    O: Copy + Send + Sync,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    radix_groupby_engine::<FixedKeyOps<N>, T, A, O>(FixedKeyOps::<N>::new(N), key_slices, values)
}

fn radix_groupby<T, A, O>(
    key_slices: &[&[i64]],
    values: &[T],
) -> Result<GroupByMultiResult<O>, String>
where
    T: Copy + Send + Sync,
    O: Copy + Send + Sync,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    radix_groupby_engine::<CompositeKeyOps, T, A, O>(
        CompositeKeyOps::new(key_slices.len()),
        key_slices,
        values,
    )
}

fn radix_groupby_dispatch<T, A, O>(
    key_slices: &[&[i64]],
    values: &[T],
) -> Result<GroupByMultiResult<O>, String>
where
    T: Copy + Send + Sync,
    O: Copy + Send + Sync,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    match key_slices.len() {
        1 => radix_groupby_fixed::<1, T, A, O>(key_slices, values),
        2 => radix_groupby_fixed::<2, T, A, O>(key_slices, values),
        3 => radix_groupby_fixed::<3, T, A, O>(key_slices, values),
        4 => radix_groupby_fixed::<4, T, A, O>(key_slices, values),
        5 => radix_groupby_fixed::<5, T, A, O>(key_slices, values),
        6 => radix_groupby_fixed::<6, T, A, O>(key_slices, values),
        7 => radix_groupby_fixed::<7, T, A, O>(key_slices, values),
        8 => radix_groupby_fixed::<8, T, A, O>(key_slices, values),
        9 => radix_groupby_fixed::<9, T, A, O>(key_slices, values),
        10 => radix_groupby_fixed::<10, T, A, O>(key_slices, values),
        _ => radix_groupby::<T, A, O>(key_slices, values),
    }
}

fn radix_groupby_dispatch_firstseen_u32<T, A, O>(
    key_slices: &[&[i64]],
    values: &[T],
) -> Result<GroupByMultiResult<O>, String>
where
    T: Copy + Send + Sync,
    O: Copy + Send + Sync,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    match key_slices.len() {
        1 => radix_groupby_engine_firstseen_u32::<FixedKeyOps<1>, T, A, O>(
            FixedKeyOps::<1>::new(1),
            key_slices,
            values,
        ),
        2 => radix_groupby_engine_firstseen_u32::<FixedKeyOps<2>, T, A, O>(
            FixedKeyOps::<2>::new(2),
            key_slices,
            values,
        ),
        3 => radix_groupby_engine_firstseen_u32::<FixedKeyOps<3>, T, A, O>(
            FixedKeyOps::<3>::new(3),
            key_slices,
            values,
        ),
        4 => radix_groupby_engine_firstseen_u32::<FixedKeyOps<4>, T, A, O>(
            FixedKeyOps::<4>::new(4),
            key_slices,
            values,
        ),
        5 => radix_groupby_engine_firstseen_u32::<FixedKeyOps<5>, T, A, O>(
            FixedKeyOps::<5>::new(5),
            key_slices,
            values,
        ),
        6 => radix_groupby_engine_firstseen_u32::<FixedKeyOps<6>, T, A, O>(
            FixedKeyOps::<6>::new(6),
            key_slices,
            values,
        ),
        7 => radix_groupby_engine_firstseen_u32::<FixedKeyOps<7>, T, A, O>(
            FixedKeyOps::<7>::new(7),
            key_slices,
            values,
        ),
        8 => radix_groupby_engine_firstseen_u32::<FixedKeyOps<8>, T, A, O>(
            FixedKeyOps::<8>::new(8),
            key_slices,
            values,
        ),
        9 => radix_groupby_engine_firstseen_u32::<FixedKeyOps<9>, T, A, O>(
            FixedKeyOps::<9>::new(9),
            key_slices,
            values,
        ),
        10 => radix_groupby_engine_firstseen_u32::<FixedKeyOps<10>, T, A, O>(
            FixedKeyOps::<10>::new(10),
            key_slices,
            values,
        ),
        _ => radix_groupby_engine_firstseen_u32::<CompositeKeyOps, T, A, O>(
            CompositeKeyOps::new(key_slices.len()),
            key_slices,
            values,
        ),
    }
}

fn radix_groupby_dispatch_firstseen_u64<T, A, O>(
    key_slices: &[&[i64]],
    values: &[T],
) -> Result<GroupByMultiResult<O>, String>
where
    T: Copy + Send + Sync,
    O: Copy + Send + Sync,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    match key_slices.len() {
        1 => radix_groupby_engine_firstseen_u64::<FixedKeyOps<1>, T, A, O>(
            FixedKeyOps::<1>::new(1),
            key_slices,
            values,
        ),
        2 => radix_groupby_engine_firstseen_u64::<FixedKeyOps<2>, T, A, O>(
            FixedKeyOps::<2>::new(2),
            key_slices,
            values,
        ),
        3 => radix_groupby_engine_firstseen_u64::<FixedKeyOps<3>, T, A, O>(
            FixedKeyOps::<3>::new(3),
            key_slices,
            values,
        ),
        4 => radix_groupby_engine_firstseen_u64::<FixedKeyOps<4>, T, A, O>(
            FixedKeyOps::<4>::new(4),
            key_slices,
            values,
        ),
        5 => radix_groupby_engine_firstseen_u64::<FixedKeyOps<5>, T, A, O>(
            FixedKeyOps::<5>::new(5),
            key_slices,
            values,
        ),
        6 => radix_groupby_engine_firstseen_u64::<FixedKeyOps<6>, T, A, O>(
            FixedKeyOps::<6>::new(6),
            key_slices,
            values,
        ),
        7 => radix_groupby_engine_firstseen_u64::<FixedKeyOps<7>, T, A, O>(
            FixedKeyOps::<7>::new(7),
            key_slices,
            values,
        ),
        8 => radix_groupby_engine_firstseen_u64::<FixedKeyOps<8>, T, A, O>(
            FixedKeyOps::<8>::new(8),
            key_slices,
            values,
        ),
        9 => radix_groupby_engine_firstseen_u64::<FixedKeyOps<9>, T, A, O>(
            FixedKeyOps::<9>::new(9),
            key_slices,
            values,
        ),
        10 => radix_groupby_engine_firstseen_u64::<FixedKeyOps<10>, T, A, O>(
            FixedKeyOps::<10>::new(10),
            key_slices,
            values,
        ),
        _ => radix_groupby_engine_firstseen_u64::<CompositeKeyOps, T, A, O>(
            CompositeKeyOps::new(key_slices.len()),
            key_slices,
            values,
        ),
    }
}

fn radix_groupby_firstseen_u32<T, A, O>(
    key_slices: &[&[i64]],
    values: &[T],
) -> Result<GroupByMultiResult<O>, String>
where
    T: Copy + Send + Sync,
    O: Copy + Send + Sync,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    radix_groupby_dispatch_firstseen_u32::<T, A, O>(key_slices, values)
}

fn radix_groupby_firstseen_u64<T, A, O>(
    key_slices: &[&[i64]],
    values: &[T],
) -> Result<GroupByMultiResult<O>, String>
where
    T: Copy + Send + Sync,
    O: Copy + Send + Sync,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    radix_groupby_dispatch_firstseen_u64::<T, A, O>(key_slices, values)
}

fn sort_groupby_result<V: Copy>(result: &mut GroupByMultiResult<V>) {
    if result.values.is_empty() {
        return;
    }

    let n_keys = result.n_keys;
    let n_groups = result.values.len();

    debug_assert_eq!(result.keys_flat.len(), n_groups * n_keys);

    let keys_flat = &result.keys_flat;
    let mut perm: Vec<usize> = (0..n_groups).collect();

    if n_groups < RADIX_SORT_THRESHOLD {
        perm.sort_unstable_by(|&i, &j| {
            let k_i = &keys_flat[i * n_keys..(i + 1) * n_keys];
            let k_j = &keys_flat[j * n_keys..(j + 1) * n_keys];
            k_i.cmp(k_j).then(i.cmp(&j))
        });
    } else {
        for col in (0..n_keys).rev() {
            let mut col_keys = Vec::with_capacity(n_groups);
            for group in 0..n_groups {
                let key = keys_flat[group * n_keys + col];
                col_keys.push(i64_to_sortable_u64(key));
            }
            perm = radix_sort_perm_by_u64_for_indices_par(&col_keys, &perm);
        }
    }

    let mut sorted_keys = Vec::with_capacity(result.keys_flat.len());
    let mut sorted_values = Vec::with_capacity(result.values.len());

    for &idx in &perm {
        sorted_keys.extend_from_slice(&keys_flat[idx * n_keys..(idx + 1) * n_keys]);
        sorted_values.push(result.values[idx]);
    }

    result.keys_flat = sorted_keys;
    result.values = sorted_values;
    result.perm = None;
}

fn radix_groupby_sorted<T, A, O>(
    key_slices: &[&[i64]],
    values: &[T],
) -> Result<GroupByMultiResult<O>, String>
where
    T: Copy + Send + Sync,
    O: Copy + Send + Sync,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    let mut result = radix_groupby_dispatch::<T, A, O>(key_slices, values)?;
    sort_groupby_result(&mut result);
    Ok(result)
}

// Public API - unsorted (fastest)

macro_rules! impl_radix_dispatch {
    ($name:ident, $val_type:ty, $agg:ty, $out_type:ty) => {
        pub fn $name(
            key_slices: &[&[i64]],
            values: &[$val_type],
        ) -> Result<GroupByMultiResult<$out_type>, String> {
            radix_groupby_dispatch::<$val_type, $agg, $out_type>(key_slices, values)
        }
    };
}

impl_radix_dispatch!(radix_groupby_sum_f64, f64, SumAggF64, f64);
impl_radix_dispatch!(radix_groupby_mean_f64, f64, MeanAggF64, f64);
impl_radix_dispatch!(radix_groupby_min_f64, f64, MinAggF64, f64);
impl_radix_dispatch!(radix_groupby_max_f64, f64, MaxAggF64, f64);

impl_radix_dispatch!(radix_groupby_sum_i64, i64, SumAggI64, i64);
impl_radix_dispatch!(radix_groupby_mean_i64, i64, MeanAggI64, f64);
impl_radix_dispatch!(radix_groupby_min_i64, i64, MinAggI64, i64);
impl_radix_dispatch!(radix_groupby_max_i64, i64, MaxAggI64, i64);

impl_radix_dispatch!(radix_groupby_count_f64, f64, CountAggF64, i64);
impl_radix_dispatch!(radix_groupby_count_i64, i64, CountAggI64, i64);

// Public API - first-seen ordered (for sort=False semantics)

macro_rules! impl_radix_firstseen_u32 {
    ($name:ident, $val_type:ty, $agg:ty, $out_type:ty) => {
        pub fn $name(
            key_slices: &[&[i64]],
            values: &[$val_type],
        ) -> Result<GroupByMultiResult<$out_type>, String> {
            radix_groupby_firstseen_u32::<$val_type, $agg, $out_type>(key_slices, values)
        }
    };
}

macro_rules! impl_radix_firstseen_u64 {
    ($name:ident, $val_type:ty, $agg:ty, $out_type:ty) => {
        pub fn $name(
            key_slices: &[&[i64]],
            values: &[$val_type],
        ) -> Result<GroupByMultiResult<$out_type>, String> {
            radix_groupby_firstseen_u64::<$val_type, $agg, $out_type>(key_slices, values)
        }
    };
}

impl_radix_firstseen_u32!(radix_groupby_sum_f64_firstseen_u32, f64, SumAggF64, f64);
impl_radix_firstseen_u32!(radix_groupby_mean_f64_firstseen_u32, f64, MeanAggF64, f64);
impl_radix_firstseen_u32!(radix_groupby_min_f64_firstseen_u32, f64, MinAggF64, f64);
impl_radix_firstseen_u32!(radix_groupby_max_f64_firstseen_u32, f64, MaxAggF64, f64);
impl_radix_firstseen_u32!(radix_groupby_count_f64_firstseen_u32, f64, CountAggF64, i64);

impl_radix_firstseen_u32!(radix_groupby_sum_i64_firstseen_u32, i64, SumAggI64, i64);
impl_radix_firstseen_u32!(radix_groupby_mean_i64_firstseen_u32, i64, MeanAggI64, f64);
impl_radix_firstseen_u32!(radix_groupby_min_i64_firstseen_u32, i64, MinAggI64, i64);
impl_radix_firstseen_u32!(radix_groupby_max_i64_firstseen_u32, i64, MaxAggI64, i64);
impl_radix_firstseen_u32!(radix_groupby_count_i64_firstseen_u32, i64, CountAggI64, i64);

impl_radix_firstseen_u64!(radix_groupby_sum_f64_firstseen_u64, f64, SumAggF64, f64);
impl_radix_firstseen_u64!(radix_groupby_mean_f64_firstseen_u64, f64, MeanAggF64, f64);
impl_radix_firstseen_u64!(radix_groupby_min_f64_firstseen_u64, f64, MinAggF64, f64);
impl_radix_firstseen_u64!(radix_groupby_max_f64_firstseen_u64, f64, MaxAggF64, f64);
impl_radix_firstseen_u64!(radix_groupby_count_f64_firstseen_u64, f64, CountAggF64, i64);

impl_radix_firstseen_u64!(radix_groupby_sum_i64_firstseen_u64, i64, SumAggI64, i64);
impl_radix_firstseen_u64!(radix_groupby_mean_i64_firstseen_u64, i64, MeanAggI64, f64);
impl_radix_firstseen_u64!(radix_groupby_min_i64_firstseen_u64, i64, MinAggI64, i64);
impl_radix_firstseen_u64!(radix_groupby_max_i64_firstseen_u64, i64, MaxAggI64, i64);
impl_radix_firstseen_u64!(radix_groupby_count_i64_firstseen_u64, i64, CountAggI64, i64);

// Public API - sorted (for sort=True)

pub fn radix_groupby_sum_f64_sorted(
    key_slices: &[&[i64]],
    values: &[f64],
) -> Result<GroupByMultiResult<f64>, String> {
    radix_groupby_sorted::<f64, SumAggF64, f64>(key_slices, values)
}

pub fn radix_groupby_mean_f64_sorted(
    key_slices: &[&[i64]],
    values: &[f64],
) -> Result<GroupByMultiResult<f64>, String> {
    radix_groupby_sorted::<f64, MeanAggF64, f64>(key_slices, values)
}

pub fn radix_groupby_min_f64_sorted(
    key_slices: &[&[i64]],
    values: &[f64],
) -> Result<GroupByMultiResult<f64>, String> {
    radix_groupby_sorted::<f64, MinAggF64, f64>(key_slices, values)
}

pub fn radix_groupby_max_f64_sorted(
    key_slices: &[&[i64]],
    values: &[f64],
) -> Result<GroupByMultiResult<f64>, String> {
    radix_groupby_sorted::<f64, MaxAggF64, f64>(key_slices, values)
}

pub fn radix_groupby_sum_i64_sorted(
    key_slices: &[&[i64]],
    values: &[i64],
) -> Result<GroupByMultiResult<i64>, String> {
    radix_groupby_sorted::<i64, SumAggI64, i64>(key_slices, values)
}

pub fn radix_groupby_mean_i64_sorted(
    key_slices: &[&[i64]],
    values: &[i64],
) -> Result<GroupByMultiResult<f64>, String> {
    radix_groupby_sorted::<i64, MeanAggI64, f64>(key_slices, values)
}

pub fn radix_groupby_min_i64_sorted(
    key_slices: &[&[i64]],
    values: &[i64],
) -> Result<GroupByMultiResult<i64>, String> {
    radix_groupby_sorted::<i64, MinAggI64, i64>(key_slices, values)
}

pub fn radix_groupby_max_i64_sorted(
    key_slices: &[&[i64]],
    values: &[i64],
) -> Result<GroupByMultiResult<i64>, String> {
    radix_groupby_sorted::<i64, MaxAggI64, i64>(key_slices, values)
}

pub fn radix_groupby_count_f64_sorted(
    key_slices: &[&[i64]],
    values: &[f64],
) -> Result<GroupByMultiResult<i64>, String> {
    radix_groupby_sorted::<f64, CountAggF64, i64>(key_slices, values)
}

pub fn radix_groupby_count_i64_sorted(
    key_slices: &[&[i64]],
    values: &[i64],
) -> Result<GroupByMultiResult<i64>, String> {
    radix_groupby_sorted::<i64, CountAggI64, i64>(key_slices, values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[inline]
    fn src_group_for_out<V>(res: &GroupByMultiResult<V>, out_g: usize) -> usize {
        match &res.perm {
            Some(p) => p[out_g],
            None => out_g,
        }
    }

    #[inline]
    fn key_at_out<V>(res: &GroupByMultiResult<V>, out_g: usize, col: usize) -> i64 {
        let src_g = src_group_for_out(res, out_g);
        res.keys_flat[src_g * res.n_keys + col]
    }

    #[inline]
    fn value_at_out<V: Copy>(res: &GroupByMultiResult<V>, out_g: usize) -> V {
        let src_g = src_group_for_out(res, out_g);
        res.values[src_g]
    }

    #[test]
    fn test_radix_groupby_sum_basic() {
        let col1 = vec![1i64, 2, 1, 2, 1];
        let col2 = vec![10i64, 20, 10, 20, 10];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let key_slices: Vec<&[i64]> = vec![&col1, &col2];
        let result = radix_groupby_sum_f64(&key_slices, &values).unwrap();

        assert_eq!(result.n_keys, 2);
        assert_eq!(result.values.len(), 2);

        let mut groups: AHashMap<(i64, i64), f64> = AHashMap::new();
        for i in 0..result.values.len() {
            let k0 = result.keys_flat[i * 2];
            let k1 = result.keys_flat[i * 2 + 1];
            groups.insert((k0, k1), result.values[i]);
        }

        assert!((groups[&(1, 10)] - 9.0).abs() < 1e-10);
        assert!((groups[&(2, 20)] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_radix_groupby_sorted_3keys() {
        let col1 = vec![1i64, 2, 1];
        let col2 = vec![10i64, 20, 10];
        let col3 = vec![100i64, 200, 100];
        let values = vec![1.0, 2.0, 3.0];

        let key_slices: Vec<&[i64]> = vec![&col1, &col2, &col3];
        let result = radix_groupby_sum_f64_sorted(&key_slices, &values).unwrap();

        assert_eq!(result.n_keys, 3);
        assert_eq!(result.values.len(), 2);

        // Verify sorted order: (1,10,100), (2,20,200)
        assert_eq!(key_at_out(&result, 0, 0), 1);
        assert_eq!(key_at_out(&result, 0, 1), 10);
        assert_eq!(key_at_out(&result, 0, 2), 100);
        assert!((value_at_out(&result, 0) - 4.0).abs() < 1e-10);

        assert_eq!(key_at_out(&result, 1, 0), 2);
        assert_eq!(key_at_out(&result, 1, 1), 20);
        assert_eq!(key_at_out(&result, 1, 2), 200);
        assert!((value_at_out(&result, 1) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_radix_groupby_sorted() {
        let col1 = vec![2i64, 1, 2, 1, 3];
        let col2 = vec![20i64, 10, 20, 10, 30];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let key_slices: Vec<&[i64]> = vec![&col1, &col2];
        let result = radix_groupby_sum_f64_sorted(&key_slices, &values).unwrap();

        assert_eq!(result.n_keys, 2);
        assert_eq!(result.values.len(), 3);

        // Verify sorted order: (1,10), (2,20), (3,30)
        assert_eq!(key_at_out(&result, 0, 0), 1);
        assert_eq!(key_at_out(&result, 0, 1), 10);
        assert_eq!(key_at_out(&result, 1, 0), 2);
        assert_eq!(key_at_out(&result, 1, 1), 20);
        assert_eq!(key_at_out(&result, 2, 0), 3);
        assert_eq!(key_at_out(&result, 2, 1), 30);

        assert!((value_at_out(&result, 0) - 6.0).abs() < 1e-10); // (1,10): 2+4
        assert!((value_at_out(&result, 1) - 4.0).abs() < 1e-10); // (2,20): 1+3
        assert!((value_at_out(&result, 2) - 5.0).abs() < 1e-10); // (3,30): 5
    }

    #[test]
    fn test_radix_groupby_sorted_negative_keys() {
        let col1 = vec![0i64, -1, 2, -1, -3];
        let col2 = vec![5i64, 4, 6, 4, 3];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let key_slices: Vec<&[i64]> = vec![&col1, &col2];
        let result = radix_groupby_sum_f64_sorted(&key_slices, &values).unwrap();

        assert_eq!(result.n_keys, 2);
        assert_eq!(result.values.len(), 4);

        // Verify sorted order: (-3,3), (-1,4), (0,5), (2,6)
        assert_eq!(key_at_out(&result, 0, 0), -3);
        assert_eq!(key_at_out(&result, 0, 1), 3);
        assert_eq!(key_at_out(&result, 1, 0), -1);
        assert_eq!(key_at_out(&result, 1, 1), 4);
        assert_eq!(key_at_out(&result, 2, 0), 0);
        assert_eq!(key_at_out(&result, 2, 1), 5);
        assert_eq!(key_at_out(&result, 3, 0), 2);
        assert_eq!(key_at_out(&result, 3, 1), 6);

        assert!((value_at_out(&result, 0) - 5.0).abs() < 1e-10);
        assert!((value_at_out(&result, 1) - 6.0).abs() < 1e-10);
        assert!((value_at_out(&result, 2) - 1.0).abs() < 1e-10);
        assert!((value_at_out(&result, 3) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_radix_groupby_empty() {
        let col1: Vec<i64> = vec![];
        let col2: Vec<i64> = vec![];
        let values: Vec<f64> = vec![];

        let key_slices: Vec<&[i64]> = vec![&col1, &col2];
        let result = radix_groupby_sum_f64(&key_slices, &values).unwrap();

        assert_eq!(result.n_keys, 2);
        assert_eq!(result.values.len(), 0);
    }

    #[test]
    fn test_radix_groupby_with_nan() {
        let col1 = vec![1i64, 1, 1];
        let col2 = vec![10i64, 10, 10];
        let values = vec![1.0, f64::NAN, 2.0];

        let key_slices: Vec<&[i64]> = vec![&col1, &col2];
        let result = radix_groupby_sum_f64(&key_slices, &values).unwrap();

        assert_eq!(result.values.len(), 1);
        assert!((result.values[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_radix_groupby_three_keys() {
        let col1 = vec![1i64, 1, 2];
        let col2 = vec![10i64, 10, 20];
        let col3 = vec![100i64, 100, 200];
        let values = vec![1.0, 2.0, 3.0];

        let key_slices: Vec<&[i64]> = vec![&col1, &col2, &col3];
        let result = radix_groupby_sum_f64(&key_slices, &values).unwrap();

        assert_eq!(result.n_keys, 3);
        assert_eq!(result.values.len(), 2);
    }

    #[test]
    fn test_radix_groupby_mean() {
        let col1 = vec![1i64, 2, 1, 2, 1];
        let col2 = vec![10i64, 20, 10, 20, 10];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let key_slices: Vec<&[i64]> = vec![&col1, &col2];
        let result = radix_groupby_mean_f64(&key_slices, &values).unwrap();

        let mut groups: AHashMap<(i64, i64), f64> = AHashMap::new();
        for i in 0..result.values.len() {
            let k0 = result.keys_flat[i * 2];
            let k1 = result.keys_flat[i * 2 + 1];
            groups.insert((k0, k1), result.values[i]);
        }

        assert!((groups[&(1, 10)] - 3.0).abs() < 1e-10); // (1+3+5)/3
        assert!((groups[&(2, 20)] - 3.0).abs() < 1e-10); // (2+4)/2
    }

    #[test]
    fn test_radix_groupby_i64() {
        let col1 = vec![1i64, 2, 1];
        let col2 = vec![10i64, 20, 10];
        let values: Vec<i64> = vec![100, 200, 300];

        let key_slices: Vec<&[i64]> = vec![&col1, &col2];
        let result = radix_groupby_sum_i64(&key_slices, &values).unwrap();

        let mut groups: AHashMap<(i64, i64), i64> = AHashMap::new();
        for i in 0..result.values.len() {
            let k0 = result.keys_flat[i * 2];
            let k1 = result.keys_flat[i * 2 + 1];
            groups.insert((k0, k1), result.values[i]);
        }

        assert_eq!(groups[&(1, 10)], 400);
        assert_eq!(groups[&(2, 20)], 200);
    }

    #[test]
    fn test_equivalence_fixed_vs_generic() {
        // Test N=2..=10
        let n_rows = 1000;

        for n_keys in 2..=10 {
            let keys: Vec<Vec<i64>> = (0..n_keys)
                .map(|k| {
                    (0..n_rows)
                        .map(|i| ((i as i64) * (k as i64 + 1)) % 20)
                        .collect()
                })
                .collect();

            let values: Vec<f64> = (0..n_rows).map(|i| i as f64).collect();
            let key_slices: Vec<&[i64]> = keys.iter().map(|v| v.as_slice()).collect();

            // Generic result
            let res_generic = radix_groupby::<f64, SumAggF64, f64>(&key_slices, &values).unwrap();

            // Fixed result
            let res_fixed = match n_keys {
                2 => radix_groupby_fixed::<2, f64, SumAggF64, f64>(&key_slices, &values),
                3 => radix_groupby_fixed::<3, f64, SumAggF64, f64>(&key_slices, &values),
                4 => radix_groupby_fixed::<4, f64, SumAggF64, f64>(&key_slices, &values),
                5 => radix_groupby_fixed::<5, f64, SumAggF64, f64>(&key_slices, &values),
                6 => radix_groupby_fixed::<6, f64, SumAggF64, f64>(&key_slices, &values),
                7 => radix_groupby_fixed::<7, f64, SumAggF64, f64>(&key_slices, &values),
                8 => radix_groupby_fixed::<8, f64, SumAggF64, f64>(&key_slices, &values),
                9 => radix_groupby_fixed::<9, f64, SumAggF64, f64>(&key_slices, &values),
                10 => radix_groupby_fixed::<10, f64, SumAggF64, f64>(&key_slices, &values),
                _ => panic!("Unsupported N"),
            }
            .unwrap();

            assert_eq!(
                res_generic.values.len(),
                res_fixed.values.len(),
                "N={}: Group counts differ",
                n_keys
            );

            // Convert to sorted vectors for comparison
            let sort_result = |res: &GroupByMultiResult<f64>| {
                let n_groups = res.values.len();
                let mut pairs = Vec::with_capacity(n_groups);
                for i in 0..n_groups {
                    let mut k = Vec::new();
                    for j in 0..n_keys {
                        k.push(res.keys_flat[i * n_keys + j]);
                    }
                    pairs.push((k, res.values[i]));
                }
                pairs.sort_by(|a, b| a.0.cmp(&b.0));
                pairs
            };

            let sorted_generic = sort_result(&res_generic);
            let sorted_fixed = sort_result(&res_fixed);

            for i in 0..sorted_generic.len() {
                assert_eq!(
                    sorted_generic[i].0, sorted_fixed[i].0,
                    "N={}: Key mismatch at index {}",
                    n_keys, i
                );
                assert!(
                    (sorted_generic[i].1 - sorted_fixed[i].1).abs() < 1e-10,
                    "N={}: Value mismatch at index {}",
                    n_keys,
                    i
                );
            }
        }
    }

    #[test]
    fn test_radix_groupby_sorted_4keys() {
        let col1 = vec![1i64, 2, 1];
        let col2 = vec![10i64, 20, 10];
        let col3 = vec![100i64, 200, 100];
        let col4 = vec![1000i64, 2000, 1000];
        let values = vec![1.0, 2.0, 3.0];

        let key_slices: Vec<&[i64]> = vec![&col1, &col2, &col3, &col4];
        let result = radix_groupby_sum_f64_sorted(&key_slices, &values).unwrap();

        assert_eq!(result.n_keys, 4);
        assert_eq!(result.values.len(), 2);

        // (1,10,100,1000) -> 1.0 + 3.0 = 4.0
        assert_eq!(key_at_out(&result, 0, 0), 1);
        assert_eq!(key_at_out(&result, 0, 1), 10);
        assert_eq!(key_at_out(&result, 0, 2), 100);
        assert_eq!(key_at_out(&result, 0, 3), 1000);
        assert!((value_at_out(&result, 0) - 4.0).abs() < 1e-10);

        // (2,20,200,2000) -> 2.0
        assert_eq!(key_at_out(&result, 1, 0), 2);
        assert_eq!(key_at_out(&result, 1, 1), 20);
        assert_eq!(key_at_out(&result, 1, 2), 200);
        assert_eq!(key_at_out(&result, 1, 3), 2000);
        assert!((value_at_out(&result, 1) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_radix_groupby_sorted_5keys() {
        let col1 = vec![1i64, 1];
        let col2 = vec![2i64, 2];
        let col3 = vec![3i64, 3];
        let col4 = vec![4i64, 4];
        let col5 = vec![5i64, 5];
        let values = vec![1.0, 2.0];

        let key_slices: Vec<&[i64]> = vec![&col1, &col2, &col3, &col4, &col5];
        let result = radix_groupby_sum_f64_sorted(&key_slices, &values).unwrap();

        assert_eq!(result.n_keys, 5);
        assert_eq!(result.values.len(), 1);

        for (col, expected) in [1i64, 2, 3, 4, 5].into_iter().enumerate() {
            assert_eq!(key_at_out(&result, 0, col), expected);
        }
        assert!((value_at_out(&result, 0) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_radix_groupby_sorted_empty() {
        let col1: Vec<i64> = vec![];
        let col2: Vec<i64> = vec![];
        let values: Vec<f64> = vec![];

        let key_slices: Vec<&[i64]> = vec![&col1, &col2];
        let result = radix_groupby_sum_f64_sorted(&key_slices, &values).unwrap();

        assert_eq!(result.n_keys, 2);
        assert_eq!(result.values.len(), 0);
        assert!(result.keys_flat.is_empty());
    }

    #[test]
    fn test_radix_groupby_sorted_i64_count() {
        let col1 = vec![1i64, 1, 2, 2, 1];
        let col2 = vec![10i64, 10, 20, 20, 10];
        let values = vec![100i64, 200, 300, 400, 500]; // Values don't matter for count

        let key_slices: Vec<&[i64]> = vec![&col1, &col2];
        let result = radix_groupby_count_i64_sorted(&key_slices, &values).unwrap();

        // Groups: (1,10) -> 3 rows, (2,20) -> 2 rows
        // Sorted: (1,10), (2,20)

        assert_eq!(result.n_keys, 2);
        assert_eq!(result.values.len(), 2);

        assert_eq!(key_at_out(&result, 0, 0), 1);
        assert_eq!(key_at_out(&result, 0, 1), 10);
        assert_eq!(value_at_out(&result, 0), 3);

        assert_eq!(key_at_out(&result, 1, 0), 2);
        assert_eq!(key_at_out(&result, 1, 1), 20);
        assert_eq!(value_at_out(&result, 1), 2);
    }

    #[test]
    fn test_stable_scatter_by_partition_preserves_row_order_within_partition() {
        let n_rows = 180_000;
        let hashes: Vec<u64> = (0..n_rows)
            .map(|row| (row as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .collect();

        let (perm, offsets) = stable_scatter_by_partition(&hashes);
        assert_eq!(perm.len(), n_rows);
        assert_eq!(offsets.len(), NUM_PARTITIONS + 1);
        assert_eq!(offsets[0], 0);
        assert_eq!(offsets[NUM_PARTITIONS], n_rows);

        for p in 0..NUM_PARTITIONS {
            let start = offsets[p];
            let end = offsets[p + 1];
            if end.saturating_sub(start) <= 1 {
                continue;
            }
            let slice = &perm[start..end];
            assert!(slice.windows(2).all(|w| w[0] < w[1]));
        }
    }

    #[test]
    fn test_firstseen_large_output_returns_perm_without_materialized_reorder() {
        let n_groups = 120_000usize; // n_groups * n_keys (2) > SMALL_DIRECT_THRESHOLD_ELEMS
        let k1: Vec<i64> = (0..n_groups as i64).collect();
        let k2: Vec<i64> = (0..n_groups as i64)
            .map(|i| i.wrapping_mul(0x9E37_79B9_i64))
            .collect();
        let values: Vec<f64> = (0..n_groups).map(|i| i as f64).collect();

        let key_slices: Vec<&[i64]> = vec![&k1, &k2];
        let result = radix_groupby_sum_f64_firstseen_u32(&key_slices, &values).unwrap();

        assert_eq!(result.n_keys, 2);
        assert_eq!(result.values.len(), n_groups);
        assert_eq!(result.keys_flat.len(), n_groups * 2);
        assert!(result.perm.is_some());

        // Applying perm must restore exact first-seen order.
        let perm = result.perm.as_ref().unwrap();
        assert_eq!(perm.len(), n_groups);
        for (out_g, &src_g) in perm.iter().enumerate().take(n_groups) {
            assert_eq!(result.values[src_g], out_g as f64);
        }
    }
}
