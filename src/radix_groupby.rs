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
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::aggregation::{
    Aggregator, CountAggF64, CountAggI64, MaxAggF64, MaxAggI64, MeanAggF64, MeanAggI64, MinAggF64,
    MinAggI64, SumAggF64, SumAggI64,
};

const INLINE_KEYS: usize = 12;
const NUM_PARTITIONS_BITS: usize = 10;
const NUM_PARTITIONS: usize = 1 << NUM_PARTITIONS_BITS;

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
pub struct GroupByMultiResult {
    pub keys_flat: Vec<i64>,
    pub n_keys: usize,
    pub values: Vec<f64>,
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

fn radix_groupby_engine<Ops, T, A>(
    ops: Ops,
    key_slices: &[&[i64]],
    values: &[T],
) -> Result<GroupByMultiResult, String>
where
    Ops: RadixKeyOps,
    T: Copy + Send + Sync,
    A: Aggregator<T, f64> + Clone + Default + Send,
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
        return Ok(GroupByMultiResult {
            keys_flat: Vec::new(),
            n_keys: ops.n_keys(),
            values: Vec::new(),
        });
    }

    // Phase 1: Parallel histogram with hash caching
    let histogram: Vec<AtomicUsize> = (0..NUM_PARTITIONS).map(|_| AtomicUsize::new(0)).collect();

    // Allocate vector to store computed hashes
    let mut hashes = vec![0u64; n_rows];

    // Phase 1: Parallel histogram with hash caching
    hashes.par_iter_mut().enumerate().for_each(|(row, h_out)| {
        let h = ops.compute_hash(key_slices, row);
        *h_out = h;

        let p = hash_to_partition(h);
        histogram[p].fetch_add(1, Ordering::Relaxed);
    });

    // Phase 2: Prefix sum for write offsets
    let mut offsets = vec![0usize; NUM_PARTITIONS + 1];
    for i in 0..NUM_PARTITIONS {
        offsets[i + 1] = offsets[i] + histogram[i].load(Ordering::Relaxed);
    }

    // Phase 3: Scatter - reorder rows by partition using cached hashes
    let write_heads: Vec<AtomicUsize> = offsets[..NUM_PARTITIONS]
        .iter()
        .map(|&o| AtomicUsize::new(o))
        .collect();

    let perm = vec![0usize; n_rows];
    let perm_ptr = perm.as_ptr() as usize;

    struct PtrWrapper(usize);
    unsafe impl Send for PtrWrapper {}
    unsafe impl Sync for PtrWrapper {}
    let perm_ptr_wrap = PtrWrapper(perm_ptr);

    (0..n_rows).into_par_iter().for_each(|row| {
        // Use cached hash
        let h = hashes[row];

        let p = hash_to_partition(h);
        let pos = write_heads[p].fetch_add(1, Ordering::Relaxed);
        unsafe {
            let ptr = perm_ptr_wrap.0 as *mut usize;
            *ptr.add(pos) = row;
        }
    });

    // Phase 4: Per-partition aggregation (no merge!)
    let partition_results: Vec<Vec<(Ops::Key, f64)>> = (0..NUM_PARTITIONS)
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
    let mut keys_flat = Vec::with_capacity(total_groups * ops.n_keys());
    let mut out_values = Vec::with_capacity(total_groups);

    for partition in partition_results {
        for (key, val) in partition {
            Ops::push_flat(&mut keys_flat, &key);
            out_values.push(val);
        }
    }

    Ok(GroupByMultiResult {
        keys_flat,
        n_keys: ops.n_keys(),
        values: out_values,
    })
}

// -----------------------------------------------------------------------------
// Public Wrappers (Internal Dispatch)
// -----------------------------------------------------------------------------

fn radix_groupby_fixed<const N: usize, T, A>(
    key_slices: &[&[i64]],
    values: &[T],
) -> Result<GroupByMultiResult, String>
where
    T: Copy + Send + Sync,
    A: Aggregator<T, f64> + Clone + Default + Send,
{
    radix_groupby_engine::<FixedKeyOps<N>, T, A>(FixedKeyOps::<N>::new(N), key_slices, values)
}

fn radix_groupby<T, A>(key_slices: &[&[i64]], values: &[T]) -> Result<GroupByMultiResult, String>
where
    T: Copy + Send + Sync,
    A: Aggregator<T, f64> + Clone + Default + Send,
{
    radix_groupby_engine::<CompositeKeyOps, T, A>(
        CompositeKeyOps::new(key_slices.len()),
        key_slices,
        values,
    )
}

fn radix_groupby_dispatch<T, A>(
    key_slices: &[&[i64]],
    values: &[T],
) -> Result<GroupByMultiResult, String>
where
    T: Copy + Send + Sync,
    A: Aggregator<T, f64> + Clone + Default + Send,
{
    match key_slices.len() {
        2 => radix_groupby_fixed::<2, T, A>(key_slices, values),
        3 => radix_groupby_fixed::<3, T, A>(key_slices, values),
        4 => radix_groupby_fixed::<4, T, A>(key_slices, values),
        _ => radix_groupby::<T, A>(key_slices, values),
    }
}

fn sort_groupby_result(result: &mut GroupByMultiResult) {
    if result.values.is_empty() {
        return;
    }

    let n_keys = result.n_keys;
    let n_groups = result.values.len();

    let mut perm: Vec<usize> = (0..n_groups).collect();
    let keys_flat = &result.keys_flat;

    perm.par_sort_unstable_by(|&i, &j| {
        let k_i = &keys_flat[i * n_keys..(i + 1) * n_keys];
        let k_j = &keys_flat[j * n_keys..(j + 1) * n_keys];
        k_i.cmp(k_j)
    });

    let mut sorted_keys = Vec::with_capacity(result.keys_flat.len());
    let mut sorted_values = Vec::with_capacity(result.values.len());

    for &idx in &perm {
        sorted_keys.extend_from_slice(&keys_flat[idx * n_keys..(idx + 1) * n_keys]);
        sorted_values.push(result.values[idx]);
    }

    result.keys_flat = sorted_keys;
    result.values = sorted_values;
}

fn radix_groupby_sorted<T, A>(
    key_slices: &[&[i64]],
    values: &[T],
) -> Result<GroupByMultiResult, String>
where
    T: Copy + Send + Sync,
    A: Aggregator<T, f64> + Clone + Default + Send,
{
    let mut result = radix_groupby_dispatch::<T, A>(key_slices, values)?;
    sort_groupby_result(&mut result);
    Ok(result)
}

// Public API - unsorted (fastest)

macro_rules! impl_radix_dispatch {
    ($name:ident, $val_type:ty, $agg:ty) => {
        pub fn $name(
            key_slices: &[&[i64]],
            values: &[$val_type],
        ) -> Result<GroupByMultiResult, String> {
            radix_groupby_dispatch::<$val_type, $agg>(key_slices, values)
        }
    };
}

impl_radix_dispatch!(radix_groupby_sum_f64, f64, SumAggF64);
impl_radix_dispatch!(radix_groupby_mean_f64, f64, MeanAggF64);
impl_radix_dispatch!(radix_groupby_min_f64, f64, MinAggF64);
impl_radix_dispatch!(radix_groupby_max_f64, f64, MaxAggF64);

impl_radix_dispatch!(radix_groupby_sum_i64, i64, SumAggI64);
impl_radix_dispatch!(radix_groupby_mean_i64, i64, MeanAggI64);
impl_radix_dispatch!(radix_groupby_min_i64, i64, MinAggI64);
impl_radix_dispatch!(radix_groupby_max_i64, i64, MaxAggI64);

impl_radix_dispatch!(radix_groupby_count_f64, f64, CountAggF64);
impl_radix_dispatch!(radix_groupby_count_i64, i64, CountAggI64);

// Public API - sorted (for sort=True)

pub fn radix_groupby_sum_f64_sorted(
    key_slices: &[&[i64]],
    values: &[f64],
) -> Result<GroupByMultiResult, String> {
    radix_groupby_sorted::<f64, SumAggF64>(key_slices, values)
}

pub fn radix_groupby_mean_f64_sorted(
    key_slices: &[&[i64]],
    values: &[f64],
) -> Result<GroupByMultiResult, String> {
    radix_groupby_sorted::<f64, MeanAggF64>(key_slices, values)
}

pub fn radix_groupby_min_f64_sorted(
    key_slices: &[&[i64]],
    values: &[f64],
) -> Result<GroupByMultiResult, String> {
    radix_groupby_sorted::<f64, MinAggF64>(key_slices, values)
}

pub fn radix_groupby_max_f64_sorted(
    key_slices: &[&[i64]],
    values: &[f64],
) -> Result<GroupByMultiResult, String> {
    radix_groupby_sorted::<f64, MaxAggF64>(key_slices, values)
}

pub fn radix_groupby_sum_i64_sorted(
    key_slices: &[&[i64]],
    values: &[i64],
) -> Result<GroupByMultiResult, String> {
    radix_groupby_sorted::<i64, SumAggI64>(key_slices, values)
}

pub fn radix_groupby_mean_i64_sorted(
    key_slices: &[&[i64]],
    values: &[i64],
) -> Result<GroupByMultiResult, String> {
    radix_groupby_sorted::<i64, MeanAggI64>(key_slices, values)
}

pub fn radix_groupby_min_i64_sorted(
    key_slices: &[&[i64]],
    values: &[i64],
) -> Result<GroupByMultiResult, String> {
    radix_groupby_sorted::<i64, MinAggI64>(key_slices, values)
}

pub fn radix_groupby_max_i64_sorted(
    key_slices: &[&[i64]],
    values: &[i64],
) -> Result<GroupByMultiResult, String> {
    radix_groupby_sorted::<i64, MaxAggI64>(key_slices, values)
}

pub fn radix_groupby_count_f64_sorted(
    key_slices: &[&[i64]],
    values: &[f64],
) -> Result<GroupByMultiResult, String> {
    radix_groupby_sorted::<f64, CountAggF64>(key_slices, values)
}

pub fn radix_groupby_count_i64_sorted(
    key_slices: &[&[i64]],
    values: &[i64],
) -> Result<GroupByMultiResult, String> {
    radix_groupby_sorted::<i64, CountAggI64>(key_slices, values)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(result.keys_flat[0], 1);
        assert_eq!(result.keys_flat[1], 10);
        assert_eq!(result.keys_flat[2], 100);
        assert!((result.values[0] - 4.0).abs() < 1e-10);

        assert_eq!(result.keys_flat[3], 2);
        assert_eq!(result.keys_flat[4], 20);
        assert_eq!(result.keys_flat[5], 200);
        assert!((result.values[1] - 2.0).abs() < 1e-10);
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
        assert_eq!(result.keys_flat[0], 1);
        assert_eq!(result.keys_flat[1], 10);
        assert_eq!(result.keys_flat[2], 2);
        assert_eq!(result.keys_flat[3], 20);
        assert_eq!(result.keys_flat[4], 3);
        assert_eq!(result.keys_flat[5], 30);

        assert!((result.values[0] - 6.0).abs() < 1e-10); // (1,10): 2+4
        assert!((result.values[1] - 4.0).abs() < 1e-10); // (2,20): 1+3
        assert!((result.values[2] - 5.0).abs() < 1e-10); // (3,30): 5
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

        let mut groups: AHashMap<(i64, i64), f64> = AHashMap::new();
        for i in 0..result.values.len() {
            let k0 = result.keys_flat[i * 2];
            let k1 = result.keys_flat[i * 2 + 1];
            groups.insert((k0, k1), result.values[i]);
        }

        assert!((groups[&(1, 10)] - 400.0).abs() < 1e-10);
        assert!((groups[&(2, 20)] - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_equivalence_fixed_vs_generic() {
        // Test N=2, 3, 4
        let n_rows = 1000;

        for n_keys in 2..=4 {
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
            let res_generic = radix_groupby::<f64, SumAggF64>(&key_slices, &values).unwrap();

            // Fixed result
            let res_fixed = match n_keys {
                2 => radix_groupby_fixed::<2, f64, SumAggF64>(&key_slices, &values),
                3 => radix_groupby_fixed::<3, f64, SumAggF64>(&key_slices, &values),
                4 => radix_groupby_fixed::<4, f64, SumAggF64>(&key_slices, &values),
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
            let sort_result = |res: &GroupByMultiResult| {
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
        assert_eq!(result.keys_flat[0], 1);
        assert_eq!(result.keys_flat[1], 10);
        assert_eq!(result.keys_flat[2], 100);
        assert_eq!(result.keys_flat[3], 1000);
        assert!((result.values[0] - 4.0).abs() < 1e-10);

        // (2,20,200,2000) -> 2.0
        assert_eq!(result.keys_flat[4], 2);
        assert_eq!(result.keys_flat[5], 20);
        assert_eq!(result.keys_flat[6], 200);
        assert_eq!(result.keys_flat[7], 2000);
        assert!((result.values[1] - 2.0).abs() < 1e-10);
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
        // This should hit the generic path (dispatch handles 2,3,4 specially)
        let result = radix_groupby_sum_f64_sorted(&key_slices, &values).unwrap();

        assert_eq!(result.n_keys, 5);
        assert_eq!(result.values.len(), 1);

        assert_eq!(result.keys_flat, vec![1, 2, 3, 4, 5]);
        assert!((result.values[0] - 3.0).abs() < 1e-10);
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

        assert_eq!(result.keys_flat[0], 1);
        assert_eq!(result.keys_flat[1], 10);
        assert!((result.values[0] - 3.0).abs() < 1e-10);

        assert_eq!(result.keys_flat[2], 2);
        assert_eq!(result.keys_flat[3], 20);
        assert!((result.values[1] - 2.0).abs() < 1e-10);
    }
}
