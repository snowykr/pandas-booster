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
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::aggregation::{
    Aggregator, CountAggF64, CountAggI64, MaxAggF64, MaxAggI64, MeanAggF64, MeanAggI64, MinAggF64,
    MinAggI64, SumAggF64, SumAggI64,
};

const INLINE_KEYS: usize = 4;
const NUM_PARTITIONS_BITS: usize = 10;
const NUM_PARTITIONS: usize = 1 << NUM_PARTITIONS_BITS;

#[derive(Clone, Debug)]
pub struct CompositeKey(SmallVec<[i64; INLINE_KEYS]>);

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

fn radix_groupby<T, A>(key_slices: &[&[i64]], values: &[T]) -> GroupByMultiResult
where
    T: Copy + Send + Sync,
    A: Aggregator<T, f64> + Clone + Default + Send,
{
    let n_keys = key_slices.len();
    let n_rows = values.len();

    if n_rows == 0 {
        return GroupByMultiResult {
            keys_flat: Vec::new(),
            n_keys,
            values: Vec::new(),
        };
    }


    // Phase 1: Parallel histogram with hash caching
    let histogram: Vec<AtomicUsize> = (0..NUM_PARTITIONS).map(|_| AtomicUsize::new(0)).collect();
    
    // Allocate vector to store computed hashes
    let mut hashes = vec![0u64; n_rows];
    
    // Pointer to hashes for safe parallel access (we write disjoint indices)
    let hashes_ptr = hashes.as_mut_ptr() as usize;
    // Send pointer wrapper
    struct PtrWrapper(usize);
    unsafe impl Send for PtrWrapper {}
    unsafe impl Sync for PtrWrapper {}
    let hashes_ptr_wrap = PtrWrapper(hashes_ptr);

    let chunk_size = (n_rows / rayon::current_num_threads()).max(8192);
    (0..n_rows)
        .into_par_iter()
        .with_min_len(chunk_size)
        .for_each(|row| {
            let h = compute_hash(key_slices, row);
            
            // Store hash for Phase 3
            unsafe {
                let ptr = hashes_ptr_wrap.0 as *mut u64;
                *ptr.add(row) = h;
            }
            
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
    // Wrapper for perm pointer
    let perm_ptr = perm.as_ptr() as usize;
    let perm_ptr_wrap = PtrWrapper(perm_ptr);
    
    (0..n_rows)
        .into_par_iter()
        .with_min_len(chunk_size)
        .for_each(|row| {
            // Use cached hash
            let h = unsafe {
                let ptr = hashes_ptr_wrap.0 as *const u64;
                *ptr.add(row)
            };
            
            let p = hash_to_partition(h);
            let pos = write_heads[p].fetch_add(1, Ordering::Relaxed);
            unsafe {
                let ptr = perm_ptr_wrap.0 as *mut usize;
                *ptr.add(pos) = row;
            }
        });

    // Phase 4: Per-partition aggregation (no merge!)
    let partition_results: Vec<Vec<(CompositeKey, f64)>> = (0..NUM_PARTITIONS)
        .into_par_iter()
        .map(|p| {
            let start = offsets[p];
            let end = offsets[p + 1];
            if start == end {
                return Vec::new();
            }

            let mut local_map: AHashMap<CompositeKey, A> = AHashMap::new();
            for &row in &perm[start..end] {
                let key = extract_key(key_slices, row);
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
    let mut keys_flat = Vec::with_capacity(total_groups * n_keys);
    let mut out_values = Vec::with_capacity(total_groups);

    for partition in partition_results {
        for (key, val) in partition {
            keys_flat.extend(key.iter().copied());
            out_values.push(val);
        }
    }

    GroupByMultiResult {
        keys_flat,
        n_keys,
        values: out_values,
    }
}

fn radix_groupby_sorted<T, A>(key_slices: &[&[i64]], values: &[T]) -> GroupByMultiResult
where
    T: Copy + Send + Sync,
    A: Aggregator<T, f64> + Clone + Default + Send,
{
    let mut result = radix_groupby::<T, A>(key_slices, values);

    if result.values.is_empty() {
        return result;
    }

    let n_keys = result.n_keys;
    let n_groups = result.values.len();

    // Build (key, value) pairs for sorting
    let mut pairs: Vec<(CompositeKey, f64)> = Vec::with_capacity(n_groups);
    for i in 0..n_groups {
        let mut key = CompositeKey::with_capacity(n_keys);
        for j in 0..n_keys {
            key.push(result.keys_flat[i * n_keys + j]);
        }
        pairs.push((key, result.values[i]));
    }

    // Parallel sort by key (lexicographic)
    pairs.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));

    // Rebuild flat output
    result.keys_flat.clear();
    result.values.clear();
    for (key, val) in pairs {
        result.keys_flat.extend(key.iter().copied());
        result.values.push(val);
    }

    result
}

// Public API - unsorted (fastest)

pub fn radix_groupby_sum_f64(key_slices: &[&[i64]], values: &[f64]) -> GroupByMultiResult {
    radix_groupby::<f64, SumAggF64>(key_slices, values)
}

pub fn radix_groupby_mean_f64(key_slices: &[&[i64]], values: &[f64]) -> GroupByMultiResult {
    radix_groupby::<f64, MeanAggF64>(key_slices, values)
}

pub fn radix_groupby_min_f64(key_slices: &[&[i64]], values: &[f64]) -> GroupByMultiResult {
    radix_groupby::<f64, MinAggF64>(key_slices, values)
}

pub fn radix_groupby_max_f64(key_slices: &[&[i64]], values: &[f64]) -> GroupByMultiResult {
    radix_groupby::<f64, MaxAggF64>(key_slices, values)
}

pub fn radix_groupby_sum_i64(key_slices: &[&[i64]], values: &[i64]) -> GroupByMultiResult {
    radix_groupby::<i64, SumAggI64>(key_slices, values)
}

pub fn radix_groupby_mean_i64(key_slices: &[&[i64]], values: &[i64]) -> GroupByMultiResult {
    radix_groupby::<i64, MeanAggI64>(key_slices, values)
}

pub fn radix_groupby_min_i64(key_slices: &[&[i64]], values: &[i64]) -> GroupByMultiResult {
    radix_groupby::<i64, MinAggI64>(key_slices, values)
}

pub fn radix_groupby_max_i64(key_slices: &[&[i64]], values: &[i64]) -> GroupByMultiResult {
    radix_groupby::<i64, MaxAggI64>(key_slices, values)
}

pub fn radix_groupby_count_f64(key_slices: &[&[i64]], values: &[f64]) -> GroupByMultiResult {
    radix_groupby::<f64, CountAggF64>(key_slices, values)
}

pub fn radix_groupby_count_i64(key_slices: &[&[i64]], values: &[i64]) -> GroupByMultiResult {
    radix_groupby::<i64, CountAggI64>(key_slices, values)
}

// Public API - sorted (for sort=True)

pub fn radix_groupby_sum_f64_sorted(key_slices: &[&[i64]], values: &[f64]) -> GroupByMultiResult {
    radix_groupby_sorted::<f64, SumAggF64>(key_slices, values)
}

pub fn radix_groupby_mean_f64_sorted(key_slices: &[&[i64]], values: &[f64]) -> GroupByMultiResult {
    radix_groupby_sorted::<f64, MeanAggF64>(key_slices, values)
}

pub fn radix_groupby_min_f64_sorted(key_slices: &[&[i64]], values: &[f64]) -> GroupByMultiResult {
    radix_groupby_sorted::<f64, MinAggF64>(key_slices, values)
}

pub fn radix_groupby_max_f64_sorted(key_slices: &[&[i64]], values: &[f64]) -> GroupByMultiResult {
    radix_groupby_sorted::<f64, MaxAggF64>(key_slices, values)
}

pub fn radix_groupby_sum_i64_sorted(key_slices: &[&[i64]], values: &[i64]) -> GroupByMultiResult {
    radix_groupby_sorted::<i64, SumAggI64>(key_slices, values)
}

pub fn radix_groupby_mean_i64_sorted(key_slices: &[&[i64]], values: &[i64]) -> GroupByMultiResult {
    radix_groupby_sorted::<i64, MeanAggI64>(key_slices, values)
}

pub fn radix_groupby_min_i64_sorted(key_slices: &[&[i64]], values: &[i64]) -> GroupByMultiResult {
    radix_groupby_sorted::<i64, MinAggI64>(key_slices, values)
}

pub fn radix_groupby_max_i64_sorted(key_slices: &[&[i64]], values: &[i64]) -> GroupByMultiResult {
    radix_groupby_sorted::<i64, MaxAggI64>(key_slices, values)
}

pub fn radix_groupby_count_f64_sorted(key_slices: &[&[i64]], values: &[f64]) -> GroupByMultiResult {
    radix_groupby_sorted::<f64, CountAggF64>(key_slices, values)
}

pub fn radix_groupby_count_i64_sorted(key_slices: &[&[i64]], values: &[i64]) -> GroupByMultiResult {
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
        let result = radix_groupby_sum_f64(&key_slices, &values);

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
    fn test_radix_groupby_sorted() {
        let col1 = vec![2i64, 1, 2, 1, 3];
        let col2 = vec![20i64, 10, 20, 10, 30];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let key_slices: Vec<&[i64]> = vec![&col1, &col2];
        let result = radix_groupby_sum_f64_sorted(&key_slices, &values);

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
        let result = radix_groupby_sum_f64(&key_slices, &values);

        assert_eq!(result.n_keys, 2);
        assert_eq!(result.values.len(), 0);
    }

    #[test]
    fn test_radix_groupby_with_nan() {
        let col1 = vec![1i64, 1, 1];
        let col2 = vec![10i64, 10, 10];
        let values = vec![1.0, f64::NAN, 2.0];

        let key_slices: Vec<&[i64]> = vec![&col1, &col2];
        let result = radix_groupby_sum_f64(&key_slices, &values);

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
        let result = radix_groupby_sum_f64(&key_slices, &values);

        assert_eq!(result.n_keys, 3);
        assert_eq!(result.values.len(), 2);
    }

    #[test]
    fn test_radix_groupby_mean() {
        let col1 = vec![1i64, 2, 1, 2, 1];
        let col2 = vec![10i64, 20, 10, 20, 10];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let key_slices: Vec<&[i64]> = vec![&col1, &col2];
        let result = radix_groupby_mean_f64(&key_slices, &values);

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
        let result = radix_groupby_sum_i64(&key_slices, &values);

        let mut groups: AHashMap<(i64, i64), f64> = AHashMap::new();
        for i in 0..result.values.len() {
            let k0 = result.keys_flat[i * 2];
            let k1 = result.keys_flat[i * 2 + 1];
            groups.insert((k0, k1), result.values[i]);
        }

        assert!((groups[&(1, 10)] - 400.0).abs() < 1e-10);
        assert!((groups[&(2, 20)] - 200.0).abs() < 1e-10);
    }
}
