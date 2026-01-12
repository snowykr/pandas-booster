//! Parallel multi-column groupby implementation using Rayon's map-reduce pattern.
//!
//! This module extends the single-key groupby functionality to support grouping by
//! multiple integer columns (2-10 keys). It uses:
//!
//! - **CompositeKey as HashMap key**: Direct key comparison without collision resolution
//! - **Rayon par_chunks**: Parallel chunk processing with per-thread hash maps
//! - **Aggregator merge**: Thread-safe merge of partial results
//!
//! ## Design
//!
//! - **SmallVec optimization**: Keys ≤4 columns are stored inline without heap allocation
//! - **Zero collision overhead**: CompositeKey implements Hash+Eq for direct HashMap usage
//! - **Parallel reduction**: Same pattern as single-key groupby for consistent performance

use ahash::AHashMap;
use pyo3::prelude::*;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::hash::{Hash, Hasher};

use crate::aggregation::{
    Aggregator, MaxAggF64, MaxAggI64, MeanAggF64, MeanAggI64, MinAggF64, MinAggI64, SumAggF64,
    SumAggI64,
};

/// Maximum number of key columns that can be inlined without heap allocation.
/// Keys beyond this count will spill to the heap.
const INLINE_KEYS: usize = 4;

/// Compact storage for composite keys, avoiding heap allocation for ≤4 keys.
/// Implements Hash and Eq for direct use as HashMap key.
#[derive(Clone, Debug)]
pub struct CompositeKey(SmallVec<[i64; INLINE_KEYS]>);

impl CompositeKey {
    /// Create a new CompositeKey with the given capacity.
    #[inline]
    fn with_capacity(cap: usize) -> Self {
        Self(SmallVec::with_capacity(cap))
    }

    /// Push a key value.
    #[inline]
    fn push(&mut self, val: i64) {
        self.0.push(val);
    }

    /// Get iterator over values.
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
        // Hash each key value directly - AHash will handle mixing
        for &v in &self.0 {
            v.hash(state);
        }
    }
}

/// Result container for multi-key groupby operations.
/// Keys are stored as a flat Vec with row-major layout: [k0_g0, k1_g0, ..., k0_g1, k1_g1, ...]
#[derive(Debug)]
pub struct GroupByMultiResult {
    /// Flattened key data in row-major order (n_groups × n_keys)
    pub keys_flat: Vec<i64>,
    /// Number of key columns
    pub n_keys: usize,
    /// Aggregated values (one per group)
    pub values: Vec<f64>,
}

/// Extract key values for a specific row into a CompositeKey.
#[inline]
fn extract_keys(key_slices: &[&[i64]], row: usize) -> CompositeKey {
    let mut keys = CompositeKey::with_capacity(key_slices.len());
    for col in key_slices {
        keys.push(col[row]);
    }
    keys
}

/// Parallel multi-key groupby implementation using Rayon's map-reduce pattern.
///
/// This follows the same pattern as single-key groupby:
/// 1. Chunk input arrays across CPU cores
/// 2. Build per-thread hash maps with partial aggregations
/// 3. Merge partial results via Aggregator::merge
fn parallel_groupby_multi<T, A>(key_slices: &[&[i64]], values: &[T]) -> GroupByMultiResult
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

    // Determine chunk size based on available threads
    let chunk_size = (n_rows / rayon::current_num_threads()).max(10_000);

    // Create row indices for chunking (we need to access multiple columns per row)
    let row_indices: Vec<usize> = (0..n_rows).collect();

    // Parallel map-reduce
    let merged: AHashMap<CompositeKey, A> = row_indices
        .par_chunks(chunk_size)
        .fold(
            AHashMap::default,
            |mut acc: AHashMap<CompositeKey, A>, chunk| {
                for &row in chunk {
                    let key = extract_keys(key_slices, row);
                    let val = values[row];
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

    // Flatten results
    let n_groups = merged.len();
    let mut keys_flat = Vec::with_capacity(n_groups * n_keys);
    let mut out_values = Vec::with_capacity(n_groups);

    for (key, agg) in merged {
        // Append keys in row-major order
        keys_flat.extend(key.iter().copied());
        out_values.push(agg.finalize());
    }

    GroupByMultiResult {
        keys_flat,
        n_keys,
        values: out_values,
    }
}

/// Wrapper for f64 aggregation.
fn parallel_groupby_multi_f64<A>(key_slices: &[&[i64]], values: &[f64]) -> GroupByMultiResult
where
    A: Aggregator<f64, f64> + Clone + Default + Send,
{
    parallel_groupby_multi::<f64, A>(key_slices, values)
}

/// Wrapper for i64 aggregation.
fn parallel_groupby_multi_i64<A>(key_slices: &[&[i64]], values: &[i64]) -> GroupByMultiResult
where
    A: Aggregator<i64, f64> + Clone + Default + Send,
{
    parallel_groupby_multi::<i64, A>(key_slices, values)
}

// =============================================================================
// Public API functions for f64 values
// =============================================================================

/// Multi-key groupby sum for f64 values.
pub fn multi_groupby_sum_f64(
    key_slices: &[&[i64]],
    values: &[f64],
) -> PyResult<GroupByMultiResult> {
    Ok(parallel_groupby_multi_f64::<SumAggF64>(key_slices, values))
}

/// Multi-key groupby mean for f64 values.
pub fn multi_groupby_mean_f64(
    key_slices: &[&[i64]],
    values: &[f64],
) -> PyResult<GroupByMultiResult> {
    Ok(parallel_groupby_multi_f64::<MeanAggF64>(key_slices, values))
}

/// Multi-key groupby min for f64 values.
pub fn multi_groupby_min_f64(
    key_slices: &[&[i64]],
    values: &[f64],
) -> PyResult<GroupByMultiResult> {
    Ok(parallel_groupby_multi_f64::<MinAggF64>(key_slices, values))
}

/// Multi-key groupby max for f64 values.
pub fn multi_groupby_max_f64(
    key_slices: &[&[i64]],
    values: &[f64],
) -> PyResult<GroupByMultiResult> {
    Ok(parallel_groupby_multi_f64::<MaxAggF64>(key_slices, values))
}

// =============================================================================
// Public API functions for i64 values
// =============================================================================

/// Multi-key groupby sum for i64 values.
pub fn multi_groupby_sum_i64(
    key_slices: &[&[i64]],
    values: &[i64],
) -> PyResult<GroupByMultiResult> {
    Ok(parallel_groupby_multi_i64::<SumAggI64>(key_slices, values))
}

/// Multi-key groupby mean for i64 values.
pub fn multi_groupby_mean_i64(
    key_slices: &[&[i64]],
    values: &[i64],
) -> PyResult<GroupByMultiResult> {
    Ok(parallel_groupby_multi_i64::<MeanAggI64>(key_slices, values))
}

/// Multi-key groupby min for i64 values.
pub fn multi_groupby_min_i64(
    key_slices: &[&[i64]],
    values: &[i64],
) -> PyResult<GroupByMultiResult> {
    Ok(parallel_groupby_multi_i64::<MinAggI64>(key_slices, values))
}

/// Multi-key groupby max for i64 values.
pub fn multi_groupby_max_i64(
    key_slices: &[&[i64]],
    values: &[i64],
) -> PyResult<GroupByMultiResult> {
    Ok(parallel_groupby_multi_i64::<MaxAggI64>(key_slices, values))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_composite_key_hash_eq() {
        let mut key1 = CompositeKey::with_capacity(2);
        key1.push(1);
        key1.push(10);

        let mut key2 = CompositeKey::with_capacity(2);
        key2.push(1);
        key2.push(10);

        let mut key3 = CompositeKey::with_capacity(2);
        key3.push(1);
        key3.push(20);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);

        // Test hash consistency
        use std::collections::hash_map::DefaultHasher;
        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();
        key1.hash(&mut hasher1);
        key2.hash(&mut hasher2);
        assert_eq!(hasher1.finish(), hasher2.finish());
    }

    #[test]
    fn test_multi_groupby_sum_f64_basic() {
        // Two key columns: (1,10), (2,20), (1,10), (2,20), (1,10)
        let col1 = vec![1i64, 2, 1, 2, 1];
        let col2 = vec![10i64, 20, 10, 20, 10];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let key_slices: Vec<&[i64]> = vec![&col1, &col2];
        let result = multi_groupby_sum_f64(&key_slices, &values).unwrap();

        assert_eq!(result.n_keys, 2);
        assert_eq!(result.values.len(), 2); // Two unique groups

        // Build map for verification
        let mut groups: AHashMap<(i64, i64), f64> = AHashMap::new();
        for i in 0..result.values.len() {
            let k0 = result.keys_flat[i * 2];
            let k1 = result.keys_flat[i * 2 + 1];
            groups.insert((k0, k1), result.values[i]);
        }

        assert!((groups[&(1, 10)] - 9.0).abs() < 1e-10); // 1 + 3 + 5
        assert!((groups[&(2, 20)] - 6.0).abs() < 1e-10); // 2 + 4
    }

    #[test]
    fn test_multi_groupby_mean_f64() {
        let col1 = vec![1i64, 2, 1, 2, 1];
        let col2 = vec![10i64, 20, 10, 20, 10];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let key_slices: Vec<&[i64]> = vec![&col1, &col2];
        let result = multi_groupby_mean_f64(&key_slices, &values).unwrap();

        let mut groups: AHashMap<(i64, i64), f64> = AHashMap::new();
        for i in 0..result.values.len() {
            let k0 = result.keys_flat[i * 2];
            let k1 = result.keys_flat[i * 2 + 1];
            groups.insert((k0, k1), result.values[i]);
        }

        assert!((groups[&(1, 10)] - 3.0).abs() < 1e-10); // (1 + 3 + 5) / 3
        assert!((groups[&(2, 20)] - 3.0).abs() < 1e-10); // (2 + 4) / 2
    }

    #[test]
    fn test_multi_groupby_with_nan() {
        let col1 = vec![1i64, 1, 1];
        let col2 = vec![10i64, 10, 10];
        let values = vec![1.0, f64::NAN, 2.0];

        let key_slices: Vec<&[i64]> = vec![&col1, &col2];
        let result = multi_groupby_sum_f64(&key_slices, &values).unwrap();

        assert_eq!(result.values.len(), 1);
        assert!((result.values[0] - 3.0).abs() < 1e-10); // NaN skipped
    }

    #[test]
    fn test_multi_groupby_three_keys() {
        let col1 = vec![1i64, 1, 2];
        let col2 = vec![10i64, 10, 20];
        let col3 = vec![100i64, 100, 200];
        let values = vec![1.0, 2.0, 3.0];

        let key_slices: Vec<&[i64]> = vec![&col1, &col2, &col3];
        let result = multi_groupby_sum_f64(&key_slices, &values).unwrap();

        assert_eq!(result.n_keys, 3);
        assert_eq!(result.values.len(), 2);

        // Check keys are stored correctly
        let mut groups: AHashMap<(i64, i64, i64), f64> = AHashMap::new();
        for i in 0..result.values.len() {
            let k0 = result.keys_flat[i * 3];
            let k1 = result.keys_flat[i * 3 + 1];
            let k2 = result.keys_flat[i * 3 + 2];
            groups.insert((k0, k1, k2), result.values[i]);
        }

        assert!((groups[&(1, 10, 100)] - 3.0).abs() < 1e-10);
        assert!((groups[&(2, 20, 200)] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_multi_groupby_sum_i64() {
        let col1 = vec![1i64, 2, 1];
        let col2 = vec![10i64, 20, 10];
        let values: Vec<i64> = vec![100, 200, 300];

        let key_slices: Vec<&[i64]> = vec![&col1, &col2];
        let result = multi_groupby_sum_i64(&key_slices, &values).unwrap();

        let mut groups: AHashMap<(i64, i64), f64> = AHashMap::new();
        for i in 0..result.values.len() {
            let k0 = result.keys_flat[i * 2];
            let k1 = result.keys_flat[i * 2 + 1];
            groups.insert((k0, k1), result.values[i]);
        }

        assert!((groups[&(1, 10)] - 400.0).abs() < 1e-10); // 100 + 300
        assert!((groups[&(2, 20)] - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_composite_key_inline_allocation() {
        // Verify SmallVec doesn't heap-allocate for ≤4 keys
        let mut key = CompositeKey::with_capacity(4);
        key.push(1);
        key.push(2);
        key.push(3);
        key.push(4);
        assert!(!key.0.spilled()); // Should be inline

        let mut key_large = CompositeKey::with_capacity(5);
        for i in 0..5 {
            key_large.push(i);
        }
        assert!(key_large.0.spilled()); // Should spill to heap
    }

    #[test]
    fn test_multi_groupby_empty() {
        let col1: Vec<i64> = vec![];
        let col2: Vec<i64> = vec![];
        let values: Vec<f64> = vec![];

        let key_slices: Vec<&[i64]> = vec![&col1, &col2];
        let result = multi_groupby_sum_f64(&key_slices, &values).unwrap();

        assert_eq!(result.n_keys, 2);
        assert_eq!(result.values.len(), 0);
        assert_eq!(result.keys_flat.len(), 0);
    }

    #[test]
    fn test_multi_groupby_min_max_f64() {
        let col1 = vec![1i64, 1, 1, 2, 2];
        let col2 = vec![10i64, 10, 10, 20, 20];
        let values = vec![5.0, 2.0, 8.0, 1.0, 9.0];

        let key_slices: Vec<&[i64]> = vec![&col1, &col2];

        let result_min = multi_groupby_min_f64(&key_slices, &values).unwrap();
        let result_max = multi_groupby_max_f64(&key_slices, &values).unwrap();

        let mut min_groups: AHashMap<(i64, i64), f64> = AHashMap::new();
        let mut max_groups: AHashMap<(i64, i64), f64> = AHashMap::new();

        for i in 0..result_min.values.len() {
            let k0 = result_min.keys_flat[i * 2];
            let k1 = result_min.keys_flat[i * 2 + 1];
            min_groups.insert((k0, k1), result_min.values[i]);
        }

        for i in 0..result_max.values.len() {
            let k0 = result_max.keys_flat[i * 2];
            let k1 = result_max.keys_flat[i * 2 + 1];
            max_groups.insert((k0, k1), result_max.values[i]);
        }

        assert!((min_groups[&(1, 10)] - 2.0).abs() < 1e-10);
        assert!((min_groups[&(2, 20)] - 1.0).abs() < 1e-10);
        assert!((max_groups[&(1, 10)] - 8.0).abs() < 1e-10);
        assert!((max_groups[&(2, 20)] - 9.0).abs() < 1e-10);
    }
}
