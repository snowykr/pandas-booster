//! Multi-column groupby implementation using row-hashing with collision resolution.
//!
//! This module extends the single-key groupby functionality to support grouping by
//! multiple integer columns (2-10 keys). It uses a u64 hash as the primary map key
//! with stored key tuples for collision resolution.
//!
//! ## Design
//!
//! - **Hash as bucket key**: Each row's key columns are hashed to u64 for fast lookup
//! - **Collision handling**: Actual key tuples are stored and compared for equality
//! - **Memory efficient**: Key tuples only stored once per unique group (via SmallVec)
//! - **Parallel ready**: Structure supports future Rayon-based parallelization

use ahash::AHashMap;
use pyo3::prelude::*;
use smallvec::SmallVec;

use crate::aggregation::{
    Aggregator, MaxAggF64, MaxAggI64, MeanAggF64, MeanAggI64, MinAggF64, MinAggI64, SumAggF64,
    SumAggI64,
};

/// Maximum number of key columns that can be inlined without heap allocation.
/// Keys beyond this count will spill to the heap.
const INLINE_KEYS: usize = 4;

/// Compact storage for composite keys, avoiding heap allocation for ≤4 keys.
pub type CompositeKey = SmallVec<[i64; INLINE_KEYS]>;

/// State for a single group in multi-key aggregation.
#[derive(Clone)]
struct GroupState<A> {
    /// The actual key values for this group (used for collision resolution)
    keys: CompositeKey,
    /// The aggregator instance holding partial/final results
    agg: A,
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

/// Fast hash mixing function (MurmurHash3 finalizer variant).
#[inline]
fn mix64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}

/// Compute a combined hash for a row across multiple key columns.
///
/// Uses a multiplicative hash combining strategy similar to Boost's hash_combine.
#[inline]
fn row_hash(key_slices: &[&[i64]], row: usize) -> u64 {
    let mut h: u64 = 0x9e3779b97f4a7c15; // Golden ratio seed
    for col in key_slices {
        let v = col[row] as u64;
        // Combine: h ^= mix(v + seed) then rotate and multiply
        h ^= mix64(v.wrapping_add(0x9e3779b97f4a7c15));
        h = h.rotate_left(27).wrapping_mul(0x3c79ac492ba7b653);
    }
    h
}

/// Check if stored keys match the keys at a given row.
#[inline]
fn keys_equal(stored: &CompositeKey, key_slices: &[&[i64]], row: usize) -> bool {
    debug_assert_eq!(stored.len(), key_slices.len());
    for (i, col) in key_slices.iter().enumerate() {
        if stored[i] != col[row] {
            return false;
        }
    }
    true
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

/// Generic multi-key groupby implementation.
///
/// Uses hash buckets with collision resolution via stored key comparison.
/// The aggregator type determines the operation (sum, mean, min, max).
fn groupby_multi<T, A>(key_slices: &[&[i64]], values: &[T]) -> GroupByMultiResult
where
    T: Copy,
    A: Aggregator<T, f64> + Clone + Default,
{
    let n_keys = key_slices.len();
    let n_rows = values.len();

    // Hash bucket -> list of groups (for collision handling)
    // Most buckets will have exactly 1 group; Vec handles rare collisions
    let mut map: AHashMap<u64, Vec<GroupState<A>>> = AHashMap::new();

    for row in 0..n_rows {
        let v = values[row];
        let h = row_hash(key_slices, row);

        let bucket = map.entry(h).or_insert_with(Vec::new);

        // Find existing group in bucket (collision resolution)
        if let Some(state) = bucket
            .iter_mut()
            .find(|s| keys_equal(&s.keys, key_slices, row))
        {
            state.agg.update(v);
        } else {
            // New group in this bucket
            let keys = extract_keys(key_slices, row);
            let mut agg = A::init();
            agg.update(v);
            bucket.push(GroupState { keys, agg });
        }
    }

    // Flatten results
    let n_groups: usize = map.values().map(|v| v.len()).sum();
    let mut keys_flat = Vec::with_capacity(n_groups * n_keys);
    let mut out_values = Vec::with_capacity(n_groups);

    for (_h, states) in map {
        for state in states {
            // Append keys in row-major order
            keys_flat.extend(state.keys.iter().copied());
            out_values.push(state.agg.finalize());
        }
    }

    GroupByMultiResult {
        keys_flat,
        n_keys,
        values: out_values,
    }
}

/// Wrapper for f64 aggregation that skips NaN values.
fn groupby_multi_f64<A>(key_slices: &[&[i64]], values: &[f64]) -> GroupByMultiResult
where
    A: Aggregator<f64, f64> + Clone + Default,
{
    groupby_multi::<f64, A>(key_slices, values)
}

/// Wrapper for i64 aggregation.
fn groupby_multi_i64<A>(key_slices: &[&[i64]], values: &[i64]) -> GroupByMultiResult
where
    A: Aggregator<i64, f64> + Clone + Default,
{
    groupby_multi::<i64, A>(key_slices, values)
}

// Public API functions for f64 values

/// Multi-key groupby sum for f64 values.
pub fn multi_groupby_sum_f64(
    key_slices: &[&[i64]],
    values: &[f64],
) -> PyResult<GroupByMultiResult> {
    Ok(groupby_multi_f64::<SumAggF64>(key_slices, values))
}

/// Multi-key groupby mean for f64 values.
pub fn multi_groupby_mean_f64(
    key_slices: &[&[i64]],
    values: &[f64],
) -> PyResult<GroupByMultiResult> {
    Ok(groupby_multi_f64::<MeanAggF64>(key_slices, values))
}

/// Multi-key groupby min for f64 values.
pub fn multi_groupby_min_f64(
    key_slices: &[&[i64]],
    values: &[f64],
) -> PyResult<GroupByMultiResult> {
    Ok(groupby_multi_f64::<MinAggF64>(key_slices, values))
}

/// Multi-key groupby max for f64 values.
pub fn multi_groupby_max_f64(
    key_slices: &[&[i64]],
    values: &[f64],
) -> PyResult<GroupByMultiResult> {
    Ok(groupby_multi_f64::<MaxAggF64>(key_slices, values))
}

// Public API functions for i64 values

/// Multi-key groupby sum for i64 values.
pub fn multi_groupby_sum_i64(
    key_slices: &[&[i64]],
    values: &[i64],
) -> PyResult<GroupByMultiResult> {
    Ok(groupby_multi_i64::<SumAggI64>(key_slices, values))
}

/// Multi-key groupby mean for i64 values.
pub fn multi_groupby_mean_i64(
    key_slices: &[&[i64]],
    values: &[i64],
) -> PyResult<GroupByMultiResult> {
    Ok(groupby_multi_i64::<MeanAggI64>(key_slices, values))
}

/// Multi-key groupby min for i64 values.
pub fn multi_groupby_min_i64(
    key_slices: &[&[i64]],
    values: &[i64],
) -> PyResult<GroupByMultiResult> {
    Ok(groupby_multi_i64::<MinAggI64>(key_slices, values))
}

/// Multi-key groupby max for i64 values.
pub fn multi_groupby_max_i64(
    key_slices: &[&[i64]],
    values: &[i64],
) -> PyResult<GroupByMultiResult> {
    Ok(groupby_multi_i64::<MaxAggI64>(key_slices, values))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_row_hash_deterministic() {
        let col1 = vec![1i64, 2, 3];
        let col2 = vec![10i64, 20, 30];
        let key_slices: Vec<&[i64]> = vec![&col1, &col2];

        let h0 = row_hash(&key_slices, 0);
        let h0_again = row_hash(&key_slices, 0);
        assert_eq!(h0, h0_again);

        let h1 = row_hash(&key_slices, 1);
        assert_ne!(h0, h1); // Different rows should (usually) have different hashes
    }

    #[test]
    fn test_keys_equal() {
        let col1 = vec![1i64, 2, 1];
        let col2 = vec![10i64, 20, 10];
        let key_slices: Vec<&[i64]> = vec![&col1, &col2];

        let stored: CompositeKey = smallvec::smallvec![1, 10];
        assert!(keys_equal(&stored, &key_slices, 0));
        assert!(!keys_equal(&stored, &key_slices, 1));
        assert!(keys_equal(&stored, &key_slices, 2)); // Same keys as row 0
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
        let key: CompositeKey = smallvec::smallvec![1, 2, 3, 4];
        assert!(!key.spilled()); // Should be inline

        let key_large: CompositeKey = smallvec::smallvec![1, 2, 3, 4, 5];
        assert!(key_large.spilled()); // Should spill to heap
    }
}
