use ahash::AHashMap;
use rayon::prelude::*;
use std::time::Instant;

use crate::aggregation::Aggregator;

use super::keys::RadixKeyOps;
use super::partition::{stable_scatter_by_partition, NUM_PARTITIONS};
use super::result::{GroupByMultiResult, MultiKeySortedPhaseProfile};

pub(super) fn radix_groupby_engine<Ops, T, A, O>(
    ops: Ops,
    key_slices: &[&[i64]],
    values: &[T],
    mut profile: Option<&mut MultiKeySortedPhaseProfile>,
) -> Result<GroupByMultiResult<O>, String>
where
    Ops: RadixKeyOps,
    T: Copy + Send + Sync,
    O: Copy + Send + Sync,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    #[cfg(test)]
    super::test_support::observe_hash_aggregation();
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
    let hash_start = profile.as_ref().map(|_| Instant::now());
    let mut hashes = vec![0u64; n_rows];
    hashes.par_iter_mut().enumerate().for_each(|(row, h_out)| {
        let h = ops.compute_hash(key_slices, row);
        *h_out = h;
    });
    if let Some((profile, start)) = profile.as_deref_mut().zip(hash_start) {
        profile.hash_build_s = start.elapsed().as_secs_f64();
    }

    // Phase 2/3: deterministic stable scatter by partition.
    let scatter_start = profile.as_ref().map(|_| Instant::now());
    let (perm, offsets) = stable_scatter_by_partition(&hashes);
    if let Some((profile, start)) = profile.as_deref_mut().zip(scatter_start) {
        profile.partition_scatter_s = start.elapsed().as_secs_f64();
    }

    // Phase 4: Per-partition aggregation (no merge!)
    let aggregation_start = profile.as_ref().map(|_| Instant::now());
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
                .map(|(k, agg)| (k, agg.finalize_owned()))
                .collect()
        })
        .collect();
    if let Some((profile, start)) = profile.as_deref_mut().zip(aggregation_start) {
        profile.partition_aggregation_s = start.elapsed().as_secs_f64();
    }

    // Flatten results
    let flatten_start = profile.as_ref().map(|_| Instant::now());
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
    if let Some((profile, start)) = profile.zip(flatten_start) {
        profile.flatten_s = start.elapsed().as_secs_f64();
        profile.partial_group_total = total_groups;
    }

    Ok(GroupByMultiResult {
        keys_flat,
        n_keys,
        values: out_values,
        perm: None,
    })
}
