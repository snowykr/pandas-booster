use ahash::AHashMap;
use rayon::prelude::*;

use crate::aggregation::Aggregator;

use super::keys::RadixKeyOps;
use super::partition::{stable_scatter_by_partition, NUM_PARTITIONS};
use super::result::GroupByMultiResult;

pub(super) fn radix_groupby_engine<Ops, T, A, O>(
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
                .map(|(k, agg)| (k, agg.finalize_owned()))
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
