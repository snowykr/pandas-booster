use ahash::AHashMap;
use rayon::prelude::*;

use crate::aggregation::Aggregator;

use super::keys::RadixKeyOps;
use super::order::reorder_result_by_first_seen_u32;
use super::partition::{stable_scatter_by_partition, NUM_PARTITIONS};
use super::result::GroupByMultiResult;

type PartitionResults<K, V> = Vec<(Vec<(K, V)>, Vec<u32>)>;

pub(super) fn radix_groupby_engine_firstseen_u32<Ops, T, A, O>(
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

    let mut hashes = vec![0u64; n_rows];
    hashes.par_iter_mut().enumerate().for_each(|(row, h_out)| {
        *h_out = ops.compute_hash(key_slices, row);
    });

    let (perm, offsets) = stable_scatter_by_partition(&hashes);
    let partition_results: PartitionResults<Ops::Key, O> = (0..NUM_PARTITIONS)
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
            let mut aggs = aggs.into_iter().map(Some).collect::<Vec<_>>();
            for (k, gid) in gid_map {
                let g = gid as usize;
                let agg = aggs[g]
                    .take()
                    .expect("first-seen group id should reference one accumulator");
                out_pairs.push((k, agg.finalize_owned()));
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
