use ahash::AHashMap;
use rayon::prelude::*;
use std::time::Instant;

use crate::aggregation::Aggregator;

use super::keys::{CompositeKeyOps, FixedKeyOps, RadixKeyOps};
use super::order::sort_groupby_result_profiled;
use super::partition::{stable_scatter_by_partition, NUM_PARTITIONS};
use super::result::{GroupByMultiResult, MultiKeySortedPhaseProfile, ProfiledGroupByMultiResult};

fn profile_radix_groupby_engine<Ops, T, A, O>(
    ops: Ops,
    key_slices: &[&[i64]],
    values: &[T],
) -> Result<ProfiledGroupByMultiResult<O>, String>
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
        return Ok(ProfiledGroupByMultiResult {
            result: GroupByMultiResult {
                keys_flat: Vec::new(),
                n_keys,
                values: Vec::new(),
                perm: None,
            },
            profile: MultiKeySortedPhaseProfile {
                route_label: "hash_first",
                hash_build_s: 0.0,
                partition_scatter_s: 0.0,
                partition_aggregation_s: 0.0,
                flatten_s: 0.0,
                sort_key_construction_s: 0.0,
                radix_sort_s: 0.0,
                sorted_materialization_s: 0.0,
                sort_first_permutation_s: 0.0,
                sort_first_segment_scan_s: 0.0,
                sort_first_segment_scan_count: 0,
                partial_group_total: 0,
                final_group_count: 0,
            },
        });
    }

    let hash_start = Instant::now();
    let mut hashes = vec![0u64; n_rows];
    hashes.par_iter_mut().enumerate().for_each(|(row, h_out)| {
        *h_out = ops.compute_hash(key_slices, row);
    });
    let hash_build_s = hash_start.elapsed().as_secs_f64();

    let scatter_start = Instant::now();
    let (perm, offsets) = stable_scatter_by_partition(&hashes);
    let partition_scatter_s = scatter_start.elapsed().as_secs_f64();

    let aggregation_start = Instant::now();
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
    let partition_aggregation_s = aggregation_start.elapsed().as_secs_f64();

    let flatten_start = Instant::now();
    let partial_group_total: usize = partition_results.iter().map(Vec::len).sum();
    let n_keys = ops.n_keys();
    let mut keys_flat = Vec::with_capacity(partial_group_total * n_keys);
    let mut out_values: Vec<O> = Vec::with_capacity(partial_group_total);

    for partition in partition_results {
        for (key, val) in partition {
            Ops::push_flat(&mut keys_flat, &key);
            out_values.push(val);
        }
    }
    let flatten_s = flatten_start.elapsed().as_secs_f64();

    let mut result = GroupByMultiResult {
        keys_flat,
        n_keys,
        values: out_values,
        perm: None,
    };
    let sort_profile = sort_groupby_result_profiled(&mut result);
    let final_group_count = result.values.len();

    Ok(ProfiledGroupByMultiResult {
        result,
        profile: MultiKeySortedPhaseProfile {
            route_label: "hash_first",
            hash_build_s,
            partition_scatter_s,
            partition_aggregation_s,
            flatten_s,
            sort_key_construction_s: sort_profile.sort_key_construction_s,
            radix_sort_s: sort_profile.radix_sort_s,
            sorted_materialization_s: sort_profile.sorted_materialization_s,
            sort_first_permutation_s: 0.0,
            sort_first_segment_scan_s: 0.0,
            sort_first_segment_scan_count: 0,
            partial_group_total,
            final_group_count,
        },
    })
}

fn profile_radix_groupby_dispatch<T, A, O>(
    key_slices: &[&[i64]],
    values: &[T],
) -> Result<ProfiledGroupByMultiResult<O>, String>
where
    T: Copy + Send + Sync,
    O: Copy + Send + Sync,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    match key_slices.len() {
        1 => profile_radix_groupby_engine::<FixedKeyOps<1>, T, A, O>(
            FixedKeyOps::<1>::new(1),
            key_slices,
            values,
        ),
        2 => profile_radix_groupby_engine::<FixedKeyOps<2>, T, A, O>(
            FixedKeyOps::<2>::new(2),
            key_slices,
            values,
        ),
        3 => profile_radix_groupby_engine::<FixedKeyOps<3>, T, A, O>(
            FixedKeyOps::<3>::new(3),
            key_slices,
            values,
        ),
        4 => profile_radix_groupby_engine::<FixedKeyOps<4>, T, A, O>(
            FixedKeyOps::<4>::new(4),
            key_slices,
            values,
        ),
        5 => profile_radix_groupby_engine::<FixedKeyOps<5>, T, A, O>(
            FixedKeyOps::<5>::new(5),
            key_slices,
            values,
        ),
        6 => profile_radix_groupby_engine::<FixedKeyOps<6>, T, A, O>(
            FixedKeyOps::<6>::new(6),
            key_slices,
            values,
        ),
        7 => profile_radix_groupby_engine::<FixedKeyOps<7>, T, A, O>(
            FixedKeyOps::<7>::new(7),
            key_slices,
            values,
        ),
        8 => profile_radix_groupby_engine::<FixedKeyOps<8>, T, A, O>(
            FixedKeyOps::<8>::new(8),
            key_slices,
            values,
        ),
        9 => profile_radix_groupby_engine::<FixedKeyOps<9>, T, A, O>(
            FixedKeyOps::<9>::new(9),
            key_slices,
            values,
        ),
        10 => profile_radix_groupby_engine::<FixedKeyOps<10>, T, A, O>(
            FixedKeyOps::<10>::new(10),
            key_slices,
            values,
        ),
        _ => profile_radix_groupby_engine::<CompositeKeyOps, T, A, O>(
            CompositeKeyOps::new(key_slices.len()),
            key_slices,
            values,
        ),
    }
}

pub(super) fn profile_radix_groupby_sorted<T, A, O>(
    key_slices: &[&[i64]],
    values: &[T],
) -> Result<ProfiledGroupByMultiResult<O>, String>
where
    T: Copy + Send + Sync,
    O: Copy + Send + Sync,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    profile_radix_groupby_dispatch::<T, A, O>(key_slices, values)
}
