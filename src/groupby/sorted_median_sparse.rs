use ahash::AHashMap;
use pyo3::prelude::*;
use std::time::Instant;

use super::result::GroupByResultF64;
use super::sorted_median::{
    checked_total, count_overflow, materialize_group_medians, remap_incomplete_error,
    ProfiledSortedMedianDirect, SortedMedianDirectStats, SortedMedianValue,
};

#[cfg(test)]
use super::sorted_median::increment_scatter_write_count;

pub(super) fn groupby_median_sparse_direct_with_stats<T>(
    keys: &[i64],
    values: &[T],
) -> PyResult<ProfiledSortedMedianDirect>
where
    T: SortedMedianValue,
{
    let unique_build_start = Instant::now();
    let mut gid_by_key = AHashMap::new();
    let mut unique_keys: Vec<i64> = Vec::new();
    let mut counts_by_old_gid: Vec<usize> = Vec::new();
    let mut old_gids_by_row: Vec<usize> = Vec::with_capacity(keys.len());

    for row in 0..keys.len() {
        let key = keys[row];
        let old_gid = match gid_by_key.get(&key).copied() {
            Some(gid) => gid,
            None => {
                let gid = unique_keys.len();
                gid_by_key.insert(key, gid);
                unique_keys.push(key);
                counts_by_old_gid.push(0);
                gid
            }
        };
        old_gids_by_row.push(old_gid);

        let value = values[row];
        if value.is_accepted() {
            counts_by_old_gid[old_gid] = counts_by_old_gid[old_gid]
                .checked_add(1)
                .ok_or_else(count_overflow)?;
        }
    }
    let unique_build_s = unique_build_start.elapsed().as_secs_f64();
    let count_s = 0.0;

    let key_sort_start = Instant::now();
    let group_count = unique_keys.len();
    let mut unique_pairs: Vec<(i64, usize)> = unique_keys
        .iter()
        .copied()
        .enumerate()
        .map(|(old_gid, key)| (key, old_gid))
        .collect();
    unique_pairs.sort_unstable_by_key(|&(key, _)| key);

    let mut sorted_keys = Vec::with_capacity(group_count);
    let mut sorted_gid_by_old_gid = vec![0usize; group_count];
    let mut counts_by_sorted_gid = vec![0usize; group_count];

    for (sorted_gid, (key, old_gid)) in unique_pairs.into_iter().enumerate() {
        if old_gid >= group_count {
            return Err(remap_incomplete_error());
        }
        sorted_keys.push(key);
        sorted_gid_by_old_gid[old_gid] = sorted_gid;
        counts_by_sorted_gid[sorted_gid] = counts_by_old_gid[old_gid];
    }
    let key_sort_s = key_sort_start.elapsed().as_secs_f64();

    let buffer_setup_start = Instant::now();
    let accepted_value_count = checked_total(&counts_by_sorted_gid)?;
    let _ = accepted_value_count;
    let mut grouped_values: Vec<Vec<T>> = counts_by_sorted_gid
        .iter()
        .copied()
        .map(Vec::with_capacity)
        .collect();
    #[cfg(test)]
    let mut scatter_write_count = 0usize;
    let buffer_setup_s = buffer_setup_start.elapsed().as_secs_f64();

    let scatter_start = Instant::now();
    for row in 0..keys.len() {
        let value = values[row];
        if value.is_accepted() {
            let old_gid = old_gids_by_row[row];
            let gid = sorted_gid_by_old_gid
                .get(old_gid)
                .copied()
                .ok_or_else(remap_incomplete_error)?;
            grouped_values[gid].push(value);
            #[cfg(test)]
            let () = increment_scatter_write_count(&mut scatter_write_count)?;
        }
    }

    for gid in 0..group_count {
        if grouped_values[gid].len() != counts_by_sorted_gid[gid] {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "median scatter did not fill checked group buffer",
            ));
        }
    }
    let scatter_s = scatter_start.elapsed().as_secs_f64();

    let median_select_start = Instant::now();
    let medians = materialize_group_medians(&mut grouped_values);
    let median_select_s = median_select_start.elapsed().as_secs_f64();

    let stats = SortedMedianDirectStats {
        partial_group_total: group_count,
        final_group_count: group_count,
        #[cfg(test)]
        accepted_value_count,
        #[cfg(test)]
        value_buffer_len: grouped_values.iter().map(Vec::len).sum(),
        #[cfg(test)]
        scatter_write_count,
        unique_build_s,
        key_sort_s,
        count_s,
        buffer_setup_s,
        scatter_s,
        median_select_s,
    };

    Ok(ProfiledSortedMedianDirect {
        result: GroupByResultF64 {
            keys: sorted_keys,
            values: medians,
        },
        stats,
    })
}
