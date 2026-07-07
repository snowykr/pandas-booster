use pyo3::prelude::*;
use std::time::Instant;

use super::result::GroupByResultF64;
#[cfg(test)]
use super::routing::dense_sorted_median_key_range_len;
use super::sorted_median::{
    checked_total, count_overflow, materialize_group_medians_from_pairs, remap_incomplete_error,
    ProfiledSortedMedianDirect, SortedMedianDirectStats, SortedMedianValue,
};

#[cfg(test)]
use super::sorted_median::increment_scatter_write_count;

#[derive(Debug, Clone, Copy)]
pub(super) struct DenseKeyRange {
    min_key: i64,
    len: usize,
}

impl DenseKeyRange {
    pub(super) fn new(min_key: i64, len: usize) -> Self {
        Self { min_key, len }
    }
    #[cfg(test)]
    pub(super) fn from_keys(keys: &[i64]) -> Option<Self> {
        let (&first, rest) = keys.split_first()?;
        let mut min_key = first;
        let mut max_key = first;

        for key in rest.iter().copied() {
            min_key = min_key.min(key);
            max_key = max_key.max(key);
        }

        let len = dense_sorted_median_key_range_len(keys)?;

        Some(Self { min_key, len })
    }

    fn index_for(self, key: i64) -> PyResult<usize> {
        let offset = key
            .checked_sub(self.min_key)
            .ok_or_else(remap_incomplete_error)?;
        let index = usize::try_from(offset).map_err(|_| remap_incomplete_error())?;
        if index >= self.len {
            return Err(remap_incomplete_error());
        }
        Ok(index)
    }

    fn key_for(self, index: usize) -> PyResult<i64> {
        let offset = i64::try_from(index).map_err(|_| remap_incomplete_error())?;
        self.min_key
            .checked_add(offset)
            .ok_or_else(remap_incomplete_error)
    }
}

pub(super) fn groupby_median_dense_direct_with_stats<T>(
    keys: &[i64],
    values: &[T],
    dense_range: DenseKeyRange,
) -> PyResult<ProfiledSortedMedianDirect>
where
    T: SortedMedianValue,
{
    let unique_build_start = Instant::now();
    let mut seen_by_gid = vec![false; dense_range.len];
    let mut counts_by_gid = vec![0usize; dense_range.len];
    let mut group_count = 0usize;

    for row in 0..keys.len() {
        let gid = dense_range.index_for(keys[row])?;
        if !seen_by_gid[gid] {
            seen_by_gid[gid] = true;
            group_count = group_count.checked_add(1).ok_or_else(count_overflow)?;
        }

        let value = values[row];
        if value.is_accepted() {
            counts_by_gid[gid] = counts_by_gid[gid]
                .checked_add(1)
                .ok_or_else(count_overflow)?;
        }
    }
    let unique_build_s = unique_build_start.elapsed().as_secs_f64();

    let key_sort_start = Instant::now();
    let mut sorted_keys = Vec::with_capacity(group_count);
    let mut seen_gids = Vec::with_capacity(group_count);
    for (gid, seen) in seen_by_gid.iter().copied().enumerate() {
        if seen {
            sorted_keys.push(dense_range.key_for(gid)?);
            seen_gids.push(gid);
        }
    }
    let key_sort_s = key_sort_start.elapsed().as_secs_f64();

    let buffer_setup_start = Instant::now();
    let accepted_value_count = checked_total(&counts_by_gid)?;
    let _ = accepted_value_count;
    let mut grouped_values: Vec<Vec<T>> = counts_by_gid
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
            let gid = dense_range.index_for(keys[row])?;
            grouped_values[gid].push(value);
            #[cfg(test)]
            let () = increment_scatter_write_count(&mut scatter_write_count)?;
        }
    }

    for gid in seen_gids.iter().copied() {
        if grouped_values[gid].len() != counts_by_gid[gid] {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "median scatter did not fill checked group buffer",
            ));
        }
    }
    let scatter_s = scatter_start.elapsed().as_secs_f64();

    let median_select_start = Instant::now();
    let mut final_groups = Vec::with_capacity(group_count);
    for gid in seen_gids {
        final_groups.push((
            dense_range.key_for(gid)?,
            std::mem::take(&mut grouped_values[gid]),
        ));
    }
    let medians = materialize_group_medians_from_pairs(&mut final_groups);
    let median_select_s = median_select_start.elapsed().as_secs_f64();

    let stats = SortedMedianDirectStats {
        partial_group_total: group_count,
        final_group_count: group_count,
        #[cfg(test)]
        accepted_value_count,
        #[cfg(test)]
        value_buffer_len: final_groups.iter().map(|(_, values)| values.len()).sum(),
        #[cfg(test)]
        scatter_write_count,
        unique_build_s,
        key_sort_s,
        count_s: 0.0,
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
