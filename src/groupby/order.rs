use pyo3::prelude::*;

use crate::radix_sort::{
    radix_sort_perm_by_i64_par, radix_sort_perm_by_u32, radix_sort_perm_by_u64,
};

use super::result::GroupByResult;

const RADIX_SORT_THRESHOLD: usize = 2048;

pub(super) fn reorder_single_result_by_first_seen_u32<V: Copy>(
    result: &mut GroupByResult<V>,
    first_seen: &[u32],
) {
    if result.values.is_empty() {
        return;
    }
    debug_assert_eq!(result.values.len(), first_seen.len());
    debug_assert_eq!(result.keys.len(), first_seen.len());

    let perm = radix_sort_perm_by_u32(first_seen);

    let keys = &result.keys;
    let values = &result.values;
    let mut sorted_keys = Vec::with_capacity(keys.len());
    let mut sorted_values = Vec::with_capacity(values.len());
    for &g in &perm {
        sorted_keys.push(keys[g]);
        sorted_values.push(values[g]);
    }
    result.keys = sorted_keys;
    result.values = sorted_values;
}

pub(super) fn reorder_single_result_by_first_seen_u64<V: Copy>(
    result: &mut GroupByResult<V>,
    first_seen: &[u64],
) {
    if result.values.is_empty() {
        return;
    }
    debug_assert_eq!(result.values.len(), first_seen.len());
    debug_assert_eq!(result.keys.len(), first_seen.len());

    let perm = radix_sort_perm_by_u64(first_seen);

    let keys = &result.keys;
    let values = &result.values;
    let mut sorted_keys = Vec::with_capacity(keys.len());
    let mut sorted_values = Vec::with_capacity(values.len());
    for &g in &perm {
        sorted_keys.push(keys[g]);
        sorted_values.push(values[g]);
    }
    result.keys = sorted_keys;
    result.values = sorted_values;
}

pub(super) fn reorder_single_result_by_key<V: Copy>(result: &mut GroupByResult<V>) {
    if result.values.is_empty() {
        return;
    }
    debug_assert_eq!(result.keys.len(), result.values.len());

    let keys = &result.keys;
    let values = &result.values;

    let perm = if keys.len() < RADIX_SORT_THRESHOLD {
        let mut perm: Vec<usize> = (0..keys.len()).collect();
        perm.sort_unstable_by(|&i, &j| keys[i].cmp(&keys[j]).then(i.cmp(&j)));
        perm
    } else {
        radix_sort_perm_by_i64_par(keys)
    };

    let mut sorted_keys = Vec::with_capacity(keys.len());
    let mut sorted_values = Vec::with_capacity(values.len());
    for &idx in &perm {
        sorted_keys.push(keys[idx]);
        sorted_values.push(values[idx]);
    }

    result.keys = sorted_keys;
    result.values = sorted_values;
}

pub(super) trait FirstSeenRowIndex: Copy + Ord + Send + Sync {
    fn validate_n_rows(_n_rows: usize) -> PyResult<()> {
        Ok(())
    }

    fn from_row_index(row: usize) -> Self;

    fn reorder_result<V: Copy>(result: &mut GroupByResult<V>, first_seen: &[Self]);
}

impl FirstSeenRowIndex for u32 {
    fn validate_n_rows(n_rows: usize) -> PyResult<()> {
        if n_rows > (u32::MAX as usize) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "idx32 kernel requires n_rows <= u32::MAX",
            ));
        }
        Ok(())
    }

    fn from_row_index(row: usize) -> Self {
        row as u32
    }

    fn reorder_result<V: Copy>(result: &mut GroupByResult<V>, first_seen: &[Self]) {
        reorder_single_result_by_first_seen_u32(result, first_seen);
    }
}

impl FirstSeenRowIndex for u64 {
    fn from_row_index(row: usize) -> Self {
        row as u64
    }

    fn reorder_result<V: Copy>(result: &mut GroupByResult<V>, first_seen: &[Self]) {
        reorder_single_result_by_first_seen_u64(result, first_seen);
    }
}

pub(super) struct FirstSeenMaterializedResult<I, V> {
    pub(super) result: GroupByResult<V>,
    pub(super) first_seen: Vec<I>,
}

pub(super) fn finalize_deterministic_firstseen_result<I, V: Copy>(
    mut materialized: FirstSeenMaterializedResult<I, V>,
) -> GroupByResult<V>
where
    I: FirstSeenRowIndex,
{
    I::reorder_result(&mut materialized.result, &materialized.first_seen);
    materialized.result
}
