use crate::radix_sort::{
    i64_to_sortable_u64, radix_sort_perm_by_u32, radix_sort_perm_by_u64,
    radix_sort_perm_by_u64_for_indices_par,
};

use super::partition::SMALL_DIRECT_THRESHOLD_ELEMS;
use super::result::GroupByMultiResult;

const RADIX_SORT_THRESHOLD: usize = 2048;

pub(super) fn reorder_result_by_first_seen_u32<V: Copy>(
    result: &mut GroupByMultiResult<V>,
    first_seen: &[u32],
) {
    if result.values.is_empty() {
        return;
    }
    debug_assert_eq!(result.values.len(), first_seen.len());

    let n_keys = result.n_keys;
    let n_groups = result.values.len();
    debug_assert_eq!(result.keys_flat.len(), n_groups * n_keys);

    let perm = radix_sort_perm_by_u32(first_seen);

    let keys_flat = &result.keys_flat;
    let values = &result.values;

    if n_groups.saturating_mul(n_keys) > SMALL_DIRECT_THRESHOLD_ELEMS {
        result.perm = Some(perm);
        return;
    }

    let mut sorted_keys = Vec::with_capacity(keys_flat.len());
    let mut sorted_values = Vec::with_capacity(values.len());

    for &g in &perm {
        sorted_keys.extend_from_slice(&keys_flat[g * n_keys..(g + 1) * n_keys]);
        sorted_values.push(values[g]);
    }

    result.keys_flat = sorted_keys;
    result.values = sorted_values;
    result.perm = None;
}

pub(super) fn reorder_result_by_first_seen_u64<V: Copy>(
    result: &mut GroupByMultiResult<V>,
    first_seen: &[u64],
) {
    if result.values.is_empty() {
        return;
    }
    debug_assert_eq!(result.values.len(), first_seen.len());

    let n_keys = result.n_keys;
    let n_groups = result.values.len();
    debug_assert_eq!(result.keys_flat.len(), n_groups * n_keys);

    let perm = radix_sort_perm_by_u64(first_seen);

    let keys_flat = &result.keys_flat;
    let values = &result.values;

    if n_groups.saturating_mul(n_keys) > SMALL_DIRECT_THRESHOLD_ELEMS {
        result.perm = Some(perm);
        return;
    }

    let mut sorted_keys = Vec::with_capacity(keys_flat.len());
    let mut sorted_values = Vec::with_capacity(values.len());

    for &g in &perm {
        sorted_keys.extend_from_slice(&keys_flat[g * n_keys..(g + 1) * n_keys]);
        sorted_values.push(values[g]);
    }

    result.keys_flat = sorted_keys;
    result.values = sorted_values;
    result.perm = None;
}

pub(super) fn sort_groupby_result<V: Copy>(result: &mut GroupByMultiResult<V>) {
    if result.values.is_empty() {
        return;
    }

    let n_keys = result.n_keys;
    let n_groups = result.values.len();

    debug_assert_eq!(result.keys_flat.len(), n_groups * n_keys);

    let keys_flat = &result.keys_flat;
    let mut perm: Vec<usize> = (0..n_groups).collect();

    if n_groups < RADIX_SORT_THRESHOLD {
        perm.sort_unstable_by(|&i, &j| {
            let k_i = &keys_flat[i * n_keys..(i + 1) * n_keys];
            let k_j = &keys_flat[j * n_keys..(j + 1) * n_keys];
            k_i.cmp(k_j).then(i.cmp(&j))
        });
    } else {
        for col in (0..n_keys).rev() {
            let mut col_keys = Vec::with_capacity(n_groups);
            for group in 0..n_groups {
                let key = keys_flat[group * n_keys + col];
                col_keys.push(i64_to_sortable_u64(key));
            }
            perm = radix_sort_perm_by_u64_for_indices_par(&col_keys, &perm);
        }
    }

    let mut sorted_keys = Vec::with_capacity(result.keys_flat.len());
    let mut sorted_values = Vec::with_capacity(result.values.len());

    for &idx in &perm {
        sorted_keys.extend_from_slice(&keys_flat[idx * n_keys..(idx + 1) * n_keys]);
        sorted_values.push(result.values[idx]);
    }

    result.keys_flat = sorted_keys;
    result.values = sorted_values;
    result.perm = None;
}
