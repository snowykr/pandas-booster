use crate::radix_sort::{
    multi_key_sort_perm_with_profile, multi_key_sort_perm_with_proof, radix_sort_perm_by_u32,
    radix_sort_perm_by_u64,
};

use super::partition::SMALL_DIRECT_THRESHOLD_ELEMS;
use super::result::GroupByMultiResult;

const RADIX_SORT_THRESHOLD: usize = 2048;

#[derive(Debug, Clone, Default)]
pub(super) struct SortPhaseProfile {
    pub sort_key_construction_s: f64,
    pub radix_sort_s: f64,
    pub sorted_materialization_s: f64,
    pub selected_sort_strategy: &'static str,
    pub sort_key_bit_widths: Vec<u32>,
}

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
        perm = multi_key_sort_perm_with_proof(keys_flat, n_keys).0;
    }

    if n_groups.saturating_mul(n_keys) > SMALL_DIRECT_THRESHOLD_ELEMS {
        result.perm = Some(perm);
        return;
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

pub(super) fn sort_groupby_result_profiled<V: Copy>(
    result: &mut GroupByMultiResult<V>,
) -> SortPhaseProfile {
    use std::time::Instant;

    if result.values.is_empty() {
        return SortPhaseProfile {
            selected_sort_strategy: "empty",
            ..SortPhaseProfile::default()
        };
    }

    let n_keys = result.n_keys;
    let n_groups = result.values.len();

    debug_assert_eq!(result.keys_flat.len(), n_groups * n_keys);

    let keys_flat = &result.keys_flat;
    let mut perm: Vec<usize> = (0..n_groups).collect();
    let mut sort_key_construction_s = 0.0;
    let mut radix_sort_s = 0.0;
    let mut selected_sort_strategy = "small_comparator";
    let mut sort_key_bit_widths = Vec::new();

    if n_groups < RADIX_SORT_THRESHOLD {
        let sort_start = Instant::now();
        perm.sort_unstable_by(|&i, &j| {
            let k_i = &keys_flat[i * n_keys..(i + 1) * n_keys];
            let k_j = &keys_flat[j * n_keys..(j + 1) * n_keys];
            k_i.cmp(k_j).then(i.cmp(&j))
        });
        radix_sort_s = sort_start.elapsed().as_secs_f64();
    } else {
        let (sorted_perm, proof, construction_s, sort_s) =
            multi_key_sort_perm_with_profile(keys_flat, n_keys);
        sort_key_construction_s += construction_s;
        radix_sort_s += sort_s;
        selected_sort_strategy = proof.strategy.label();
        sort_key_bit_widths = proof.bit_widths;
        perm = sorted_perm;
    }

    if n_groups.saturating_mul(n_keys) > SMALL_DIRECT_THRESHOLD_ELEMS {
        result.perm = Some(perm);
        return SortPhaseProfile {
            sort_key_construction_s,
            radix_sort_s,
            sorted_materialization_s: 0.0,
            selected_sort_strategy,
            sort_key_bit_widths,
        };
    }

    let materialize_start = Instant::now();
    let mut sorted_keys = Vec::with_capacity(result.keys_flat.len());
    let mut sorted_values = Vec::with_capacity(result.values.len());

    for &idx in &perm {
        sorted_keys.extend_from_slice(&keys_flat[idx * n_keys..(idx + 1) * n_keys]);
        sorted_values.push(result.values[idx]);
    }

    result.keys_flat = sorted_keys;
    result.values = sorted_values;
    result.perm = None;

    SortPhaseProfile {
        sort_key_construction_s,
        radix_sort_s,
        sorted_materialization_s: materialize_start.elapsed().as_secs_f64(),
        selected_sort_strategy,
        sort_key_bit_widths,
    }
}
