use std::mem::size_of;

use super::routing::SortedMedianValueKind;

pub(in crate::groupby) const SORTED_MEDIAN_SPARSE_MAX_EXACT_GROUPS: usize = 4_096;
const SORTED_MEDIAN_HASH_TABLE_SLACK_NUMERATOR: usize = 2;
const SORTED_MEDIAN_ABSOLUTE_MEMORY_CAP_BYTES: usize = 512 * 1024 * 1024;
const SORTED_MEDIAN_INPUT_MEMORY_RATIO_CAP: f64 = 4.0;
const SORTED_MEDIAN_MIN_DIRECT_MEMORY_BUDGET_BYTES: usize = 1 << 20;
pub(in crate::groupby) const SORTED_MEDIAN_DIRECT_SAFETY_MARGIN: f64 = 0.05;

#[derive(Debug, Clone, Copy)]
pub(in crate::groupby) struct RouteWorkEstimate {
    pub(in crate::groupby) total_score: f64,
}

fn checked_add(a: usize, b: usize) -> Option<usize> {
    a.checked_add(b)
}

fn checked_mul(a: usize, b: usize) -> Option<usize> {
    a.checked_mul(b)
}

pub(in crate::groupby) fn checked_input_bytes(
    n_rows: usize,
    value_kind: SortedMedianValueKind,
) -> Option<usize> {
    checked_mul(n_rows, size_of::<i64>() + value_kind.element_size())
}

fn add_term(total: &mut Option<usize>, count: usize, size: usize) {
    *total = total
        .and_then(|current| checked_mul(count, size).and_then(|term| checked_add(current, term)));
}

pub(in crate::groupby) fn estimate_dense_direct_peak_bytes(
    span: usize,
    groups_upper_bound: usize,
    accepted_values: usize,
    value_kind: SortedMedianValueKind,
) -> Option<usize> {
    let mut total = Some(0usize);
    add_term(&mut total, span, size_of::<bool>());
    add_term(&mut total, span, size_of::<usize>());
    add_term(&mut total, span, size_of::<Vec<i64>>());
    add_term(&mut total, accepted_values, value_kind.element_size());
    add_term(&mut total, groups_upper_bound, size_of::<i64>());
    add_term(&mut total, groups_upper_bound, size_of::<usize>());
    add_term(&mut total, groups_upper_bound, size_of::<(i64, Vec<i64>)>());
    add_term(&mut total, groups_upper_bound, size_of::<f64>());
    total.and_then(|bytes| checked_mul(bytes, 2))
}

pub(in crate::groupby) fn estimate_sparse_direct_peak_bytes(
    n_rows: usize,
    groups: usize,
    accepted_values: usize,
    value_kind: SortedMedianValueKind,
) -> Option<usize> {
    let mut total = Some(0usize);
    add_term(&mut total, n_rows, size_of::<usize>());
    add_term(
        &mut total,
        groups.saturating_mul(SORTED_MEDIAN_HASH_TABLE_SLACK_NUMERATOR),
        size_of::<(i64, usize)>(),
    );
    add_term(&mut total, groups, size_of::<i64>());
    add_term(&mut total, groups, size_of::<usize>());
    add_term(&mut total, groups, size_of::<(i64, usize)>());
    add_term(&mut total, groups, size_of::<usize>());
    add_term(&mut total, groups, size_of::<usize>());
    add_term(&mut total, groups, size_of::<Vec<i64>>());
    add_term(&mut total, accepted_values, value_kind.element_size());
    add_term(&mut total, groups, size_of::<i64>() + size_of::<f64>());
    total.and_then(|bytes| checked_mul(bytes, 2))
}

pub(in crate::groupby) fn memory_within_caps(
    estimated_bytes: Option<usize>,
    input_bytes: usize,
) -> bool {
    let Some(estimated_bytes) = estimated_bytes else {
        return false;
    };
    if estimated_bytes > SORTED_MEDIAN_ABSOLUTE_MEMORY_CAP_BYTES {
        return false;
    }

    let ratio_budget = input_bytes as f64 * SORTED_MEDIAN_INPUT_MEMORY_RATIO_CAP;
    let small_input_floor = SORTED_MEDIAN_MIN_DIRECT_MEMORY_BUDGET_BYTES as f64;
    (estimated_bytes as f64) <= ratio_budget.max(small_input_floor)
}

pub(in crate::groupby) fn estimate_dense_work(
    n_rows: usize,
    accepted_values: usize,
    span: usize,
    max_group_rows: usize,
) -> RouteWorkEstimate {
    let total_score = 1.25 * n_rows as f64
        + span as f64
        + accepted_values as f64
        + accepted_values as f64
        + max_group_rows as f64;
    RouteWorkEstimate { total_score }
}

pub(in crate::groupby) fn estimate_sparse_work(
    n_rows: usize,
    accepted_values: usize,
    groups: usize,
    max_group_rows: usize,
) -> RouteWorkEstimate {
    let total_score = n_rows as f64
        + 1.25 * n_rows as f64
        + groups as f64 * (groups.max(2) as f64).log2()
        + accepted_values as f64
        + accepted_values as f64
        + max_group_rows as f64;
    RouteWorkEstimate { total_score }
}

pub(in crate::groupby) fn estimate_existing_work(
    n_rows: usize,
    accepted_values: usize,
    groups: usize,
) -> RouteWorkEstimate {
    let total_score = 5.0 * n_rows as f64
        + groups as f64 * (groups.max(2) as f64).log2()
        + accepted_values as f64;
    RouteWorkEstimate { total_score }
}

pub(in crate::groupby) fn estimate_partitioned_work(
    n_rows: usize,
    accepted_values: usize,
    groups: usize,
) -> RouteWorkEstimate {
    let total_score = 4.0 * n_rows as f64
        + groups as f64 * (groups.max(2) as f64).log2()
        + accepted_values as f64;
    RouteWorkEstimate { total_score }
}
