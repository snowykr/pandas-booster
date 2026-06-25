use ahash::{AHashMap, AHashSet};
use std::mem::size_of;

const PARTITIONED_ENGINE_SAMPLE_SIZE: usize = 16_384;
const PARTITIONED_ENGINE_MIN_SAMPLE_UNIQUES: usize = 4_096;
const SORTED_MEDIAN_SPARSE_MAX_EXACT_GROUPS: usize = 4_096;
const SORTED_MEDIAN_HASH_TABLE_SLACK_NUMERATOR: usize = 2;
pub(super) const SORTED_MEDIAN_ABSOLUTE_MEMORY_CAP_BYTES: usize = 512 * 1024 * 1024;
pub(super) const SORTED_MEDIAN_INPUT_MEMORY_RATIO_CAP: f64 = 4.0;
const SORTED_MEDIAN_MIN_DIRECT_MEMORY_BUDGET_BYTES: usize = 1 << 20;
pub(super) const SORTED_MEDIAN_DIRECT_SAFETY_MARGIN: f64 = 0.05;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SortedMedianValueKind {
    F64,
    I64,
}

impl SortedMedianValueKind {
    fn element_size(self) -> usize {
        match self {
            Self::F64 => size_of::<f64>(),
            Self::I64 => size_of::<i64>(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]

pub(super) enum SortedMedianRouteKind {
    DirectDense,
    DirectSparse,
    ExistingSortedFallback,
    PartitionedFallback,
}

impl SortedMedianRouteKind {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::DirectDense => "direct_dense",
            Self::DirectSparse => "direct_sparse",
            Self::ExistingSortedFallback => "existing_sorted_fallback",
            Self::PartitionedFallback => "partitioned_fallback",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]

pub(super) enum SortedMedianRouteReason {
    DenseSpanMemorySafe,
    SparseExactLowCardinalityMemorySafe,

    EmptyInput,
    HighCardinalityPartitionedSample,
    DenseSpanOverflow,

    SparseExactGroupsExceedCap,
    SparseMemoryCapExceeded,

    FallbackSafetyMarginNotMet,

    CheckedArithmeticOverflow,
}

impl SortedMedianRouteReason {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::DenseSpanMemorySafe => "dense_span_memory_safe",
            Self::SparseExactLowCardinalityMemorySafe => "sparse_exact_low_cardinality_memory_safe",

            Self::EmptyInput => "empty_input",
            Self::HighCardinalityPartitionedSample => "high_cardinality_partitioned_sample",
            Self::DenseSpanOverflow => "dense_span_overflow",

            Self::SparseExactGroupsExceedCap => "sparse_exact_groups_exceed_cap",
            Self::SparseMemoryCapExceeded => "sparse_memory_cap_exceeded",

            Self::FallbackSafetyMarginNotMet => "fallback_safety_margin_not_met",

            Self::CheckedArithmeticOverflow => "checked_arithmetic_overflow",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct RouteWorkEstimate {
    total_score: f64,
}

#[derive(Debug, Clone)]
pub(super) struct SortedMedianRouteDecision {
    pub(super) kind: SortedMedianRouteKind,
    pub(super) reason: SortedMedianRouteReason,
    pub(super) dense_span: Option<usize>,
    pub(super) dense_min_key: Option<i64>,
}

impl SortedMedianRouteDecision {
    fn new(kind: SortedMedianRouteKind, reason: SortedMedianRouteReason) -> Self {
        Self {
            kind,
            reason,
            dense_span: None,
            dense_min_key: None,
        }
    }

    #[cfg(test)]
    pub(super) fn is_direct(&self) -> bool {
        matches!(
            self.kind,
            SortedMedianRouteKind::DirectDense | SortedMedianRouteKind::DirectSparse
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct DenseSpan {
    min_key: i64,
    len: usize,
}

#[derive(Debug)]
struct ExactKeyStats {
    groups: usize,
    max_group_rows: usize,
}

fn estimate_sample_unique_keys(keys: &[i64]) -> usize {
    let sample_size = keys.len().min(PARTITIONED_ENGINE_SAMPLE_SIZE);
    if sample_size == 0 {
        return 0;
    }

    let stride = keys.len().div_ceil(sample_size);
    let mut seen = AHashSet::with_capacity(sample_size);
    let mut row = 0usize;
    let mut sampled = 0usize;

    while row < keys.len() && sampled < sample_size {
        seen.insert(keys[row]);
        row += stride;
        sampled += 1;
    }

    seen.len()
}

#[inline]
pub(super) fn should_use_partitioned_firstseen_engine(keys: &[i64]) -> bool {
    let sample_size = keys.len().min(PARTITIONED_ENGINE_SAMPLE_SIZE);
    sample_size > PARTITIONED_ENGINE_MIN_SAMPLE_UNIQUES
        && estimate_sample_unique_keys(keys) >= PARTITIONED_ENGINE_MIN_SAMPLE_UNIQUES
}

#[inline]
pub(super) fn should_use_partitioned_std_var_engine(keys: &[i64]) -> bool {
    should_use_partitioned_firstseen_engine(keys)
}

#[inline]
pub(super) fn should_use_partitioned_median_engine(keys: &[i64]) -> bool {
    should_use_partitioned_firstseen_engine(keys)
}

#[cfg(test)]
pub(super) fn should_use_direct_sorted_median_engine(keys: &[i64]) -> bool {
    sorted_median_route_decision(keys, SortedMedianValueKind::F64).is_direct()
}

pub(super) fn sorted_median_route_decision(
    keys: &[i64],
    value_kind: SortedMedianValueKind,
) -> SortedMedianRouteDecision {
    let n_rows = keys.len();
    if n_rows == 0 {
        return SortedMedianRouteDecision::new(
            SortedMedianRouteKind::DirectDense,
            SortedMedianRouteReason::EmptyInput,
        );
    }

    let accepted_values = n_rows;
    let input_bytes = match checked_mul(n_rows, size_of::<i64>() + value_kind.element_size()) {
        Some(bytes) => bytes,
        None => {
            return SortedMedianRouteDecision::new(
                SortedMedianRouteKind::ExistingSortedFallback,
                SortedMedianRouteReason::CheckedArithmeticOverflow,
            )
        }
    };

    let dense_span = match exact_dense_span(keys) {
        Ok(span) => span,
        Err(reason) => {
            return SortedMedianRouteDecision::new(
                SortedMedianRouteKind::ExistingSortedFallback,
                reason,
            )
        }
    };

    let dense_bytes = estimate_dense_direct_peak_bytes(
        dense_span.len,
        dense_span.len.min(n_rows),
        accepted_values,
        value_kind,
    );
    let dense_work = estimate_dense_work(n_rows, accepted_values, dense_span.len, n_rows);
    let estimated_groups_for_dense = dense_span.len.min(n_rows);
    let existing_for_dense =
        estimate_existing_work(n_rows, accepted_values, estimated_groups_for_dense);
    let partitioned_for_dense =
        estimate_partitioned_work(n_rows, accepted_values, estimated_groups_for_dense);

    if memory_within_caps(dense_bytes, input_bytes)
        && dense_work.total_score * (1.0 + SORTED_MEDIAN_DIRECT_SAFETY_MARGIN)
            < existing_for_dense
                .total_score
                .min(partitioned_for_dense.total_score)
    {
        let mut decision = SortedMedianRouteDecision::new(
            SortedMedianRouteKind::DirectDense,
            SortedMedianRouteReason::DenseSpanMemorySafe,
        );
        decision.dense_span = Some(dense_span.len);
        decision.dense_min_key = Some(dense_span.min_key);
        return decision;
    }

    let sample_uniques = estimate_sample_unique_keys(keys);
    if sample_uniques >= PARTITIONED_ENGINE_MIN_SAMPLE_UNIQUES {
        let mut decision = SortedMedianRouteDecision::new(
            SortedMedianRouteKind::PartitionedFallback,
            SortedMedianRouteReason::HighCardinalityPartitionedSample,
        );
        decision.dense_span = Some(dense_span.len);
        decision.dense_min_key = Some(dense_span.min_key);
        return decision;
    }

    let exact_stats = match exact_key_stats(keys) {
        Some(stats) => stats,
        None => {
            return SortedMedianRouteDecision::new(
                SortedMedianRouteKind::ExistingSortedFallback,
                SortedMedianRouteReason::CheckedArithmeticOverflow,
            )
        }
    };

    let sparse_bytes =
        estimate_sparse_direct_peak_bytes(n_rows, exact_stats.groups, accepted_values, value_kind);
    let sparse_work = estimate_sparse_work(
        n_rows,
        accepted_values,
        exact_stats.groups,
        exact_stats.max_group_rows,
    );
    let existing_work = estimate_existing_work(n_rows, accepted_values, exact_stats.groups);
    let partitioned_work = estimate_partitioned_work(n_rows, accepted_values, exact_stats.groups);

    let mut decision = SortedMedianRouteDecision::new(
        SortedMedianRouteKind::DirectSparse,
        SortedMedianRouteReason::SparseExactLowCardinalityMemorySafe,
    );
    decision.dense_span = Some(dense_span.len);
    decision.dense_min_key = Some(dense_span.min_key);

    if exact_stats.groups > SORTED_MEDIAN_SPARSE_MAX_EXACT_GROUPS {
        decision.kind = SortedMedianRouteKind::ExistingSortedFallback;
        decision.reason = SortedMedianRouteReason::SparseExactGroupsExceedCap;
    } else if !memory_within_caps(sparse_bytes, input_bytes) {
        decision.kind = SortedMedianRouteKind::ExistingSortedFallback;
        decision.reason = SortedMedianRouteReason::SparseMemoryCapExceeded;
    } else if sparse_work.total_score * (1.0 + SORTED_MEDIAN_DIRECT_SAFETY_MARGIN)
        >= existing_work.total_score.min(partitioned_work.total_score)
    {
        decision.kind = SortedMedianRouteKind::ExistingSortedFallback;
        decision.reason = SortedMedianRouteReason::FallbackSafetyMarginNotMet;
    }

    decision
}

fn exact_dense_span(keys: &[i64]) -> Result<DenseSpan, SortedMedianRouteReason> {
    let (&first, rest) = keys
        .split_first()
        .ok_or(SortedMedianRouteReason::EmptyInput)?;
    let mut min_key = first;
    let mut max_key = first;

    for key in rest.iter().copied() {
        min_key = min_key.min(key);
        max_key = max_key.max(key);
    }

    let span = i128::from(max_key) - i128::from(min_key) + 1;
    if span <= 0 || span > usize::MAX as i128 {
        return Err(SortedMedianRouteReason::DenseSpanOverflow);
    }

    Ok(DenseSpan {
        min_key,
        len: span as usize,
    })
}

#[cfg(test)]
pub(super) fn dense_sorted_median_key_range_len(keys: &[i64]) -> Option<usize> {
    let decision = sorted_median_route_decision(keys, SortedMedianValueKind::F64);
    if matches!(decision.kind, SortedMedianRouteKind::DirectDense) {
        decision.dense_span
    } else {
        None
    }
}

fn exact_key_stats(keys: &[i64]) -> Option<ExactKeyStats> {
    let mut counts: AHashMap<i64, usize> = AHashMap::new();
    let mut max_group_rows = 0usize;

    for key in keys.iter().copied() {
        let count = counts.entry(key).or_insert(0);
        *count = count.checked_add(1)?;
        max_group_rows = max_group_rows.max(*count);
    }

    Some(ExactKeyStats {
        groups: counts.len(),
        max_group_rows,
    })
}

fn checked_add(a: usize, b: usize) -> Option<usize> {
    a.checked_add(b)
}

fn checked_mul(a: usize, b: usize) -> Option<usize> {
    a.checked_mul(b)
}

fn add_term(total: &mut Option<usize>, count: usize, size: usize) {
    *total = total
        .and_then(|current| checked_mul(count, size).and_then(|term| checked_add(current, term)));
}

fn estimate_dense_direct_peak_bytes(
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

fn estimate_sparse_direct_peak_bytes(
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

fn memory_within_caps(estimated_bytes: Option<usize>, input_bytes: usize) -> bool {
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

fn estimate_dense_work(
    n_rows: usize,
    accepted_values: usize,
    span: usize,
    max_group_rows: usize,
) -> RouteWorkEstimate {
    let row_pass_work = 1.25 * n_rows as f64;
    let span_work = span as f64;
    let scatter_work = accepted_values as f64;
    let median_select_work = accepted_values as f64;
    let parallel_span_penalty = max_group_rows as f64;
    let total_score =
        row_pass_work + span_work + scatter_work + median_select_work + parallel_span_penalty;
    RouteWorkEstimate { total_score }
}

fn estimate_sparse_work(
    n_rows: usize,
    accepted_values: usize,
    groups: usize,
    max_group_rows: usize,
) -> RouteWorkEstimate {
    let row_pass_work = n_rows as f64;
    let hash_work = 1.25 * n_rows as f64;
    let unique_sort_work = groups as f64 * (groups.max(2) as f64).log2();
    let scatter_work = accepted_values as f64;
    let median_select_work = accepted_values as f64;
    let parallel_span_penalty = max_group_rows as f64;
    let total_score = row_pass_work
        + hash_work
        + unique_sort_work
        + scatter_work
        + median_select_work
        + parallel_span_penalty;
    RouteWorkEstimate { total_score }
}

fn estimate_existing_work(
    n_rows: usize,
    accepted_values: usize,
    groups: usize,
) -> RouteWorkEstimate {
    let row_pass_work = 5.0 * n_rows as f64;
    let reorder_or_output_sort_work = groups as f64 * (groups.max(2) as f64).log2();
    let median_select_work = accepted_values as f64;
    let total_score = row_pass_work + reorder_or_output_sort_work + median_select_work;
    RouteWorkEstimate { total_score }
}

fn estimate_partitioned_work(
    n_rows: usize,
    accepted_values: usize,
    groups: usize,
) -> RouteWorkEstimate {
    let row_pass_work = 4.0 * n_rows as f64;
    let reorder_or_output_sort_work = groups as f64 * (groups.max(2) as f64).log2();
    let median_select_work = accepted_values as f64;
    let total_score = row_pass_work + reorder_or_output_sort_work + median_select_work;
    RouteWorkEstimate { total_score }
}

#[inline]
pub(super) fn should_use_partitioned_prod_engine(keys: &[i64]) -> bool {
    should_use_partitioned_firstseen_engine(keys)
}
