use ahash::AHashMap;

use super::routing::{
    should_use_partitioned_median_engine, SortedMedianRouteDecision, SortedMedianRouteKind,
    SortedMedianRouteReason, SortedMedianValueKind,
};
use super::routing_median_estimates::{
    checked_input_bytes, estimate_dense_direct_peak_bytes, estimate_dense_work,
    estimate_existing_work, estimate_partitioned_work, estimate_sparse_direct_peak_bytes,
    estimate_sparse_work, memory_within_caps, SORTED_MEDIAN_DIRECT_SAFETY_MARGIN,
    SORTED_MEDIAN_SPARSE_MAX_EXACT_GROUPS,
};

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

pub(in crate::groupby) fn sorted_median_route_decision(
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
    let input_bytes = match checked_input_bytes(n_rows, value_kind) {
        Some(bytes) => bytes,
        None => return overflow_fallback(),
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
    let groups_for_dense = dense_span.len.min(n_rows);
    let existing_for_dense = estimate_existing_work(n_rows, accepted_values, groups_for_dense);
    let partitioned_for_dense =
        estimate_partitioned_work(n_rows, accepted_values, groups_for_dense);

    if memory_within_caps(dense_bytes, input_bytes)
        && dense_work.total_score * (1.0 + SORTED_MEDIAN_DIRECT_SAFETY_MARGIN)
            < existing_for_dense
                .total_score
                .min(partitioned_for_dense.total_score)
    {
        return decision_with_span(
            SortedMedianRouteKind::DirectDense,
            SortedMedianRouteReason::DenseSpanMemorySafe,
            dense_span,
        );
    }

    if should_use_partitioned_median_engine(keys) {
        return decision_with_span(
            SortedMedianRouteKind::PartitionedFallback,
            SortedMedianRouteReason::HighCardinalityPartitionedSample,
            dense_span,
        );
    }

    decide_sparse_route(keys, value_kind, input_bytes, accepted_values, dense_span)
}

fn decide_sparse_route(
    keys: &[i64],
    value_kind: SortedMedianValueKind,
    input_bytes: usize,
    accepted_values: usize,
    dense_span: DenseSpan,
) -> SortedMedianRouteDecision {
    let exact_stats = match exact_key_stats(keys) {
        Some(stats) => stats,
        None => return overflow_fallback(),
    };
    let sparse_bytes = estimate_sparse_direct_peak_bytes(
        keys.len(),
        exact_stats.groups,
        accepted_values,
        value_kind,
    );
    let sparse_work = estimate_sparse_work(
        keys.len(),
        accepted_values,
        exact_stats.groups,
        exact_stats.max_group_rows,
    );
    let existing_work = estimate_existing_work(keys.len(), accepted_values, exact_stats.groups);
    let partitioned_work =
        estimate_partitioned_work(keys.len(), accepted_values, exact_stats.groups);

    let mut decision = decision_with_span(
        SortedMedianRouteKind::DirectSparse,
        SortedMedianRouteReason::SparseExactLowCardinalityMemorySafe,
        dense_span,
    );
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

fn decision_with_span(
    kind: SortedMedianRouteKind,
    reason: SortedMedianRouteReason,
    dense_span: DenseSpan,
) -> SortedMedianRouteDecision {
    let mut decision = SortedMedianRouteDecision::new(kind, reason);
    decision.dense_span = Some(dense_span.len);
    decision.dense_min_key = Some(dense_span.min_key);
    decision
}

fn overflow_fallback() -> SortedMedianRouteDecision {
    SortedMedianRouteDecision::new(
        SortedMedianRouteKind::ExistingSortedFallback,
        SortedMedianRouteReason::CheckedArithmeticOverflow,
    )
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
pub(in crate::groupby) fn dense_sorted_median_key_range_len(keys: &[i64]) -> Option<usize> {
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
