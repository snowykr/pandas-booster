use ahash::AHashSet;

#[cfg(test)]
pub(super) use super::routing_median::dense_sorted_median_key_range_len;
pub(super) use super::routing_median::sorted_median_route_decision;

const PARTITIONED_ENGINE_SAMPLE_SIZE: usize = 16_384;
const PARTITIONED_ENGINE_MIN_SAMPLE_UNIQUES: usize = 4_096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SortedMedianValueKind {
    F64,
    I64,
}

impl SortedMedianValueKind {
    pub(in crate::groupby) fn element_size(self) -> usize {
        match self {
            Self::F64 => std::mem::size_of::<f64>(),
            Self::I64 => std::mem::size_of::<i64>(),
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

#[derive(Debug, Clone)]
pub(super) struct SortedMedianRouteDecision {
    pub(super) kind: SortedMedianRouteKind,
    pub(super) reason: SortedMedianRouteReason,
    pub(super) dense_span: Option<usize>,
    pub(super) dense_min_key: Option<i64>,
}

impl SortedMedianRouteDecision {
    pub(in crate::groupby) fn new(
        kind: SortedMedianRouteKind,
        reason: SortedMedianRouteReason,
    ) -> Self {
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

#[inline]
pub(super) fn should_use_partitioned_prod_engine(keys: &[i64]) -> bool {
    should_use_partitioned_firstseen_engine(keys)
}
