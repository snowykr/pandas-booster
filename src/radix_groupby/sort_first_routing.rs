use ahash::AHashSet;
use smallvec::SmallVec;

pub(super) const MAX_KEY_COLUMNS: usize = 10;
pub(super) const SAMPLE_SIZE: usize = 16_384;
pub(super) const MIN_SAMPLE_ROWS: usize = 4_096;
pub(super) const UNIQUE_RATIO_NUMERATOR: usize = 3;
pub(super) const UNIQUE_RATIO_DENOMINATOR: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SortFirstReducer {
    #[cfg(test)]
    SumF64,
    MaxF64,
    MaxI64,
    #[cfg(test)]
    MinF64,
    CountF64,
    CountI64,
    #[cfg(test)]
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SortFirstRoute {
    HashFirst,
    SortFirst,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SortFirstFallbackReason {
    UnsupportedReducer,
    UnsupportedKeyCount,
    InvalidInput,
    TooFewRows,
    LowTupleRatio,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SortFirstRoutingDecision {
    pub route: SortFirstRoute,
    pub sample_rows: usize,
    pub sample_unique_tuples: usize,
    pub fallback_reason: Option<SortFirstFallbackReason>,
}

pub(super) fn choose_sort_first_route(
    reducer: SortFirstReducer,
    key_slices: &[&[i64]],
    n_rows: usize,
) -> SortFirstRoutingDecision {
    if !is_supported_reducer(reducer) {
        return hash_first(SortFirstFallbackReason::UnsupportedReducer, 0, 0);
    }

    if key_slices.len() < 2 || key_slices.len() > MAX_KEY_COLUMNS {
        return hash_first(SortFirstFallbackReason::UnsupportedKeyCount, 0, 0);
    }

    if key_slices.iter().any(|col| col.len() != n_rows) {
        return hash_first(SortFirstFallbackReason::InvalidInput, 0, 0);
    }

    let sample_rows = n_rows.min(SAMPLE_SIZE);
    if sample_rows < MIN_SAMPLE_ROWS {
        return hash_first(SortFirstFallbackReason::TooFewRows, sample_rows, 0);
    }

    let sample_unique_tuples = sampled_unique_tuples(key_slices, sample_rows);
    if sample_unique_tuples.saturating_mul(UNIQUE_RATIO_DENOMINATOR)
        < sample_rows.saturating_mul(UNIQUE_RATIO_NUMERATOR)
    {
        return hash_first(
            SortFirstFallbackReason::LowTupleRatio,
            sample_rows,
            sample_unique_tuples,
        );
    }

    SortFirstRoutingDecision {
        route: SortFirstRoute::SortFirst,
        sample_rows,
        sample_unique_tuples,
        fallback_reason: None,
    }
}

const fn is_supported_reducer(reducer: SortFirstReducer) -> bool {
    matches!(
        reducer,
        SortFirstReducer::MaxF64
            | SortFirstReducer::MaxI64
            | SortFirstReducer::CountF64
            | SortFirstReducer::CountI64
    )
}

const fn hash_first(
    reason: SortFirstFallbackReason,
    sample_rows: usize,
    sample_unique_tuples: usize,
) -> SortFirstRoutingDecision {
    SortFirstRoutingDecision {
        route: SortFirstRoute::HashFirst,
        sample_rows,
        sample_unique_tuples,
        fallback_reason: Some(reason),
    }
}

fn sampled_unique_tuples(key_slices: &[&[i64]], sample_rows: usize) -> usize {
    let mut seen = AHashSet::with_capacity(sample_rows);

    for row in 0..sample_rows {
        let mut tuple: SmallVec<[i64; MAX_KEY_COLUMNS]> = SmallVec::with_capacity(key_slices.len());
        for col in key_slices {
            tuple.push(col[row]);
        }
        seen.insert(tuple);
    }

    seen.len()
}
