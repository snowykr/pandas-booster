pub(super) const MAX_KEY_COLUMNS: usize = 10;
pub(super) const MIN_SORT_FIRST_ROWS: usize = 4_096;

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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SortFirstFallbackReason {
    UnsupportedReducer,
    UnsupportedKeyCount,
    InvalidInput,
    TooFewRows,
    SortFirstNotCertified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SortFirstRoutingDecision {
    pub route: SortFirstRoute,
    pub fallback_reason: Option<SortFirstFallbackReason>,
}

pub(super) fn choose_sort_first_route(
    reducer: SortFirstReducer,
    key_slices: &[&[i64]],
    n_rows: usize,
) -> SortFirstRoutingDecision {
    if !is_supported_reducer(reducer) {
        return hash_first(SortFirstFallbackReason::UnsupportedReducer);
    }

    if key_slices.len() < 2 || key_slices.len() > MAX_KEY_COLUMNS {
        return hash_first(SortFirstFallbackReason::UnsupportedKeyCount);
    }

    if key_slices.iter().any(|col| col.len() != n_rows) {
        return hash_first(SortFirstFallbackReason::InvalidInput);
    }

    if n_rows < MIN_SORT_FIRST_ROWS {
        return hash_first(SortFirstFallbackReason::TooFewRows);
    }

    // No tuple sampling is useful while the production sort-first route is disabled.
    hash_first(SortFirstFallbackReason::SortFirstNotCertified)
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

const fn hash_first(reason: SortFirstFallbackReason) -> SortFirstRoutingDecision {
    SortFirstRoutingDecision {
        route: SortFirstRoute::HashFirst,
        fallback_reason: Some(reason),
    }
}
