use super::sort_first_routing::{
    choose_sort_first_route, SortFirstFallbackReason, SortFirstReducer, SortFirstRoute,
    MAX_KEY_COLUMNS, MIN_SAMPLE_ROWS, SAMPLE_SIZE, UNIQUE_RATIO_DENOMINATOR,
    UNIQUE_RATIO_NUMERATOR,
};

fn key_slices<'a>(left: &'a [i64], right: &'a [i64]) -> Vec<&'a [i64]> {
    vec![left, right]
}

fn assert_hash_first(
    reducer: SortFirstReducer,
    key_slices: &[&[i64]],
    n_rows: usize,
    reason: SortFirstFallbackReason,
) {
    let decision = choose_sort_first_route(reducer, key_slices, n_rows);
    assert_eq!(decision.route, SortFirstRoute::HashFirst);
    assert_eq!(decision.fallback_reason, Some(reason));
}

fn high_unique_keys(n_rows: i64) -> (Vec<i64>, Vec<i64>) {
    let left: Vec<i64> = (0..n_rows).collect();
    let right: Vec<i64> = (0..n_rows).map(|row| n_rows - row).collect();
    (left, right)
}

#[test]
fn routing_constants_match_conservative_plan_contract() {
    assert_eq!(SAMPLE_SIZE, 16_384);
    assert_eq!(MIN_SAMPLE_ROWS, 4_096);
    assert_eq!(UNIQUE_RATIO_NUMERATOR, 3);
    assert_eq!(UNIQUE_RATIO_DENOMINATOR, 4);
}

#[test]
fn routing_low_tuple_ratio_stays_hash_first() {
    // Given enough rows but repeated tuple combinations in the sampled prefix.
    let n_rows = 5_000i64;
    let left: Vec<i64> = (0..n_rows).map(|row| row % 1_000).collect();
    let right: Vec<i64> = (0..n_rows).map(|row| (row * 7) % 1_000).collect();
    let keys = key_slices(&left, &right);

    // When the route predicate estimates tuple uniqueness.
    let decision = choose_sort_first_route(SortFirstReducer::MaxF64, &keys, left.len());

    // Then low sampled tuple ratio falls back to hash-first.
    assert_eq!(decision.route, SortFirstRoute::HashFirst);
    assert_eq!(
        decision.fallback_reason,
        Some(SortFirstFallbackReason::LowTupleRatio)
    );
    assert_eq!(decision.sample_rows, left.len());
    assert_eq!(decision.sample_unique_tuples, 1_000);
}

#[test]
fn routing_medium_tuple_ratio_below_threshold_stays_hash_first() {
    // Given enough rows with sampled tuple uniqueness just below the 3/4 threshold.
    let n_rows = 5_000i64;
    let unique_tuples = 3_700i64;
    let left: Vec<i64> = (0..n_rows).map(|row| row % unique_tuples).collect();
    let right: Vec<i64> = (0..n_rows).map(|row| row % unique_tuples).collect();
    let keys = key_slices(&left, &right);

    // When the route predicate estimates tuple uniqueness.
    let decision = choose_sort_first_route(SortFirstReducer::MaxF64, &keys, left.len());

    // Then a medium ratio below the route cutoff still falls back to hash-first.
    assert!(
        (unique_tuples as usize).saturating_mul(UNIQUE_RATIO_DENOMINATOR)
            < left.len().saturating_mul(UNIQUE_RATIO_NUMERATOR)
    );
    assert_eq!(decision.route, SortFirstRoute::HashFirst);
    assert_eq!(
        decision.fallback_reason,
        Some(SortFirstFallbackReason::LowTupleRatio)
    );
    assert_eq!(decision.sample_rows, left.len());
    assert_eq!(decision.sample_unique_tuples, unique_tuples as usize);
}

#[test]
fn routing_high_tuple_ratio_uses_hash_first_until_sort_first_is_certified() {
    // Given high-cardinality-like tuple samples.
    let (left, right) = high_unique_keys(5_000);
    let keys = key_slices(&left, &right);

    // When each production reducer sees high sampled tuple uniqueness.
    for reducer in [
        SortFirstReducer::MaxF64,
        SortFirstReducer::MaxI64,
        SortFirstReducer::CountF64,
        SortFirstReducer::CountI64,
    ] {
        let decision = choose_sort_first_route(reducer, &keys, left.len());

        // Then unique ratio alone is not enough to certify the current
        // sequential comparator sort-first implementation for production.
        assert_eq!(decision.route, SortFirstRoute::HashFirst);
        assert_eq!(
            decision.fallback_reason,
            Some(SortFirstFallbackReason::SortFirstNotCertified)
        );
        assert_eq!(decision.sample_rows, left.len());
        assert_eq!(decision.sample_unique_tuples, left.len());
    }
}

#[test]
fn routing_unsupported_reducer_stays_hash_first_even_for_high_ratio() {
    // Given high tuple uniqueness but an unsupported reducer.
    let (left, right) = high_unique_keys(5_000);
    let keys = key_slices(&left, &right);

    // When routing sees the unsupported reducer.
    assert_hash_first(
        SortFirstReducer::Unsupported,
        &keys,
        left.len(),
        SortFirstFallbackReason::UnsupportedReducer,
    );
}

#[test]
fn routing_too_few_rows_stays_hash_first_even_for_high_ratio() {
    // Given unique tuples below the minimum sample confidence size.
    let below_minimum = 4_095i64;
    let (left, right) = high_unique_keys(below_minimum);
    let keys = key_slices(&left, &right);

    // When routing samples too few rows.
    let decision = choose_sort_first_route(SortFirstReducer::MaxF64, &keys, left.len());

    // Then uncertainty falls back to hash-first.
    assert_eq!(decision.route, SortFirstRoute::HashFirst);
    assert_eq!(
        decision.fallback_reason,
        Some(SortFirstFallbackReason::TooFewRows)
    );
    assert_eq!(decision.sample_rows, left.len());
}

#[test]
fn routing_one_hundred_percent_unique_sample_still_requires_certification() {
    // Given 100% unique sampled tuples in reverse lexicographic order.
    let n_rows = 5_000i64;
    let left: Vec<i64> = (0..n_rows).rev().collect();
    let right: Vec<i64> = (0..n_rows).map(|row| -row).collect();
    let keys = key_slices(&left, &right);

    // When routing estimates tuple uniqueness.
    let decision = choose_sort_first_route(SortFirstReducer::MaxF64, &keys, left.len());

    // Then even a perfect unique sample stays hash-first until a certified
    // physical sort-first algorithm replaces the regressed sequential path.
    assert_eq!(decision.route, SortFirstRoute::HashFirst);
    assert_eq!(
        decision.fallback_reason,
        Some(SortFirstFallbackReason::SortFirstNotCertified)
    );
    assert_eq!(decision.sample_rows, left.len());
    assert_eq!(decision.sample_unique_tuples, left.len());
}

#[test]
fn routing_duplicate_heavy_input_stays_hash_first() {
    // Given many rows but only a small repeated tuple set.
    let n_rows = 5_000i64;
    let left: Vec<i64> = (0..n_rows).map(|row| row % 64).collect();
    let right: Vec<i64> = (0..n_rows).map(|row| row % 8).collect();
    let keys = key_slices(&left, &right);

    // When routing samples the duplicate-heavy prefix.
    let decision = choose_sort_first_route(SortFirstReducer::CountI64, &keys, left.len());

    // Then low confidence falls back to hash-first.
    assert_eq!(decision.route, SortFirstRoute::HashFirst);
    assert_eq!(
        decision.fallback_reason,
        Some(SortFirstFallbackReason::LowTupleRatio)
    );
    assert_eq!(decision.sample_unique_tuples, 64);
}

#[test]
fn routing_unsupported_key_count_or_invalid_lengths_stay_hash_first() {
    // Given a single-key sorted shape and a mismatched multi-key shape.
    let (left, right) = high_unique_keys(5_000);
    let single_key: Vec<&[i64]> = vec![&left];
    let invalid_multi_key: Vec<&[i64]> = vec![&left, &right[..4_999]];

    // When routing checks key count and shape.
    assert_hash_first(
        SortFirstReducer::MaxF64,
        &single_key,
        left.len(),
        SortFirstFallbackReason::UnsupportedKeyCount,
    );
    assert_hash_first(
        SortFirstReducer::MaxF64,
        &invalid_multi_key,
        left.len(),
        SortFirstFallbackReason::InvalidInput,
    );
}

#[test]
fn routing_more_than_max_key_columns_stays_hash_first_before_sampling() {
    // Given high-cardinality-like tuples across one more key than the project supports.
    let n_rows = 5_000usize;
    let keys_storage: Vec<Vec<i64>> = (0..=MAX_KEY_COLUMNS)
        .map(|col| {
            (0..n_rows as i64)
                .map(|row| row.wrapping_mul(17) + col as i64)
                .collect()
        })
        .collect();
    let keys: Vec<&[i64]> = keys_storage.iter().map(Vec::as_slice).collect();

    // When routing sees the over-cap key count.
    let decision = choose_sort_first_route(SortFirstReducer::MaxF64, &keys, n_rows);

    // Then it falls back before sampling can select sort-first.
    assert_eq!(decision.route, SortFirstRoute::HashFirst);
    assert_eq!(
        decision.fallback_reason,
        Some(SortFirstFallbackReason::UnsupportedKeyCount)
    );
    assert_eq!(decision.sample_rows, 0);
    assert_eq!(decision.sample_unique_tuples, 0);
}
