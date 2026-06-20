use super::sort_first_routing::{
    choose_sort_first_route, SortFirstFallbackReason, SortFirstReducer, SortFirstRoute,
    MIN_SAMPLE_ROWS, SAMPLE_SIZE, UNIQUE_RATIO_DENOMINATOR, UNIQUE_RATIO_NUMERATOR,
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
    let decision = choose_sort_first_route(SortFirstReducer::SumF64, &keys, left.len());

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
fn routing_high_tuple_ratio_selects_sort_first_for_supported_reducers() {
    // Given high-cardinality-like tuple samples.
    let (left, right) = high_unique_keys(5_000);
    let keys = key_slices(&left, &right);

    // When each initial T7 reducer is checked.
    for reducer in [
        SortFirstReducer::SumF64,
        SortFirstReducer::MaxF64,
        SortFirstReducer::MinF64,
        SortFirstReducer::CountI64,
    ] {
        let decision = choose_sort_first_route(reducer, &keys, left.len());

        // Then every supported reducer selects sort-first confidently.
        assert_eq!(decision.route, SortFirstRoute::SortFirst);
        assert_eq!(decision.fallback_reason, None);
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
fn routing_reverse_sorted_high_ratio_still_selects_sort_first() {
    // Given high-cardinality-like tuples in reverse lexicographic order.
    let n_rows = 5_000i64;
    let left: Vec<i64> = (0..n_rows).rev().collect();
    let right: Vec<i64> = (0..n_rows).map(|row| -row).collect();
    let keys = key_slices(&left, &right);

    // When routing estimates tuple uniqueness.
    let decision = choose_sort_first_route(SortFirstReducer::MinF64, &keys, left.len());

    // Then input order does not prevent confident sort-first selection.
    assert_eq!(decision.route, SortFirstRoute::SortFirst);
    assert_eq!(decision.fallback_reason, None);
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
        SortFirstReducer::SumF64,
        &single_key,
        left.len(),
        SortFirstFallbackReason::UnsupportedKeyCount,
    );
    assert_hash_first(
        SortFirstReducer::SumF64,
        &invalid_multi_key,
        left.len(),
        SortFirstFallbackReason::InvalidInput,
    );
}
