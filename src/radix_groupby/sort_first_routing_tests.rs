use super::sort_first_routing::{
    choose_sort_first_route, SortFirstFallbackReason, SortFirstReducer, SortFirstRoute,
    MAX_KEY_COLUMNS, MIN_SORT_FIRST_ROWS,
};

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

#[test]
fn routing_stays_disabled_regardless_of_tuple_cardinality() {
    for n_rows in [MIN_SORT_FIRST_ROWS, 5_000, 20_000] {
        let unique: Vec<i64> = (0..n_rows as i64).rev().collect();
        let repeated = vec![-7; n_rows];
        for reducer in [
            SortFirstReducer::MaxF64,
            SortFirstReducer::MaxI64,
            SortFirstReducer::CountF64,
            SortFirstReducer::CountI64,
        ] {
            for keys in [
                [unique.as_slice(), unique.as_slice()],
                [repeated.as_slice(), repeated.as_slice()],
                [unique.as_slice(), repeated.as_slice()],
            ] {
                assert_hash_first(
                    reducer,
                    &keys,
                    n_rows,
                    SortFirstFallbackReason::SortFirstNotCertified,
                );
            }
        }
    }
}

#[test]
fn routing_unsupported_reducers_remain_diagnostic() {
    let column = vec![0; MIN_SORT_FIRST_ROWS];
    for reducer in [
        SortFirstReducer::Unsupported,
        SortFirstReducer::SumF64,
        SortFirstReducer::MinF64,
    ] {
        assert_hash_first(
            reducer,
            &[&column, &column],
            column.len(),
            SortFirstFallbackReason::UnsupportedReducer,
        );
    }
}

#[test]
fn routing_small_inputs_remain_diagnostic() {
    for n_rows in [0, 1, MIN_SORT_FIRST_ROWS - 1] {
        let column = vec![0; n_rows];
        assert_hash_first(
            SortFirstReducer::MaxF64,
            &[&column, &column],
            n_rows,
            SortFirstFallbackReason::TooFewRows,
        );
    }
}

#[test]
fn routing_key_count_limits_remain_diagnostic() {
    let column = vec![0; MIN_SORT_FIRST_ROWS];
    for n_keys in [0, 1, MAX_KEY_COLUMNS + 1] {
        let keys = vec![column.as_slice(); n_keys];
        assert_hash_first(
            SortFirstReducer::MaxF64,
            &keys,
            column.len(),
            SortFirstFallbackReason::UnsupportedKeyCount,
        );
    }
    let keys = vec![column.as_slice(); MAX_KEY_COLUMNS];
    assert_hash_first(
        SortFirstReducer::MaxF64,
        &keys,
        column.len(),
        SortFirstFallbackReason::SortFirstNotCertified,
    );
}

#[test]
fn routing_invalid_lengths_are_reported_before_size_or_disabled_route() {
    let column = vec![0; MIN_SORT_FIRST_ROWS];
    for n_rows in [0, MIN_SORT_FIRST_ROWS - 1, MIN_SORT_FIRST_ROWS] {
        assert_hash_first(
            SortFirstReducer::MaxF64,
            &[&column, &column[..MIN_SORT_FIRST_ROWS - 1]],
            n_rows,
            SortFirstFallbackReason::InvalidInput,
        );
    }
}
