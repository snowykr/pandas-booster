use super::dispatch::SortedDispatchRoute;
use super::sort_first_routing::{
    SortFirstFallbackReason, SortFirstReducer, UNIQUE_RATIO_DENOMINATOR, UNIQUE_RATIO_NUMERATOR,
};
use super::test_support::{key_at_out, value_at_out};
use super::*;

fn high_ratio_keys(n_rows: usize) -> (Vec<i64>, Vec<i64>) {
    let left: Vec<i64> = (0..n_rows as i64).rev().collect();
    let right: Vec<i64> = (0..n_rows as i64).collect();
    (left, right)
}

fn low_ratio_keys(n_rows: usize) -> (Vec<i64>, Vec<i64>) {
    let left: Vec<i64> = (0..n_rows as i64).map(|row| row % 64).collect();
    let right: Vec<i64> = (0..n_rows as i64).map(|row| (row * 3) % 64).collect();
    (left, right)
}

fn key_slices<'a>(left: &'a [i64], right: &'a [i64]) -> Vec<&'a [i64]> {
    vec![left, right]
}

fn assert_sort_first_not_reached_for_target_route(
    diagnostics: &super::dispatch::SortedDispatchDiagnostics,
) {
    assert_eq!(diagnostics.route, SortedDispatchRoute::HashFirst);
    assert_eq!(diagnostics.hash_first_aggregation_count, 1);
    assert_eq!(diagnostics.post_aggregation_sort_count, 1);
    assert_eq!(
        diagnostics.routing_decision.fallback_reason,
        Some(SortFirstFallbackReason::SortFirstNotCertified)
    );
    assert!(diagnostics.sort_first.is_none());
    assert_eq!(diagnostics.routing_decision.sample_rows, 5_000);
    assert_eq!(diagnostics.routing_decision.sample_unique_tuples, 5_000);
}

fn assert_hash_first_route(
    diagnostics: &super::dispatch::SortedDispatchDiagnostics,
    reason: SortFirstFallbackReason,
) {
    assert_eq!(diagnostics.route, SortedDispatchRoute::HashFirst);
    assert_eq!(diagnostics.hash_first_aggregation_count, 1);
    assert_eq!(diagnostics.post_aggregation_sort_count, 1);
    assert_eq!(diagnostics.routing_decision.fallback_reason, Some(reason));
    assert!(diagnostics.sort_first.is_none());
}

#[test]
fn production_routing_high_ratio_max_uses_hash_first_until_sort_first_is_certified() {
    // Given high tuple uniqueness in reverse lexicographic order.
    let (left, right) = high_ratio_keys(5_000);
    let values_f64: Vec<f64> = left.iter().map(|&key| key as f64).collect();
    let values_i64: Vec<i64> = left.clone();
    let keys = key_slices(&left, &right);

    // When production sorted max dispatch runs.
    let (float_result, float_diagnostics) =
        super::api_sorted::radix_groupby_max_f64_sorted_with_diagnostics(&keys, &values_f64)
            .expect("high-ratio f64 max should dispatch");
    let (integer_result, integer_diagnostics) =
        super::api_sorted::radix_groupby_max_i64_sorted_with_diagnostics(&keys, &values_i64)
            .expect("high-ratio i64 max should dispatch");

    // Then both value dtypes use certified hash-first plus post-aggregation sort.
    assert_sort_first_not_reached_for_target_route(&float_diagnostics);
    assert_sort_first_not_reached_for_target_route(&integer_diagnostics);
    assert_eq!(key_at_out(&float_result, 0, 0), 0);
    assert_eq!(key_at_out(&integer_result, 0, 0), 0);
    assert_eq!(value_at_out(&float_result, 0), 0.0);
    assert_eq!(value_at_out(&integer_result, 0), 0);
}

#[test]
fn production_routing_high_ratio_count_uses_hash_first_until_sort_first_is_certified() {
    // Given high tuple uniqueness with values that exercise f64 NaN count semantics.
    let (left, right) = high_ratio_keys(5_000);
    let values_f64: Vec<f64> = (0..left.len())
        .map(|row| if row % 11 == 0 { f64::NAN } else { row as f64 })
        .collect();
    let values_i64: Vec<i64> = (0..left.len() as i64).collect();
    let keys = key_slices(&left, &right);

    // When production sorted count dispatch runs.
    let (float_result, float_diagnostics) =
        super::api_sorted::radix_groupby_count_f64_sorted_with_diagnostics(&keys, &values_f64)
            .expect("high-ratio f64 count should dispatch");
    let (integer_result, integer_diagnostics) =
        super::api_sorted::radix_groupby_count_i64_sorted_with_diagnostics(&keys, &values_i64)
            .expect("high-ratio i64 count should dispatch");

    // Then both value dtypes use certified hash-first plus post-aggregation sort.
    assert_sort_first_not_reached_for_target_route(&float_diagnostics);
    assert_sort_first_not_reached_for_target_route(&integer_diagnostics);
    assert_eq!(key_at_out(&float_result, 0, 0), 0);
    assert_eq!(key_at_out(&integer_result, 0, 0), 0);
    assert_eq!(value_at_out(&float_result, 0), 1);
    assert_eq!(value_at_out(&integer_result, 0), 1);
}

#[test]
fn production_routing_low_ratio_count_stays_hash_first() {
    // Given enough rows but duplicate-heavy sampled key tuples.
    let (left, right) = low_ratio_keys(5_000);
    let values: Vec<i64> = (0..left.len() as i64).collect();
    let keys = key_slices(&left, &right);

    // When production sorted count dispatch runs.
    let (_result, diagnostics) =
        super::api_sorted::radix_groupby_count_i64_sorted_with_diagnostics(&keys, &values)
            .expect("low-ratio count should fall back to hash-first dispatch");

    // Then it keeps the hash-first aggregation plus sorted-output path.
    assert_hash_first_route(&diagnostics, SortFirstFallbackReason::LowTupleRatio);
}

#[test]
fn production_routing_medium_ratio_below_threshold_stays_hash_first() {
    // Given sampled tuple uniqueness just below the production sort-first threshold.
    let n_rows = 5_000usize;
    let unique_tuples = 3_700i64;
    let left: Vec<i64> = (0..n_rows as i64).map(|row| row % unique_tuples).collect();
    let right: Vec<i64> = (0..n_rows as i64).map(|row| row % unique_tuples).collect();
    let values_f64: Vec<f64> = left.iter().map(|&key| key as f64).collect();
    let values_i64: Vec<i64> = (0..n_rows as i64).collect();
    let keys = key_slices(&left, &right);

    // When production sorted max and count dispatch run.
    let (_max_result, max_diagnostics) =
        super::api_sorted::radix_groupby_max_f64_sorted_with_diagnostics(&keys, &values_f64)
            .expect("medium-ratio f64 max should fall back to hash-first dispatch");
    let (_count_result, count_diagnostics) =
        super::api_sorted::radix_groupby_count_i64_sorted_with_diagnostics(&keys, &values_i64)
            .expect("medium-ratio i64 count should fall back to hash-first dispatch");

    // Then both supported reducers remain on the hash-first sorted-output path.
    assert!(
        (unique_tuples as usize).saturating_mul(UNIQUE_RATIO_DENOMINATOR)
            < n_rows.saturating_mul(UNIQUE_RATIO_NUMERATOR)
    );
    for diagnostics in [&max_diagnostics, &count_diagnostics] {
        assert_hash_first_route(diagnostics, SortFirstFallbackReason::LowTupleRatio);
        assert_eq!(diagnostics.routing_decision.sample_rows, n_rows);
        assert_eq!(
            diagnostics.routing_decision.sample_unique_tuples,
            unique_tuples as usize
        );
    }
}

#[test]
fn production_routing_sum_and_min_remain_hash_first_when_ratio_is_high() {
    // Given high tuple uniqueness.
    let (left, right) = high_ratio_keys(5_000);
    let keys = key_slices(&left, &right);

    // When routing considers reducers that are not production-routed in T8.
    for reducer in [SortFirstReducer::SumF64, SortFirstReducer::MinF64] {
        let decision =
            super::sort_first_routing::choose_sort_first_route(reducer, &keys, left.len());

        // Then they are explicitly rejected before production dispatch can use sort-first.
        assert_eq!(
            decision.route,
            super::sort_first_routing::SortFirstRoute::HashFirst
        );
        assert_eq!(
            decision.fallback_reason,
            Some(SortFirstFallbackReason::UnsupportedReducer)
        );
    }
}

#[test]
fn production_routing_more_than_ten_keys_stays_hash_first_before_sort_first() {
    // Given the Rust-side maximum-supported key count is exceeded.
    let n_rows = 5_000usize;
    let keys_storage: Vec<Vec<i64>> = (0..11)
        .map(|col| {
            (0..n_rows as i64)
                .map(|row| row.wrapping_mul(13) + col)
                .collect()
        })
        .collect();
    let keys: Vec<&[i64]> = keys_storage.iter().map(Vec::as_slice).collect();
    let values: Vec<i64> = (0..n_rows as i64).collect();

    // When production sorted count dispatch runs.
    let (_result, diagnostics) =
        super::api_sorted::radix_groupby_count_i64_sorted_with_diagnostics(&keys, &values)
            .expect("over-cap key counts should fall back instead of sort-first routing");

    // Then the cap is decided before the sort-first route can run.
    assert_hash_first_route(&diagnostics, SortFirstFallbackReason::UnsupportedKeyCount);
}

#[test]
fn production_routing_profile_high_ratio_max_labels_hash_first() {
    // Given a high tuple ratio max profile workload.
    let (left, right) = high_ratio_keys(5_000);
    let values: Vec<f64> = left.iter().map(|&key| key as f64).collect();
    let keys = key_slices(&left, &right);

    // When the existing profile export runs.
    let profiled = profile_radix_groupby_max_f64_sorted(&keys, &values)
        .expect("profiled high-ratio max should dispatch");

    // Then its route label describes the certified hash-first sorted-output path.
    assert_eq!(profiled.profile.route_label, "hash_first");
    assert_eq!(profiled.profile.sort_first_permutation_s, 0.0);
    assert_eq!(profiled.profile.sort_first_segment_scan_s, 0.0);
    assert_eq!(profiled.profile.sort_first_segment_scan_count, 0);
    assert_eq!(profiled.result.values.len(), left.len());
}
