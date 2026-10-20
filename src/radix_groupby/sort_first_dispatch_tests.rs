use std::collections::BTreeMap;

use super::dispatch::{SortedDispatchDiagnostics, SortedDispatchRoute};
use super::sort_first_routing::{SortFirstFallbackReason, ROUTE_MEMORY_CAP};
use super::test_support::{key_at_out, take_physical_operations, value_at_out};
use super::*;

fn assert_admitted(diagnostics: &SortedDispatchDiagnostics, n: usize) {
    assert_eq!(diagnostics.route, SortedDispatchRoute::SortFirst);
    assert_eq!(diagnostics.hash_first_aggregation_count, 0);
    assert_eq!(diagnostics.post_aggregation_sort_count, 0);
    assert_eq!(diagnostics.routing_decision.fallback_reason, None);
    assert!(diagnostics.routing_decision.cardinality_lower_bound * 10 >= n * 9);
    assert!(diagnostics.routing_decision.modeled_peak_bytes <= ROUTE_MEMORY_CAP);
    let segment = diagnostics
        .sort_first
        .expect("admitted route must report segment execution");
    assert!(segment.lexicographic_permutation_built);
    assert_eq!(segment.segment_scan_count, 1);
    assert_eq!(segment.post_aggregation_sort_count, 0);
    assert_eq!(take_physical_operations(), (0, 0));
}

fn assert_profile_admitted(profile: &MultiKeySortedPhaseProfile, n: usize) {
    assert_eq!(profile.route_label, "sort_first");
    assert_eq!(profile.route_reason, "cardinality_lower_bound_admitted");
    assert_eq!(profile.hash_first_aggregation_count, 0);
    assert_eq!(profile.post_aggregation_sort_count, 0);
    assert!(profile.distinct_lower_bound * 10 >= n * 9);
    assert!(profile.estimated_peak_bytes <= ROUTE_MEMORY_CAP);
    assert_eq!(profile.rayon_threads, rayon::current_num_threads());
    assert_eq!(profile.sort_first_segment_scan_count, 1);
    assert_eq!(take_physical_operations(), (0, 0));
}

fn assert_fallback(diagnostics: &SortedDispatchDiagnostics, reason: SortFirstFallbackReason) {
    assert_eq!(diagnostics.route, SortedDispatchRoute::HashFirst);
    assert_eq!(diagnostics.hash_first_aggregation_count, 1);
    assert_eq!(diagnostics.post_aggregation_sort_count, 1);
    assert_eq!(diagnostics.routing_decision.fallback_reason, Some(reason));
    assert!(diagnostics.sort_first.is_none());
    assert_eq!(take_physical_operations(), (1, 1));
}

// Seven sparse pairs span uneven 1/2/4-thread chunks. Remaining tuples are unique.
fn fixture(fused: bool, order: usize) -> (Vec<Vec<i64>>, Vec<f64>, Vec<i64>) {
    let n = 5_003;
    let mut keys = vec![(0..n as i64).rev().collect::<Vec<_>>(), vec![7; n]];
    if fused {
        keys[0][0] = i64::MIN;
        keys[0][1] = i64::MAX;
        keys[1][0] = i64::MAX;
        keys[1][1] = i64::MIN;
    }
    let mut floats: Vec<f64> = (0..n).map(|row| row as f64 - 100.0).collect();
    let mut integers: Vec<i64> = (0..n).map(|row| row as i64).collect();
    let pairs = [
        (-0.0, 0.0),
        (0.0, -0.0),
        (f64::NAN, 9.0),
        (f64::NAN, f64::NAN),
        (f64::NEG_INFINITY, f64::INFINITY),
        (f64::NEG_INFINITY, f64::NEG_INFINITY),
        (f64::INFINITY, f64::NAN),
    ];
    for (j, (first, second)) in pairs.into_iter().enumerate() {
        let a = n / 4 - 1 + j;
        let b = 3 * n / 4 + j;
        for column in &mut keys {
            column[b] = column[a];
        }
        floats[a] = first;
        floats[b] = second;
        integers[a] = i64::MIN;
        integers[b] = i64::MAX;
    }
    let mut rows: Vec<usize> = (0..n).collect();
    match order {
        0 => {}
        1 => rows.reverse(),
        2 => rows.sort_by_key(|&row| (row * 7919) % n),
        _ => rows.sort_by_key(|&row| (keys[0][row], keys[1][row])),
    }
    (
        keys.iter()
            .map(|col| rows.iter().map(|&row| col[row]).collect())
            .collect(),
        rows.iter().map(|&row| floats[row]).collect(),
        rows.iter().map(|&row| integers[row]).collect(),
    )
}

fn reference_groups(keys: &[Vec<i64>]) -> BTreeMap<Vec<i64>, Vec<usize>> {
    let mut groups = BTreeMap::<Vec<i64>, Vec<usize>>::new();
    for row in 0..keys[0].len() {
        groups
            .entry(keys.iter().map(|col| col[row]).collect())
            .or_default()
            .push(row);
    }
    groups
}

fn assert_keys<V>(result: &GroupByMultiResult<V>, groups: &BTreeMap<Vec<i64>, Vec<usize>>) {
    assert!(
        result.perm.is_none(),
        "sort-first must already emit sorted materialized groups"
    );
    assert_eq!(result.values.len(), groups.len());
    for (out, key) in groups.keys().enumerate() {
        for (col, &expected) in key.iter().enumerate() {
            assert_eq!(key_at_out(result, out, col), expected);
        }
    }
}

fn assert_float_result(
    result: &GroupByMultiResult<f64>,
    groups: &BTreeMap<Vec<i64>, Vec<usize>>,
    values: &[f64],
) {
    assert_keys(result, groups);
    for (out, rows) in groups.values().enumerate() {
        let mut expected = f64::NAN;
        for &row in rows {
            let value = values[row];
            if !value.is_nan() && (expected.is_nan() || value > expected) {
                expected = value;
            }
        }
        let actual = value_at_out(result, out);
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }
}

fn assert_integer_result(
    result: &GroupByMultiResult<i64>,
    groups: &BTreeMap<Vec<i64>, Vec<usize>>,
    values: &[i64],
    count_values: Option<&[f64]>,
    max: bool,
) {
    assert_keys(result, groups);
    for (out, rows) in groups.values().enumerate() {
        let expected = if max {
            rows.iter().map(|&row| values[row]).max().unwrap()
        } else if let Some(floats) = count_values {
            rows.iter().filter(|&&row| !floats[row].is_nan()).count() as i64
        } else {
            rows.len() as i64
        };
        assert_eq!(value_at_out(result, out), expected);
    }
}

#[test]
fn production_all_four_kernels_physically_use_stable_segments_for_ordinary_and_profile_calls() {
    for threads in [1, 2, 4] {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
            for fused in [false, true] {
                for order in 0..4 {
                    let (storage, floats, integers) = fixture(fused, order);
                    let keys: Vec<&[i64]> = storage.iter().map(Vec::as_slice).collect();
                    let groups = reference_groups(&storage);
                    take_physical_operations();
                    let (result, diagnostics) = super::api_sorted::radix_groupby_max_f64_sorted_with_diagnostics(&keys, &floats).unwrap();
                    assert_admitted(&diagnostics, floats.len());
                    assert_float_result(&result, &groups, &floats);
                    take_physical_operations();
                    let profiled = profile_radix_groupby_max_f64_sorted(&keys, &floats).unwrap();
                    assert_profile_admitted(&profiled.profile, floats.len());
                    assert_eq!(profiled.profile.selected_sort_strategy,
                        if fused { "fused_multi_key_radix" } else { "packed_u64" });
                    assert_float_result(&profiled.result, &groups, &floats);

                    macro_rules! integer_kernel {
                        ($ordinary:ident, $profile:ident, $values:expr, $count_values:expr, $max:expr) => {{
                            take_physical_operations();
                            let (result, diagnostics) = super::api_sorted::$ordinary(&keys, $values).unwrap();
                            assert_admitted(&diagnostics, floats.len());
                            assert_integer_result(&result, &groups, &integers, $count_values, $max);
                            take_physical_operations();
                            let profiled = $profile(&keys, $values).unwrap();
                            assert_profile_admitted(&profiled.profile, floats.len());
                            assert_integer_result(&profiled.result, &groups, &integers, $count_values, $max);
                        }};
                    }
                    integer_kernel!(radix_groupby_max_i64_sorted_with_diagnostics, profile_radix_groupby_max_i64_sorted, &integers, None, true);
                    integer_kernel!(radix_groupby_count_f64_sorted_with_diagnostics, profile_radix_groupby_count_f64_sorted, &floats, Some(floats.as_slice()), false);
                    integer_kernel!(radix_groupby_count_i64_sorted_with_diagnostics, profile_radix_groupby_count_i64_sorted, &integers, None, false);
                }
            }
        });
    }
}

#[test]
fn production_low_medium_and_over_key_limit_fallbacks_observe_hash_and_postsort() {
    for (n, groups, k, reason) in [
        (5_003, 64, 2, SortFirstFallbackReason::CardinalityLowerBound),
        (
            5_003,
            3_700,
            2,
            SortFirstFallbackReason::CardinalityLowerBound,
        ),
        (
            5_003,
            5_003,
            11,
            SortFirstFallbackReason::UnsupportedKeyCount,
        ),
        (17, 17, 2, SortFirstFallbackReason::TooFewRows),
    ] {
        let storage: Vec<Vec<i64>> = (0..k)
            .map(|_| (0..n).map(|row| (row % groups) as i64).collect())
            .collect();
        let keys: Vec<&[i64]> = storage.iter().map(Vec::as_slice).collect();
        let values = vec![1i64; n];
        take_physical_operations();
        let (_, diagnostics) =
            super::api_sorted::radix_groupby_count_i64_sorted_with_diagnostics(&keys, &values)
                .unwrap();
        assert_fallback(&diagnostics, reason);
        take_physical_operations();
        let profiled = profile_radix_groupby_count_i64_sorted(&keys, &values).unwrap();
        assert_eq!(profiled.profile.route_label, "hash_first");
        assert_eq!(profiled.profile.hash_first_aggregation_count, 1);
        assert_eq!(profiled.profile.post_aggregation_sort_count, 1);
        assert_eq!(take_physical_operations(), (1, 1));
    }
}

#[test]
fn unsupported_reducers_remain_physical_hash_first() {
    let (storage, values, _) = fixture(false, 0);
    let keys: Vec<&[i64]> = storage.iter().map(Vec::as_slice).collect();
    take_physical_operations();
    radix_groupby_sum_f64_sorted(&keys, &values).unwrap();
    assert_eq!(take_physical_operations(), (1, 1));
    radix_groupby_min_f64_sorted(&keys, &values).unwrap();
    assert_eq!(take_physical_operations(), (1, 1));
}

#[test]
fn rejection_releases_route_scratch_before_hash_fallback() {
    use super::sort_first_routing::{take_scratch_events, with_collision_bits};
    rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap()
        .install(|| {
            let n = 5_003;
            let constant = vec![1; n];
            let unique: Vec<i64> = (0..n as i64).collect();
            let values = vec![1i64; n];
            for bitmap_rejection in [false, true] {
                take_scratch_events();
                take_physical_operations();
                let left = if bitmap_rejection { &unique } else { &constant };
                with_collision_bits(1, || {
                    let (_, diagnostics) =
                        super::api_sorted::radix_groupby_count_i64_sorted_with_diagnostics(
                            &[left, &constant],
                            &values,
                        )
                        .unwrap();
                    assert_fallback(&diagnostics, SortFirstFallbackReason::CardinalityLowerBound);
                });
                let events = take_scratch_events();
                assert_eq!(events.last(), Some(&("hash_fallback", 0)));
                let sample_release = events
                    .iter()
                    .position(|event| *event == ("sample_released", 0))
                    .unwrap();
                if bitmap_rejection {
                    let bitmap_live = events
                        .iter()
                        .position(|(phase, _)| *phase == "bitmap_live")
                        .unwrap();
                    assert!(sample_release < bitmap_live);
                    assert_eq!(events[events.len() - 2], ("bitmap_released", 0));
                } else {
                    assert_eq!(events[events.len() - 2], ("sample_released", 0));
                    assert!(!events.iter().any(|(phase, _)| *phase == "bitmap_live"));
                }
            }
        });
}

#[test]
fn admitted_engine_releases_boundary_and_permutation_scratch_before_conversion() {
    use super::sort_first_routing::take_scratch_events;
    rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap()
        .install(|| {
            let (storage, _, values) = fixture(false, 0);
            let keys: Vec<&[i64]> = storage.iter().map(Vec::as_slice).collect();
            take_scratch_events();
            take_physical_operations();
            let (result, diagnostics) =
                super::api_sorted::radix_groupby_count_i64_sorted_with_diagnostics(&keys, &values)
                    .unwrap();
            assert_admitted(&diagnostics, values.len());
            let events = take_scratch_events();
            let bytes = |phase| events.iter().find(|(name, _)| *name == phase).unwrap().1;
            let p = std::mem::size_of::<usize>();
            assert_eq!(bytes("radix_returned"), values.len() * (p + 8));
            assert_eq!(bytes("boundary_live"), values.len() * (p + 8 + 1));
            assert_eq!(bytes("packed_released"), values.len() * (p + 1));
            let position = |phase| events.iter().position(|(name, _)| *name == phase).unwrap();
            assert!(position("boundary_live") < position("packed_released"));
            assert!(position("packed_released") < position("starts_live"));
            assert!(bytes("starts_live") > bytes("boundary_released"));
            assert_eq!(
                bytes("boundary_released"),
                (values.len() + result.values.len() + 1) * p
            );
            assert_eq!(events.last(), Some(&("conversion_ready", 0)));
            assert!(!events.iter().any(|(phase, _)| *phase == "hash_fallback"));
        });
}

#[test]
fn production_packed_boundaries_cover_duplicates_extrema_exact_width_and_constants() {
    use super::sort_first::production_sort_first_groupby;
    use super::sort_first_routing::take_scratch_events;
    use crate::aggregation::{CountAggI64, MaxAggF64};
    use crate::radix_sort_multi_key::MultiKeySortStrategy;

    let n = 5_003;
    let extrema = [i64::MIN, i64::MAX, 0, i64::MIN, -1];
    let narrow = [i64::MIN, i64::MIN + u32::MAX as i64, i64::MIN + 1];
    let fixtures = [
        (
            vec![
                (0..n).map(|row| (row * 37 % 127) as i64 - 63).collect(),
                (0..n).map(|row| (row * 11 % 31) as i64 - 15).collect(),
            ],
            MultiKeySortStrategy::PackedU64,
            None,
        ),
        (
            vec![(0..n).map(|row| extrema[row % 5]).collect(), vec![7; n]],
            MultiKeySortStrategy::PackedU64,
            Some(64),
        ),
        (
            vec![
                (0..n).map(|row| narrow[row % 3]).collect(),
                (0..n).map(|row| narrow[(row / 3) % 3]).collect(),
            ],
            MultiKeySortStrategy::PackedU64,
            Some(64),
        ),
        (
            vec![vec![i64::MIN; n], vec![i64::MAX; n]],
            MultiKeySortStrategy::PackedU64,
            Some(0),
        ),
        (
            vec![
                (0..n).map(|row| extrema[row % 5]).collect(),
                (0..n).map(|row| extrema[(row / 3) % 5]).collect(),
            ],
            MultiKeySortStrategy::FusedMultiKeyRadix,
            Some(128),
        ),
    ];
    let floats: Vec<f64> = (0..n).map(|row| [-0.0, 0.0, f64::NAN][row % 3]).collect();
    let integers = vec![1i64; n];
    for threads in [1, 2, 4] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                for (storage, strategy, width) in &fixtures {
                    let keys: Vec<&[i64]> = storage.iter().map(Vec::as_slice).collect();
                    let groups = reference_groups(storage);
                    take_scratch_events();
                    let (result, _, _, proof) =
                        production_sort_first_groupby::<f64, MaxAggF64, f64>(&keys, &floats)
                            .unwrap();
                    assert_float_result(&result, &groups, &floats);
                    assert_eq!(proof.strategy, *strategy);
                    if let Some(width) = width {
                        assert_eq!(proof.total_bit_width, *width);
                    }
                    let events = take_scratch_events();
                    let bytes = |phase| events.iter().find(|(name, _)| *name == phase).unwrap().1;
                    let packed_bytes = if *strategy == MultiKeySortStrategy::PackedU64
                        && proof.total_bit_width != 0
                    {
                        8 * n
                    } else {
                        0
                    };
                    assert_eq!(
                        bytes("boundary_live") - bytes("packed_released"),
                        packed_bytes
                    );
                    let (count, _, _, _) =
                        production_sort_first_groupby::<i64, CountAggI64, i64>(&keys, &integers)
                            .unwrap();
                    assert_integer_result(&count, &groups, &integers, None, false);
                }
            });
    }
}

#[test]
fn packed_discovery_fits_existing_boundary_envelope_with_checked_limits() {
    use super::sort_first_routing::{calculate_memory_budget, phases_fit};
    for threads in [1, 2, 4] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                for n in [0usize, 1, 5_003, 5_000_000] {
                    let budget = calculate_memory_budget(n, 2).unwrap();
                    let p = std::mem::size_of::<usize>();
                    let overhead = (threads + budget.chunks + 2) * 4096 + 65536;
                    let discovery = n * (p + 1 + 8) + overhead;
                    let setup = n * (p + 1) + (n + 1) * p + budget.chunks * 2 * p + overhead;
                    assert_eq!(budget.phase_bytes[2], discovery.max(setup));
                    if p == 8 {
                        assert_eq!(budget.phase_bytes[2], setup);
                    }
                }
                assert!(calculate_memory_budget(usize::MAX, 2).is_err());
            });
    }
    for bytes in [ROUTE_MEMORY_CAP - 1, ROUTE_MEMORY_CAP, ROUTE_MEMORY_CAP + 1] {
        assert_eq!(
            phases_fit(&[bytes; 5], ROUTE_MEMORY_CAP),
            bytes <= ROUTE_MEMORY_CAP
        );
    }
}

#[test]
fn every_route_allocation_error_propagates_without_hash_retry() {
    use crate::radix_sort_digit::route_allocation_test::{with_failure, Failure};
    rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap()
        .install(|| {
            let (storage, _, values) = fixture(false, 0);
            let keys: Vec<&[i64]> = storage.iter().map(Vec::as_slice).collect();
            // A successful discovery run gives the exact finite allocation count;
            // no hard-coded assumptions about radix pass count or layout buffers.
            let (result, discovery) = with_failure(usize::MAX, Failure::Reserve, || {
                super::api_sorted::radix_groupby_count_i64_sorted_with_diagnostics(&keys, &values)
            });
            result.unwrap();
            assert!(discovery.attempts > 0);
            assert_eq!(
                discovery.live, 0,
                "packed scratch survived production return"
            );
            for failure in [Failure::Reserve, Failure::Capacity] {
                for fail_at in 1..=discovery.attempts {
                    take_physical_operations();
                    let (result, observation) = with_failure(fail_at, failure, || {
                        super::api_sorted::radix_groupby_count_i64_sorted_with_diagnostics(
                            &keys, &values,
                        )
                    });
                    assert!(
                        result.is_err(),
                        "allocation index={fail_at}, failure={failure:?}"
                    );
                    assert_eq!(observation.attempts, fail_at);
                    assert_eq!(observation.live, 0);
                    assert_eq!(take_physical_operations(), (0, 0));
                }
            }
            // The scoped injector must not poison a later call on this same worker.
            let (_, diagnostics) =
                super::api_sorted::radix_groupby_count_i64_sorted_with_diagnostics(&keys, &values)
                    .unwrap();
            assert_admitted(&diagnostics, values.len());
        });
}
