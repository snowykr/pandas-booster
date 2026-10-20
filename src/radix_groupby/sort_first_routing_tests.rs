use super::sort_first_routing::{
    choose_sort_first_route, memory_budget, SortFirstFallbackReason, SortFirstReducer,
    SortFirstRoute, MAX_KEY_COLUMNS, MIN_SORT_FIRST_ROWS, ROUTE_MEMORY_CAP,
};

fn assert_hash_first(
    reducer: SortFirstReducer,
    keys: &[&[i64]],
    n: usize,
    reason: SortFirstFallbackReason,
) {
    let decision = choose_sort_first_route(reducer, keys, n);
    assert_eq!(decision.route, SortFirstRoute::HashFirst);
    assert_eq!(decision.fallback_reason, Some(reason));
}

#[test]
fn routing_admits_all_four_kernels_only_with_a_proven_lower_bound() {
    for threads in [1, 2, 4] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let n = 20_003;
                let unique: Vec<i64> = (0..n as i64).rev().collect();
                let constant = vec![-7; n];
                for k in [2, MAX_KEY_COLUMNS] {
                    let mut keys = vec![constant.as_slice(); k];
                    keys[0] = &unique;
                    for reducer in [
                        SortFirstReducer::MaxF64,
                        SortFirstReducer::MaxI64,
                        SortFirstReducer::CountF64,
                        SortFirstReducer::CountI64,
                    ] {
                        let decision = choose_sort_first_route(reducer, &keys, n);
                        assert_eq!(decision.route, SortFirstRoute::SortFirst);
                        assert_eq!(decision.fallback_reason, None);
                        assert!(decision.cardinality_lower_bound * 10 >= n * 9);
                        assert!(decision.cardinality_lower_bound <= n);
                        // Real collisions must reduce, never inflate, the proof.
                        assert!(decision.cardinality_lower_bound < n);
                        assert!(decision.modeled_peak_bytes <= ROUTE_MEMORY_CAP);
                        assert_eq!(decision.normalized_input_bytes, 8 * n * (k + 1));
                        assert_eq!(
                            decision.modeled_peak_bytes,
                            memory_budget(n, k).unwrap().peak_bytes
                        );
                    }
                }
            });
    }
}

#[test]
fn routing_rejects_low_medium_correlated_and_periodic_cardinality() {
    let n = 20_003;
    for groups in [1, 64, n * 3 / 4, n * 89 / 100] {
        let left: Vec<i64> = (0..n).map(|row| (row % groups) as i64).collect();
        let right: Vec<i64> = left.iter().map(|&key| key.wrapping_mul(13)).collect();
        let decision = choose_sort_first_route(SortFirstReducer::MaxF64, &[&left, &right], n);
        assert_eq!(decision.route, SortFirstRoute::HashFirst);
        assert_eq!(
            decision.fallback_reason,
            Some(SortFirstFallbackReason::CardinalityLowerBound)
        );
        assert!(decision.cardinality_lower_bound <= groups);
    }
}

#[test]
fn unique_looking_distributed_sample_cannot_admit_a_duplicate_heavy_population() {
    let n = 100_003;
    let mut left = vec![0; n];
    // Every row of a common deterministic 8192-row sample is unique, but
    // the complete population has only 8193 tuples.
    for sample in 0..8192 {
        left[sample * n / 8192] = sample as i64 + 1;
    }
    let right = vec![i64::MIN; n];
    let decision = choose_sort_first_route(SortFirstReducer::CountI64, &[&left, &right], n);
    assert_eq!(decision.route, SortFirstRoute::HashFirst);
    assert!(decision.cardinality_lower_bound <= 8193);
}

#[test]
fn routing_unsupported_reducers_small_inputs_key_counts_and_lengths_remain_diagnostic() {
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
    for n in [0, 1, MIN_SORT_FIRST_ROWS - 1] {
        assert_hash_first(
            SortFirstReducer::MaxF64,
            &[&column[..n], &column[..n]],
            n,
            SortFirstFallbackReason::TooFewRows,
        );
    }
    for k in [0, 1, MAX_KEY_COLUMNS + 1] {
        assert_hash_first(
            SortFirstReducer::MaxF64,
            &vec![column.as_slice(); k],
            column.len(),
            SortFirstFallbackReason::UnsupportedKeyCount,
        );
    }
    for n in [0, MIN_SORT_FIRST_ROWS - 1, MIN_SORT_FIRST_ROWS] {
        assert_hash_first(
            SortFirstReducer::MaxF64,
            &[&column, &column[..MIN_SORT_FIRST_ROWS - 1]],
            n,
            SortFirstFallbackReason::InvalidInput,
        );
    }
}

#[test]
fn checked_budget_observes_real_uneven_chunks_cap_and_overflow_boundaries() {
    for threads in [1, 2, 4] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let budget = memory_budget(100_003, 10).unwrap();
                assert_eq!(budget.chunk_size, (100_003 / threads).max(1024));
                assert_eq!(budget.chunks, 100_003usize.div_ceil(budget.chunk_size));
                assert_eq!(budget.bitmap_bytes, 100_003usize.div_ceil(8) * 8);
                assert!(memory_budget(1_000_000, 10).is_ok());
                assert!(matches!(
                    memory_budget(5_000_000, 10),
                    Err(SortFirstFallbackReason::MemoryLimit)
                ));
                assert!(matches!(
                    memory_budget(64 * 1024 * 1024 + 1, 2),
                    Err(SortFirstFallbackReason::BitmapLimit)
                ));
                assert!(matches!(
                    memory_budget(usize::MAX, 10),
                    Err(SortFirstFallbackReason::ArithmeticOverflow)
                ));
                assert!(matches!(
                    memory_budget(100_003, usize::MAX),
                    Err(SortFirstFallbackReason::ArithmeticOverflow)
                ));
                let (mut lo, mut hi) = (1_000_000, 5_000_000);
                while hi - lo > 1 {
                    let mid = lo + (hi - lo) / 2;
                    if memory_budget(mid, 10).is_ok() {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                assert!(memory_budget(lo, 10).unwrap().peak_bytes <= ROUTE_MEMORY_CAP);
                assert!(matches!(
                    memory_budget(hi, 10),
                    Err(SortFirstFallbackReason::MemoryLimit)
                ));
            });
    }
}

#[test]
fn every_phase_enforces_inclusive_cap_without_large_allocations() {
    use super::sort_first_routing::{calculate_memory_budget, phases_fit};
    let baseline = calculate_memory_budget(5_003, 2).unwrap();
    assert_eq!(
        baseline.peak_bytes,
        *baseline.phase_bytes.iter().max().unwrap()
    );
    for phase in 0..5 {
        for (bytes, accepted) in [
            (ROUTE_MEMORY_CAP - 1, true),
            (ROUTE_MEMORY_CAP, true),
            (ROUTE_MEMORY_CAP + 1, false),
        ] {
            let mut phases = baseline.phase_bytes;
            phases[phase] = bytes;
            assert_eq!(
                phases_fit(&phases, ROUTE_MEMORY_CAP),
                accepted,
                "phase={phase} bytes={bytes}"
            );
        }
    }
    for (n, k) in [
        (usize::MAX, 2),
        (usize::MAX / 8, 2),
        (usize::MAX / 16, 10),
        (1, usize::MAX),
        (5_003, usize::MAX / 8),
    ] {
        assert!(matches!(
            calculate_memory_budget(n, k),
            Err(SortFirstFallbackReason::ArithmeticOverflow)
        ));
    }
}

#[test]
fn bitmap_rounding_and_cap_edges_are_pure_checked_arithmetic() {
    use super::sort_first_routing::{bitmap_storage, calculate_memory_budget};
    for (n, bytes) in [
        (0, 0),
        (1, 8),
        (7, 8),
        (8, 8),
        (9, 16),
        (63, 64),
        (64, 64),
        (65, 72),
    ] {
        let (words, actual) = bitmap_storage(n).unwrap();
        assert_eq!(actual, bytes);
        assert_eq!(words * 8, actual);
        assert!(words * 64 >= n * 8);
    }
    let cap = 64 * 1024 * 1024;
    for n in [cap - 1, cap, cap + 1] {
        let budget = calculate_memory_budget(n, 2).unwrap();
        assert_eq!(budget.bitmap_bytes, n.div_ceil(8) * 8);
        assert_eq!(
            matches!(
                memory_budget(n, 2),
                Err(SortFirstFallbackReason::BitmapLimit)
            ),
            n > cap
        );
    }
    assert!(bitmap_storage(usize::MAX).is_none());
    assert!(bitmap_storage(usize::MAX / 8).is_none());
}

#[test]
fn deterministic_collisions_can_only_reduce_the_cardinality_proof() {
    use super::sort_first_routing::{take_scratch_events, with_collision_bits};
    let n = 20_003;
    for groups in [n, n * 3 / 4] {
        let mut left: Vec<i64> = (0..n).map(|row| (row % groups) as i64).collect();
        // Ensure a unique-looking sample even in the low-global-cardinality case.
        if groups < n {
            left.fill(0);
            for idx in 0..8192 {
                left[idx * n / 8192] = idx as i64 + 1;
            }
        }
        let right = vec![7; n];
        let exact = left
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        for forced in [1, 8, 64] {
            take_scratch_events();
            let decision = with_collision_bits(forced, || {
                choose_sort_first_route(SortFirstReducer::CountI64, &[&left, &right], n)
            });
            assert_eq!(decision.route, SortFirstRoute::HashFirst);
            assert!(
                decision.cardinality_lower_bound > 0,
                "must reach bitmap, not sample rejection"
            );
            assert!(decision.cardinality_lower_bound <= exact.min(forced));
            let events = take_scratch_events();
            assert_eq!(events.last(), Some(&("bitmap_released", 0)));
            assert!(events
                .iter()
                .any(|(phase, bytes)| *phase == "bitmap_live" && *bytes > 0));
        }
    }
}
