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

// The replaced sampler is retained only as an exact test/diagnostic reference.
fn exact_sample_distinct(keys: &[&[i64]], n: usize) -> usize {
    use super::sort_first_routing::{checked_route_vec, sample_row};
    let len = n.min(8192);
    if len == 0 {
        return 0;
    }
    let mut sample = checked_route_vec::<usize>(len).unwrap();
    for idx in 0..len {
        sample.push(sample_row(idx, n, len).unwrap());
    }
    sample.sort_unstable_by(|left, right| {
        for col in keys {
            let ordering = col[*left].cmp(&col[*right]);
            if !ordering.is_eq() {
                return ordering;
            }
        }
        std::cmp::Ordering::Equal
    });
    1 + sample
        .windows(2)
        .filter(|pair| keys.iter().any(|col| col[pair[0]] != col[pair[1]]))
        .count()
}

#[test]
fn tuple_hash_matches_independently_frozen_vectors() {
    use super::sort_first_routing::tuple_hash;
    for (tuple, expected) in [
        (vec![0, 0], 0x50048009fb7c1a85),
        (vec![1, -1], 0x98a767e3ac621e23),
        (vec![i64::MIN, i64::MAX], 0xbd3a0982fc2f7f25),
        (vec![5, 5, 5], 0xe37e314089057e5e),
        ((0..10).collect::<Vec<i64>>(), 0xbf727afc7dbd4ec7),
    ] {
        let columns: Vec<_> = tuple.iter().map(std::slice::from_ref).collect();
        assert_eq!(tuple_hash(&columns, 0), expected);
    }
}

#[test]
fn sample_occupancy_is_bounded_by_exact_distributed_distinct_count() {
    use super::sort_first_routing::{sample_occupancy, sample_row, with_sample_control};
    for n in [0, 1, 7, 8, 9, 4096, 8191, 8192, 8193, 20_003] {
        for groups in [1, 64, 10_000] {
            let left: Vec<_> = (0..n).map(|row| (row % groups) as i64).collect();
            let right: Vec<_> = left.iter().map(|value| value.wrapping_mul(-13)).collect();
            let keys = [&left[..], &right[..]];
            let exact = exact_sample_distinct(&keys, n);
            assert!(sample_occupancy(&keys, n).unwrap() <= exact);
            for bits in [1, 8, 64] {
                let occupied =
                    with_sample_control(Some(bits), false, || sample_occupancy(&keys, n).unwrap());
                assert!(occupied <= exact.min(bits));
            }
            let len = n.min(8192);
            for idx in 0..len {
                assert_eq!(sample_row(idx, n, len).unwrap(), idx * n / len);
            }
        }
    }
}

#[test]
fn threshold_is_inclusive_checked_and_rounded_up() {
    use super::sort_first_routing::{bitmap_storage, reaches_threshold};
    for rows in [1usize, 9, 10, 11, 4096, 8191, 8192, 20_003] {
        let minimum = (rows * 9).div_ceil(10);
        assert!(!reaches_threshold(minimum - 1, rows).unwrap());
        assert!(reaches_threshold(minimum, rows).unwrap());
        let (words, bytes) = bitmap_storage(rows.min(8192)).unwrap();
        assert_eq!(words, rows.min(8192).div_ceil(8));
        assert_eq!(bytes, words * 8);
        assert!(bytes <= 8192);
    }
    assert!(reaches_threshold(usize::MAX, 1).is_err());
    assert!(reaches_threshold(1, usize::MAX).is_err());
}

#[test]
fn sample_collision_rejects_before_proof_but_positive_sample_cannot_admit() {
    use super::sort_first_routing::{take_scratch_events, with_sample_control};
    let n = 20_003;
    let unique: Vec<_> = (0..n as i64).collect();
    let constant = vec![7; n];
    take_scratch_events();
    let rejected = with_sample_control(Some(1), false, || {
        choose_sort_first_route(SortFirstReducer::MaxI64, &[&unique, &constant], n)
    });
    assert_eq!(rejected.route, SortFirstRoute::HashFirst);
    assert_eq!(rejected.cardinality_lower_bound, 0);
    assert_eq!(
        take_scratch_events(),
        vec![("sample_live", 8192), ("sample_released", 0)]
    );

    let rejected = with_sample_control(None, true, || {
        choose_sort_first_route(SortFirstReducer::CountF64, &[&constant, &constant], n)
    });
    assert_eq!(rejected.route, SortFirstRoute::HashFirst);
    assert_eq!(rejected.cardinality_lower_bound, 1);
    let events = take_scratch_events();
    assert_eq!(events[0], ("sample_live", 8192));
    assert_eq!(events[1], ("sample_released", 0));
    assert_eq!(events[2].0, "bitmap_live");
    assert_eq!(events[3], ("bitmap_released", 0));
}

#[test]
fn routing_allocation_errors_propagate_after_releasing_sample() {
    use super::sort_first_routing::{take_scratch_events, try_choose_sort_first_route};
    use crate::radix_sort_digit::route_allocation_test::{with_failure, Failure};
    let n = 20_003;
    let unique: Vec<_> = (0..n as i64).collect();
    for failure in [Failure::Reserve, Failure::Capacity] {
        for at in [1, 2] {
            take_scratch_events();
            let (result, snapshot) = with_failure(at, failure, || {
                try_choose_sort_first_route(SortFirstReducer::MaxI64, &[&unique, &unique], n)
            });
            assert!(result.is_err());
            assert_eq!(
                snapshot.attempts, at,
                "no hash retry after allocation failure"
            );
            assert_eq!(snapshot.live, 0);
            let events = take_scratch_events();
            if at == 1 {
                assert!(events.is_empty());
            } else {
                assert_eq!(events, vec![("sample_live", 8192), ("sample_released", 0)]);
            }
        }
    }
    let (result, snapshot) = with_failure(usize::MAX, Failure::Reserve, || {
        try_choose_sort_first_route(SortFirstReducer::MaxI64, &[&unique, &unique], n)
    });
    assert_eq!(result.unwrap().route, SortFirstRoute::SortFirst);
    assert_eq!(snapshot.attempts, 2);
    assert_eq!(snapshot.live, 0);
    let events = take_scratch_events();
    assert_eq!(events[1], ("sample_released", 0));
    assert_eq!(events[2].0, "bitmap_live");
    assert_eq!(events.last(), Some(&("bitmap_released", 0)));
}

fn representative_columns(n: usize, domains: [u64; 3]) -> [Vec<i64>; 3] {
    let mut state = 0x123456789abcdef0u64;
    let mut columns = std::array::from_fn(|_| Vec::with_capacity(n));
    for _ in 0..n {
        for (column, domain) in columns.iter_mut().zip(domains) {
            // Bounded SplitMix64 generation; no rejection loop or dependency.
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut word = state;
            word = (word ^ (word >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            word = (word ^ (word >> 27)).wrapping_mul(0x94d049bb133111eb);
            column.push(((word ^ (word >> 31)) % domain) as i64);
        }
    }
    columns
}

#[test]
fn seeded_primary_and_signed_domain_samples_still_require_and_pass_full_proof() {
    use super::sort_first_routing::take_scratch_events;
    let n = 100_003;
    let primary = representative_columns(n, [500, 500, 500]);
    let signed: Vec<_> = (0..n).map(|row| i64::MIN + row as i64).collect();
    let constant = vec![i64::MAX; n];
    for threads in [1, 2, 4] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                for k in [2, 3, MAX_KEY_COLUMNS] {
                    let mut keys = vec![constant.as_slice(); k];
                    if k == 3 {
                        for (key, column) in keys.iter_mut().zip(&primary) {
                            *key = column;
                        }
                    } else {
                        keys[0] = &signed;
                    }
                    for reducer in [
                        SortFirstReducer::MaxF64,
                        SortFirstReducer::MaxI64,
                        SortFirstReducer::CountF64,
                        SortFirstReducer::CountI64,
                    ] {
                        take_scratch_events();
                        let decision = choose_sort_first_route(reducer, &keys, n);
                        assert_eq!(decision.route, SortFirstRoute::SortFirst);
                        assert!(decision.cardinality_lower_bound * 10 >= n * 9);
                        let events = take_scratch_events();
                        assert_eq!(events[1], ("sample_released", 0));
                        assert_eq!(events[2].0, "bitmap_live");
                        assert_eq!(events.last(), Some(&("bitmap_released", 0)));
                    }
                }
            });
    }
}

#[test]
#[ignore = "bounded source-only sampling diagnostic; not a performance gate"]
fn sampler_recovery_probe() {
    use super::sort_first_routing::{reaches_threshold, sample_occupancy};
    use std::hint::black_box;
    use std::time::Instant;
    let n = 5_000_000;
    for (case, domains, primary) in [
        ("low", [10, 10, 10], false),
        ("ordinary", [50, 100, 5], false),
        ("primary", [500, 500, 500], true),
    ] {
        let columns = representative_columns(n, domains);
        let keys: Vec<_> = columns.iter().map(Vec::as_slice).collect();
        let old = reaches_threshold(exact_sample_distinct(&keys, n), 8192).unwrap();
        let new = reaches_threshold(sample_occupancy(&keys, n).unwrap(), 8192).unwrap();
        assert_eq!(old, primary, "{case}: reference decision");
        assert_eq!(new, primary, "{case}: bitmap decision");
        for batch in 0..5 {
            let mut durations = [std::time::Duration::ZERO; 2];
            for sampler in if batch % 2 == 0 { [0, 1] } else { [1, 0] } {
                let start = Instant::now();
                for _ in 0..20 {
                    let occupied = if sampler == 0 {
                        exact_sample_distinct(black_box(&keys), black_box(n))
                    } else {
                        sample_occupancy(black_box(&keys), black_box(n)).unwrap()
                    };
                    black_box(occupied);
                    assert_eq!(reaches_threshold(occupied, 8192).unwrap(), primary);
                }
                durations[sampler] = start.elapsed();
            }
            println!(
                "sampler_recovery_probe case={case} batch={batch} order={} old_ns={} new_ns={} calls=20",
                if batch % 2 == 0 { "old/new" } else { "new/old" },
                durations[0].as_nanos(), durations[1].as_nanos()
            );
        }
        let decision = choose_sort_first_route(SortFirstReducer::MaxI64, &keys, n);
        println!("sampler_recovery_probe case={case} complete_route={decision:?}");
        assert_eq!(decision.route == SortFirstRoute::SortFirst, primary);
        assert!(decision.cardinality_lower_bound <= n);
    }
}
