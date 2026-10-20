#[cfg(not(miri))]
use rayon::prelude::*;

const MAX_RADIX_DIGIT_WORKSPACE_BYTES: usize = 512 * 1024 * 1024;

pub(crate) fn check_route_workspace(parts: &[usize]) -> Result<usize, String> {
    let total = parts
        .iter()
        .try_fold(0usize, |sum, &part| sum.checked_add(part))
        .ok_or_else(|| "route workspace overflow".to_string())?;
    if total > MAX_RADIX_DIGIT_WORKSPACE_BYTES {
        return Err("route workspace exceeds cap".to_string());
    }
    Ok(total)
}

fn check_digit_workspace(
    permutation_bytes: usize,
    n: usize,
    workspace: usize,
) -> Result<(), String> {
    check_route_workspace(&[permutation_bytes, n])?;
    check_route_workspace(&[permutation_bytes, workspace])?;
    Ok(())
}

#[cfg(not(test))]
pub(crate) use self::route_vec as scratch_vec;

#[cfg(test)]
pub(crate) fn scratch_vec<T>(len: usize) -> Result<route_allocation_test::Scratch<T>, String> {
    Ok(route_allocation_test::Scratch::new(route_vec(len)?))
}

#[cfg(test)]
pub(crate) mod route_allocation_test {
    use std::cell::RefCell;
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Copy, Debug)]
    pub(crate) enum Failure {
        Reserve,
        Capacity,
    }

    #[derive(Default, Clone, Copy, Debug)]
    pub(crate) struct Snapshot {
        pub(crate) attempts: usize,
        pub(crate) live: usize,
        pub(crate) peak: usize,
    }

    struct State {
        at: usize,
        failure: Failure,
        snapshot: Snapshot,
    }
    thread_local! {
        static STATE: RefCell<Option<Arc<Mutex<State>>>> = const { RefCell::new(None) };
    }

    // Install inside pool.install: allocations occur on the coordinator, while
    // drop observations use an Arc so Rayon-owned scratch can drop on any worker.
    pub(crate) fn with_failure<R>(
        at: usize,
        failure: Failure,
        run: impl FnOnce() -> R,
    ) -> (R, Snapshot) {
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                STATE.with(|slot| *slot.borrow_mut() = None);
            }
        }
        let state = Arc::new(Mutex::new(State {
            at,
            failure,
            snapshot: Snapshot::default(),
        }));
        STATE.with(|slot| {
            assert!(slot.borrow().is_none(), "allocation injection cannot nest");
            *slot.borrow_mut() = Some(state.clone());
        });
        let _reset = Reset;
        let result = run();
        let snapshot = state.lock().unwrap().snapshot;
        (result, snapshot)
    }

    pub(super) fn attempt() -> Option<Failure> {
        STATE.with(|slot| {
            let state = slot.borrow();
            let mut state = state.as_ref()?.lock().unwrap();
            state.snapshot.attempts += 1;
            (state.snapshot.attempts == state.at).then_some(state.failure)
        })
    }

    pub(crate) struct Scratch<T> {
        value: Vec<T>,
        state: Option<Arc<Mutex<State>>>,
    }
    impl<T> Scratch<T> {
        pub(super) fn new(value: Vec<T>) -> Self {
            let state = STATE.with(|slot| slot.borrow().clone());
            if let Some(state) = &state {
                let mut state = state.lock().unwrap();
                state.snapshot.live += 1;
                state.snapshot.peak = state.snapshot.peak.max(state.snapshot.live);
            }
            Self { value, state }
        }
        pub(crate) fn into_vec(mut self) -> Vec<T> {
            std::mem::take(&mut self.value)
        }
    }
    impl<T> std::ops::Deref for Scratch<T> {
        type Target = Vec<T>;
        fn deref(&self) -> &Vec<T> {
            &self.value
        }
    }
    impl<T> std::ops::DerefMut for Scratch<T> {
        fn deref_mut(&mut self) -> &mut Vec<T> {
            &mut self.value
        }
    }
    impl<T> AsRef<[T]> for Scratch<T> {
        fn as_ref(&self) -> &[T] {
            &self.value
        }
    }
    impl<T> Drop for Scratch<T> {
        fn drop(&mut self) {
            // Drop storage before publishing the live-count decrement.
            drop(std::mem::take(&mut self.value));
            if let Some(state) = &self.state {
                state.lock().unwrap().snapshot.live -= 1;
            }
        }
    }
}

// Reject excess capacity rather than silently growing the route's live bound.
pub(crate) fn route_vec<T>(len: usize) -> Result<Vec<T>, String> {
    len.checked_mul(std::mem::size_of::<T>())
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| "route allocation size overflow".to_string())?;
    #[cfg(test)]
    let injected = route_allocation_test::attempt();
    #[cfg(test)]
    if matches!(injected, Some(route_allocation_test::Failure::Reserve)) {
        return Err("injected route reserve failure".to_string());
    }
    let mut result = Vec::new();
    result
        .try_reserve_exact(len)
        .map_err(|error| error.to_string())?;
    let capacity_matches = result.capacity() == len;
    #[cfg(test)]
    let capacity_matches =
        capacity_matches && !matches!(injected, Some(route_allocation_test::Failure::Capacity));
    if !capacity_matches && std::mem::size_of::<T>() != 0 {
        return Err("route allocation exceeded modeled capacity".to_string());
    }
    Ok(result)
}

// D from the route resource contract; permutation buffers and E are separate.
pub(crate) fn checked_route_digit_workspace_bytes(
    n: usize,
    threads: usize,
) -> Result<usize, String> {
    let threads = threads.max(1);
    let chunk_size = (n / threads).max(1024);
    let chunks = n / chunk_size + usize::from(!n.is_multiple_of(chunk_size));
    let active = chunks.min(threads);
    let p = std::mem::size_of::<usize>();
    let s = std::mem::size_of::<&mut [usize]>();
    let h = std::mem::size_of::<Vec<u8>>();
    let bytes = (|| {
        n.checked_add(chunks.checked_mul(256)?.checked_mul(p)?)?
            .checked_add(chunks.checked_mul(256)?.checked_mul(s)?)?
            .checked_add(2usize.checked_mul(chunks)?.checked_mul(h)?)?
            .checked_add(active.checked_mul(256)?.checked_mul(p)?)?
            .checked_add(2usize.checked_mul(256)?.checked_mul(p)?)
    })();
    bytes.ok_or_else(|| "route digit workspace overflow".to_string())
}

pub(crate) fn radix_sort_perm_by_digit_checked<F>(
    n: usize,
    passes: usize,
    digit: F,
) -> Result<Vec<usize>, String>
where
    F: Fn(usize, usize) -> usize + Sync,
{
    use rayon::prelude::*;

    let threads = rayon::current_num_threads().max(1);
    let chunk_size = (n / threads).max(1024);
    let chunks = n / chunk_size + usize::from(!n.is_multiple_of(chunk_size));
    let permutation_bytes = n
        .checked_mul(2 * std::mem::size_of::<usize>())
        .ok_or_else(|| "route permutation workspace overflow".to_string())?;
    let workspace = checked_route_digit_workspace_bytes(n, threads)?;
    check_digit_workspace(permutation_bytes, n, workspace)?;
    let mut a = scratch_vec(n)?;
    a.extend(0..n);
    if n <= 1 || passes == 0 {
        #[cfg(test)]
        let a = a.into_vec();
        return Ok(a);
    }
    let mut b = scratch_vec(n)?;
    b.resize(n, 0);
    let mut buckets = scratch_vec(chunks)?;
    for chunk in a.chunks(chunk_size) {
        let mut digits = scratch_vec(chunk.len())?;
        digits.resize(chunk.len(), 0u8);
        buckets.push(digits);
    }
    let mut counts = scratch_vec(chunks)?;
    counts.resize(chunks, [0usize; 256]);
    for pass in 0..passes {
        a.par_chunks(chunk_size)
            .zip(buckets.par_iter_mut())
            .zip(counts.par_iter_mut())
            .try_for_each(|((chunk, digits), local)| -> Result<(), String> {
                local.fill(0);
                for (&index, slot) in chunk.iter().zip(digits.iter_mut()) {
                    let value = digit(index, pass);
                    if value >= 256 {
                        return Err("radix digit must be in 0..=255".to_string());
                    }
                    *slot = value as u8;
                    local[value] += 1;
                }
                Ok(())
            })?;
        scatter_digit_chunks_checked(&a, &mut b, chunk_size, &buckets[..], &counts)?;
        std::mem::swap(&mut a, &mut b);
    }
    #[cfg(test)]
    let a = a.into_vec();
    Ok(a)
}

// Split every bucket into disjoint, input-chunk-ordered slices before sending
// them to Rayon. Bounds-checked writes need neither shared pointers nor atomics.
pub(crate) fn scatter_digit_chunks(
    input: &[usize],
    output: &mut [usize],
    chunk_size: usize,
    buckets: &[Vec<u8>],
    counts: &[[usize; 256]],
) {
    scatter_digit_chunks_checked(input, output, chunk_size, buckets, counts)
        .expect("radix destination allocation failed");
}

fn scatter_digit_chunks_checked<B: AsRef<[u8]> + Sync>(
    input: &[usize],
    output: &mut [usize],
    chunk_size: usize,
    buckets: &[B],
    counts: &[[usize; 256]],
) -> Result<(), String> {
    use rayon::prelude::*;

    let mut destinations = scratch_vec(counts.len())?;
    for _ in counts {
        destinations.push(scratch_vec::<&mut [usize]>(256)?);
    }
    let mut remaining = output;
    for bucket in 0..256 {
        let total: usize = counts.iter().map(|local| local[bucket]).sum();
        let (mut bucket_output, rest) = remaining.split_at_mut(total);
        remaining = rest;
        for (destination, local) in destinations.iter_mut().zip(counts) {
            let (chunk_output, rest) = bucket_output.split_at_mut(local[bucket]);
            destination.push(chunk_output);
            bucket_output = rest;
        }
    }
    assert!(remaining.is_empty());
    #[cfg(test)]
    let destinations = destinations.into_vec();
    input
        .par_chunks(chunk_size)
        .zip(buckets.par_iter())
        .zip(destinations.into_par_iter())
        .for_each(|((chunk, digits), mut destination)| {
            let mut positions = [0usize; 256];
            for (&index, &digit) in chunk.iter().zip(digits.as_ref()) {
                let digit = usize::from(digit);
                destination[digit][positions[digit]] = index;
                positions[digit] += 1;
            }
        });
    Ok(())
}

pub(crate) fn radix_sort_perm_by_digit_par<F>(n: usize, passes: usize, digit: F) -> Vec<usize>
where
    F: Fn(usize, usize) -> usize + Sync,
{
    if radix_digit_workspace_bytes(n, passes) > MAX_RADIX_DIGIT_WORKSPACE_BYTES {
        return radix_sort_perm_by_digit_comparison(n, passes, &digit);
    }

    #[cfg(miri)]
    {
        return radix_sort_perm_by_digit_seq(n, passes, &digit);
    }

    #[cfg(not(miri))]
    {
        radix_sort_perm_by_digit_rayon(n, passes, digit)
    }
}

fn radix_digit_workspace_bytes(n: usize, passes: usize) -> usize {
    let permutation_bytes = 2usize
        .saturating_mul(n)
        .saturating_mul(std::mem::size_of::<usize>());
    let max_digit_bytes = if passes == 0 { 0 } else { n };
    permutation_bytes.saturating_add(max_digit_bytes)
}

fn radix_sort_perm_by_digit_comparison<F>(n: usize, passes: usize, digit: &F) -> Vec<usize>
where
    F: Fn(usize, usize) -> usize + Sync,
{
    let mut perm: Vec<usize> = (0..n).collect();
    perm.sort_unstable_by(|&left, &right| {
        for pass in (0..passes).rev() {
            let left_digit = digit(left, pass);
            let right_digit = digit(right, pass);
            assert!(left_digit < 256, "radix digit must be in 0..=255");
            assert!(right_digit < 256, "radix digit must be in 0..=255");
            match left_digit.cmp(&right_digit) {
                std::cmp::Ordering::Equal => {}
                ordering => return ordering,
            }
        }
        left.cmp(&right)
    });
    perm
}

#[cfg(miri)]
fn radix_sort_perm_by_digit_seq<F>(n: usize, passes: usize, digit: &F) -> Vec<usize>
where
    F: Fn(usize, usize) -> usize + Sync,
{
    if n <= 1 {
        return (0..n).collect();
    }

    let mut a: Vec<usize> = (0..n).collect();
    let mut b: Vec<usize> = vec![0; n];

    for pass in 0..passes {
        let buckets: Vec<u8> = a
            .iter()
            .map(|&idx| {
                let bucket = digit(idx, pass);
                assert!(bucket < 256, "radix digit must be in 0..=255");
                bucket as u8
            })
            .collect();
        let mut offsets = [0usize; 256];
        for &bucket in &buckets {
            offsets[usize::from(bucket)] += 1;
        }

        let mut sum = 0usize;
        for bucket_count in &mut offsets {
            let count = *bucket_count;
            *bucket_count = sum;
            sum += count;
        }

        for (&idx, bucket) in a.iter().zip(buckets) {
            let bucket = usize::from(bucket);
            let pos = offsets[bucket];
            offsets[bucket] = pos + 1;
            b[pos] = idx;
        }
        std::mem::swap(&mut a, &mut b);
    }

    a
}

#[cfg(not(miri))]
fn radix_sort_perm_by_digit_rayon<F>(n: usize, passes: usize, digit: F) -> Vec<usize>
where
    F: Fn(usize, usize) -> usize + Sync,
{
    if n <= 1 {
        return (0..n).collect();
    }

    let n_threads = rayon::current_num_threads().max(1);
    let chunk_size = (n / n_threads).max(1024);

    let mut a: Vec<usize> = (0..n).collect();
    let mut b: Vec<usize> = vec![0; n];

    #[cfg(debug_assertions)]
    b.fill(usize::MAX);

    for pass in 0..passes {
        let chunk_buckets: Vec<Vec<u8>> = a
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut buckets = Vec::new();
                buckets
                    .try_reserve_exact(chunk.len())
                    .expect("radix digit bucket allocation failed");
                buckets.extend(chunk.iter().map(|&idx| {
                    let bucket = digit(idx, pass);
                    assert!(bucket < 256, "radix digit must be in 0..=255");
                    bucket as u8
                }));
                buckets
            })
            .collect();
        let counts: Vec<[usize; 256]> = chunk_buckets
            .par_iter()
            .map(|buckets| {
                let mut local = [0usize; 256];
                for &bucket in buckets {
                    local[usize::from(bucket)] += 1;
                }
                local
            })
            .collect();

        scatter_digit_chunks(&a, &mut b, chunk_size, &chunk_buckets, &counts);

        #[cfg(debug_assertions)]
        {
            debug_assert!(b.iter().all(|&v| v != usize::MAX));
        }

        std::mem::swap(&mut a, &mut b);
    }

    a
}

#[cfg(test)]
mod tests {
    #[test]
    fn digit_guards_include_exact_cap_and_reject_overflow() {
        const K: usize = 512 * 1024 * 1024;
        for live in [K - 1, K, K + 1] {
            assert_eq!(
                super::check_digit_workspace(16, live - 16, 0).is_ok(),
                live <= K
            );
            assert_eq!(
                super::check_digit_workspace(16, 0, live - 16).is_ok(),
                live <= K
            );
        }
        assert!(super::check_digit_workspace(usize::MAX, 1, 0).is_err());
        assert!(super::check_digit_workspace(usize::MAX, 0, 1).is_err());
        use super::route_allocation_test::{with_failure, Failure};
        let (result, snapshot) = with_failure(1, Failure::Reserve, || {
            super::radix_sort_perm_by_digit_checked(K, 1, |_, _| {
                panic!("cap rejection must precede digit evaluation")
            })
        });
        assert!(result.is_err());
        assert_eq!(snapshot.attempts, 0);
    }

    #[test]
    fn injected_reserve_and_capacity_failures_release_partial_radix_scratch() {
        use super::route_allocation_test::{with_failure, Failure};
        use std::sync::atomic::{AtomicUsize, Ordering};
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        pool.install(|| {
            // C=5. Attempts 5/11 fail after earlier buckets/destinations exist.
            for failure in [Failure::Reserve, Failure::Capacity] {
                for at in 1..=15 {
                    let digit_calls = AtomicUsize::new(0);
                    let (result, snapshot) = with_failure(at, failure, || {
                        super::radix_sort_perm_by_digit_checked(10_003, 1, |row, _| {
                            digit_calls.fetch_add(1, Ordering::Relaxed);
                            row % 7
                        })
                    });
                    assert!(result.is_err(), "missing failure at allocation {at}");
                    let error = result.unwrap_err();
                    match failure {
                        Failure::Reserve => assert!(error.contains("injected route reserve")),
                        Failure::Capacity => assert!(error.contains("exceeded modeled capacity")),
                    }
                    assert_eq!(snapshot.attempts, at);
                    assert_eq!(snapshot.live, 0, "scratch retained at allocation {at}");
                    assert_eq!(
                        digit_calls.load(Ordering::Relaxed),
                        if at <= 9 { 0 } else { 10_003 }
                    );
                    if at > 1 {
                        assert!(snapshot.peak > 0);
                    }
                }
            }
        });
    }

    #[test]
    fn checked_workspace_uses_actual_uneven_chunk_count() {
        let n = 10_003usize;
        let p = std::mem::size_of::<usize>();
        let s = std::mem::size_of::<&mut [usize]>();
        let h = std::mem::size_of::<Vec<u8>>();
        // floor(10003 / 4)=2500, hence five chunks, not four.
        let expected = n + 5 * 256 * p + 5 * 256 * s + 2 * 5 * h + 4 * 256 * p + 2 * 256 * p;
        assert_eq!(
            super::checked_route_digit_workspace_bytes(n, 4).unwrap(),
            expected
        );
        for n in [1023, 1024, 1025] {
            let chunks = n / 1024 + usize::from(n % 1024 != 0);
            let expected = n
                + chunks * 256 * p
                + chunks * 256 * s
                + 2 * chunks * h
                + chunks * 256 * p
                + 2 * 256 * p;
            assert_eq!(
                super::checked_route_digit_workspace_bytes(n, 4).unwrap(),
                expected
            );
        }
        assert!(super::checked_route_digit_workspace_bytes(usize::MAX, 4).is_err());
    }

    #[test]
    fn checked_radix_returns_errors_without_comparison_or_large_allocation() {
        assert!(
            super::radix_sort_perm_by_digit_checked(usize::MAX, 1, |_, _| {
                panic!("preflight must reject before reading digits")
            })
            .is_err()
        );
        assert!(super::radix_sort_perm_by_digit_checked(10, 1, |_, _| 256).is_err());
        assert!(super::route_vec::<usize>(usize::MAX).is_err());
    }

    #[test]
    fn checked_radix_is_stable_across_actual_uneven_chunks() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let actual = pool.install(|| {
            super::radix_sort_perm_by_digit_checked(10_003, 2, |row, pass| {
                ((row % 257) >> (pass * 8)) & 255
            })
            .unwrap()
        });
        let mut expected: Vec<usize> = (0..10_003).collect();
        expected.sort_by_key(|row| row % 257);
        assert_eq!(actual, expected);
    }

    #[test]
    fn radix_sort_orders_by_materialized_digit() {
        let keys = [3usize, 0, 2, 1, 2, 0];
        let perm = super::radix_sort_perm_by_digit_par(keys.len(), 1, |idx, _pass| keys[idx]);

        assert_eq!(perm, vec![1, 5, 3, 2, 4, 0]);
    }

    #[test]
    fn radix_sort_materializes_digits_before_scatter() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let calls = AtomicUsize::new(0);
        let perm = super::radix_sort_perm_by_digit_par(6, 1, |idx, _pass| {
            let call = calls.fetch_add(1, Ordering::SeqCst);
            (idx + call) % 4
        });

        assert_eq!(calls.load(Ordering::SeqCst), 6);
        let mut sorted = perm;
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    #[should_panic(expected = "radix digit must be in 0..=255")]
    fn radix_sort_rejects_out_of_range_digits_before_scatter() {
        let _ = super::radix_sort_perm_by_digit_par(2, 1, |_idx, _pass| 256);
    }

    #[test]
    #[cfg(not(miri))]
    fn radix_sort_rayon_path_orders_multiple_chunks() {
        let n = 4096usize;
        let keys: Vec<usize> = (0..n).map(|idx| (idx * 37 + 11) & 0xFFFF).collect();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("test thread pool builds");

        let perm = pool.install(|| {
            super::radix_sort_perm_by_digit_par(n, 2, |idx, pass| (keys[idx] >> (pass * 8)) & 0xFF)
        });
        let mut expected: Vec<usize> = (0..n).collect();
        expected
            .sort_unstable_by(|&left, &right| keys[left].cmp(&keys[right]).then(left.cmp(&right)));

        assert_eq!(perm, expected);
    }

    #[test]
    fn radix_sort_comparison_fallback_matches_radix_order() {
        let keys = [0x0201usize, 0x0102, 0x0200, 0x0101, 0x0101];
        let perm = super::radix_sort_perm_by_digit_comparison(keys.len(), 2, &|idx, pass| {
            (keys[idx] >> (pass * 8)) & 0xFF
        });

        assert_eq!(perm, vec![3, 4, 1, 2, 0]);
    }

    #[test]
    #[cfg(not(miri))]
    fn parallel_scatter_preserves_duplicate_order_with_uneven_chunks() {
        let n = 10_003;
        for threads in [1, 2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for modulus in [1, 7, 256] {
                let perm = pool
                    .install(|| super::radix_sort_perm_by_digit_par(n, 1, |idx, _| idx % modulus));
                let mut expected: Vec<usize> = (0..n).collect();
                expected.sort_by_key(|idx| idx % modulus);
                assert_eq!(perm, expected);
            }
        }
    }

    #[test]
    fn radix_digit_workspace_is_capped_before_parallel_materialization() {
        let capped = super::radix_digit_workspace_bytes(
            usize::MAX / std::mem::size_of::<usize>(),
            usize::MAX,
        );

        assert_eq!(capped, usize::MAX);
    }
}
