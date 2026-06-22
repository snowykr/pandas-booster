#[cfg(not(miri))]
use rayon::prelude::*;

const MAX_RADIX_DIGIT_WORKSPACE_BYTES: usize = 512 * 1024 * 1024;

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

        let mut global = [0usize; 256];
        for local in &counts {
            for bkt in 0..256 {
                global[bkt] += local[bkt];
            }
        }

        let mut offsets = [0usize; 256];
        let mut sum = 0usize;
        for bkt in 0..256 {
            let tmp = global[bkt];
            offsets[bkt] = sum;
            sum += tmp;
        }

        let mut thread_offsets: Vec<[usize; 256]> = vec![[0usize; 256]; counts.len()];
        let mut running = offsets;
        for (t, local) in counts.iter().enumerate() {
            thread_offsets[t] = running;
            for bkt in 0..256 {
                running[bkt] += local[bkt];
            }
        }

        for ((chunk, buckets), mut write) in
            a.chunks(chunk_size).zip(chunk_buckets).zip(thread_offsets)
        {
            for (&idx, bucket) in chunk.iter().zip(buckets) {
                let bucket = usize::from(bucket);
                let pos = write[bucket];
                write[bucket] = pos + 1;
                b[pos] = idx;
            }
        }

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
    fn radix_digit_workspace_is_capped_before_parallel_materialization() {
        let capped = super::radix_digit_workspace_bytes(
            usize::MAX / std::mem::size_of::<usize>(),
            usize::MAX,
        );

        assert_eq!(capped, usize::MAX);
    }
}
