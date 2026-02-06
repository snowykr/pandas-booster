// Shared radix-sort utilities used for first-seen group ordering.

const I64_SIGN_MASK: u64 = 1 << 63;

#[inline]
pub(crate) fn i64_to_sortable_u64(value: i64) -> u64 {
    (value as u64) ^ I64_SIGN_MASK
}

pub(crate) fn radix_sort_perm_by_u32(keys: &[u32]) -> Vec<usize> {
    let n = keys.len();

    if n <= 1 {
        return (0..n).collect();
    }

    const SMALL_SORT_THRESHOLD: usize = 2048;
    const RADIX16_THRESHOLD: usize = 131_072;

    if n < SMALL_SORT_THRESHOLD {
        let mut perm: Vec<usize> = (0..n).collect();
        perm.sort_unstable_by(|&i, &j| keys[i].cmp(&keys[j]).then(i.cmp(&j)));
        return perm;
    }

    if n >= RADIX16_THRESHOLD {
        let mut a: Vec<usize> = (0..n).collect();
        let mut b: Vec<usize> = vec![0; n];
        let mut counts = vec![0usize; 1 << 16];

        for shift in [0usize, 16usize] {
            counts.fill(0);
            for &idx in &a {
                let digit = ((keys[idx] >> shift) & 0xFFFF) as usize;
                counts[digit] += 1;
            }

            let mut sum = 0usize;
            for c in &mut counts {
                let tmp = *c;
                *c = sum;
                sum += tmp;
            }

            for &idx in &a {
                let digit = ((keys[idx] >> shift) & 0xFFFF) as usize;
                let pos = counts[digit];
                b[pos] = idx;
                counts[digit] = pos + 1;
            }

            std::mem::swap(&mut a, &mut b);
        }

        return a;
    }

    let mut a: Vec<usize> = (0..n).collect();
    let mut b: Vec<usize> = vec![0; n];

    let mut counts = [0usize; 256];

    // 4 passes for u32 (8 bits each)
    for pass in 0..4 {
        counts.fill(0);
        let shift = pass * 8;

        for &idx in &a {
            let byte = ((keys[idx] >> shift) & 0xFF) as usize;
            counts[byte] += 1;
        }

        // Prefix sum -> counts become write offsets
        let mut sum = 0usize;
        for c in &mut counts {
            let tmp = *c;
            *c = sum;
            sum += tmp;
        }

        // Stable scatter
        for &idx in &a {
            let byte = ((keys[idx] >> shift) & 0xFF) as usize;
            let pos = counts[byte];
            b[pos] = idx;
            counts[byte] = pos + 1;
        }

        std::mem::swap(&mut a, &mut b);
    }

    a
}

pub(crate) fn radix_sort_perm_by_u64(keys: &[u64]) -> Vec<usize> {
    let n = keys.len();
    let mut a: Vec<usize> = (0..n).collect();
    let mut b: Vec<usize> = vec![0; n];

    let mut counts = [0usize; 256];

    // 8 passes for u64 (8 bits each)
    for pass in 0..8 {
        counts.fill(0);
        let shift = pass * 8;

        for &idx in &a {
            let byte = ((keys[idx] >> shift) & 0xFF) as usize;
            counts[byte] += 1;
        }

        // Prefix sum -> counts become write offsets
        let mut sum = 0usize;
        for c in &mut counts {
            let tmp = *c;
            *c = sum;
            sum += tmp;
        }

        // Stable scatter
        for &idx in &a {
            let byte = ((keys[idx] >> shift) & 0xFF) as usize;
            let pos = counts[byte];
            b[pos] = idx;
            counts[byte] = pos + 1;
        }

        std::mem::swap(&mut a, &mut b);
    }

    a
}

// Parallel radix sort for u64 keys, operating on an existing permutation.
// Stable with respect to the input `indices` order.
pub(crate) fn radix_sort_perm_by_u64_for_indices_par(
    keys: &[u64],
    indices: &[usize],
) -> Vec<usize> {
    use rayon::prelude::*;

    let n = indices.len();
    if n <= 1 {
        return indices.to_vec();
    }

    let n_threads = rayon::current_num_threads().max(1);
    let chunk_size = (n / n_threads).max(1024);

    let mut a: Vec<usize> = indices.to_vec();
    let mut b: Vec<usize> = vec![0; n];

    #[cfg(debug_assertions)]
    b.fill(usize::MAX);

    #[derive(Clone, Copy)]
    struct Ptr(usize);
    unsafe impl Send for Ptr {}
    unsafe impl Sync for Ptr {}

    impl Ptr {
        #[inline(always)]
        unsafe fn write(self, pos: usize, val: usize) {
            (self.0 as *mut usize).add(pos).write(val)
        }
    }

    for pass in 0..8 {
        let shift = pass * 8;

        // Phase 1: local histograms per chunk
        let counts: Vec<[usize; 256]> = a
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut local = [0usize; 256];
                for &idx in chunk {
                    let byte = ((keys[idx] >> shift) & 0xFF) as usize;
                    local[byte] += 1;
                }
                local
            })
            .collect();

        // Phase 2: global offsets
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

        // Phase 3: per-thread starting offsets (preserve chunk order)
        let mut thread_offsets: Vec<[usize; 256]> = vec![[0usize; 256]; counts.len()];
        let mut running = offsets;
        for (t, local) in counts.iter().enumerate() {
            thread_offsets[t] = running;
            for bkt in 0..256 {
                running[bkt] += local[bkt];
            }
        }

        // Phase 4: stable scatter in parallel
        let out_ptr = Ptr(b.as_mut_ptr() as usize);
        a.par_chunks(chunk_size)
            .zip(thread_offsets.into_par_iter())
            .for_each(|(chunk, mut write)| {
                for &idx in chunk {
                    let byte = ((keys[idx] >> shift) & 0xFF) as usize;
                    let pos = write[byte];
                    write[byte] = pos + 1;
                    unsafe { out_ptr.write(pos, idx) };
                }
            });

        #[cfg(debug_assertions)]
        {
            debug_assert!(b.iter().all(|&v| v != usize::MAX));
        }

        std::mem::swap(&mut a, &mut b);
    }

    a
}

pub(crate) fn radix_sort_perm_by_i64_par(keys: &[i64]) -> Vec<usize> {
    let mut sortable_keys = Vec::with_capacity(keys.len());
    sortable_keys.extend(keys.iter().map(|&key| i64_to_sortable_u64(key)));
    let indices: Vec<usize> = (0..keys.len()).collect();
    radix_sort_perm_by_u64_for_indices_par(&sortable_keys, &indices)
}
