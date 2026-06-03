use rayon::prelude::*;
use std::ptr::NonNull;

const NUM_PARTITIONS_BITS: usize = 10;
pub(super) const NUM_PARTITIONS: usize = 1 << NUM_PARTITIONS_BITS;
pub(crate) const SMALL_DIRECT_THRESHOLD_ELEMS: usize = 200_000;

#[derive(Clone, Copy)]
struct PermPtr(NonNull<usize>);

// SAFETY: `PermPtr` only exposes disjoint indexed writes to the allocation
// owned by `stable_scatter_by_partition`. The wrapped pointer always refers to
// the `perm` buffer created in the same function, and each `pos` is written at
// most once across threads by construction of the partition offsets.
unsafe impl Send for PermPtr {}
// SAFETY: shared access to `PermPtr` only allows indexed writes through the
// per-thread `write` API above. The algorithm guarantees non-overlapping
// positions between chunks, so concurrent reads of the wrapper cannot race on
// the underlying memory.
unsafe impl Sync for PermPtr {}

impl PermPtr {
    #[inline]
    unsafe fn write(self, pos: usize, row: usize) {
        // SAFETY: callers must ensure `pos < perm.len()` and that each `pos` is written
        // exactly once (no concurrent writes to the same location).
        unsafe { *self.0.as_ptr().add(pos) = row };
    }
}
fn hash_to_partition(hash: u64) -> usize {
    (hash as usize) & (NUM_PARTITIONS - 1)
}

pub(crate) fn stable_scatter_by_partition(hashes: &[u64]) -> (Vec<usize>, Vec<usize>) {
    let n_rows = hashes.len();
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_size = (n_rows / n_threads).max(1024);

    let counts: Vec<[usize; NUM_PARTITIONS]> = hashes
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut local = [0usize; NUM_PARTITIONS];
            for &h in chunk {
                local[hash_to_partition(h)] += 1;
            }
            local
        })
        .collect();

    let mut offsets = vec![0usize; NUM_PARTITIONS + 1];
    for local in &counts {
        for p in 0..NUM_PARTITIONS {
            offsets[p + 1] += local[p];
        }
    }
    for p in 0..NUM_PARTITIONS {
        offsets[p + 1] += offsets[p];
    }

    let mut thread_offsets: Vec<[usize; NUM_PARTITIONS]> =
        vec![[0usize; NUM_PARTITIONS]; counts.len()];
    let mut running = [0usize; NUM_PARTITIONS];
    running[..NUM_PARTITIONS].copy_from_slice(&offsets[..NUM_PARTITIONS]);
    for (chunk_idx, local) in counts.iter().enumerate() {
        thread_offsets[chunk_idx] = running;
        for p in 0..NUM_PARTITIONS {
            running[p] += local[p];
        }
    }

    let mut perm = vec![0usize; n_rows];
    #[cfg(debug_assertions)]
    perm.fill(usize::MAX);
    let perm_ptr = PermPtr(NonNull::new(perm.as_mut_ptr()).expect("perm allocation is non-null"));

    hashes
        .par_chunks(chunk_size)
        .enumerate()
        .zip(thread_offsets.into_par_iter())
        .for_each(|((chunk_idx, chunk), mut write)| {
            let base = chunk_idx * chunk_size;
            for (i, &h) in chunk.iter().enumerate() {
                let row = base + i;
                let p = hash_to_partition(h);
                let pos = write[p];
                write[p] = pos + 1;
                // SAFETY:
                // - `pos` is unique within this chunk due to local `write[p]` increments.
                // - Chunk starting offsets are disjoint across chunks.
                unsafe { perm_ptr.write(pos, row) };
            }
        });

    #[cfg(debug_assertions)]
    {
        debug_assert!(perm.iter().all(|&r| r != usize::MAX));
        debug_assert!(perm.iter().all(|&r| r < n_rows));
    }

    (perm, offsets)
}

#[cfg(test)]
mod tests {
    use super::PermPtr;
    use std::ptr::NonNull;

    #[test]
    fn perm_ptr_writes_unique_slots() {
        let mut values = vec![usize::MAX; 4];
        let ptr = PermPtr(NonNull::new(values.as_mut_ptr()).expect("vec pointer is non-null"));

        for (pos, row) in [3, 2, 1, 0].into_iter().enumerate() {
            // SAFETY: `pos` is within `values`, and each slot is written once.
            unsafe { ptr.write(pos, row) };
        }

        assert_eq!(values, [3, 2, 1, 0]);
    }
}
