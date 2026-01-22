// Shared radix-sort utilities used for first-seen group ordering.

pub(crate) fn radix_sort_perm_by_u32(keys: &[u32]) -> Vec<usize> {
    let n = keys.len();
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
