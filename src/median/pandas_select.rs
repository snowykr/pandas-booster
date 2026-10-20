// Preserve pandas' exact pivot and swap sequence, including signed-zero ties,
// without repeatedly scanning a nearly sorted suffix. A min/max tree over fixed
// size blocks locates each scan's stopping block in O(log n); only that block
// and the starting block need a linear scan. Scratch space is O(n / BLOCK_LEN).
const BLOCK_LEN: usize = 64;

// For n > 64, 32 * next_power_of_two(ceil(n/64)) < n + 64 <= 2*n
// bytes. Smaller groups allocate no tree. This also bounds concurrent groups
// by their total value count; routing accounts for it before memory slack.
pub(crate) const SCRATCH_BYTES_PER_VALUE: usize = 2;

struct ScanIndex {
    leaves: usize,
    extrema: Vec<(f64, f64)>,
    #[cfg(test)]
    work: usize,
}

impl ScanIndex {
    fn new(values: &[f64]) -> Self {
        if values.len() <= BLOCK_LEN {
            return Self {
                leaves: 0,
                extrema: Vec::new(),
                #[cfg(test)]
                work: 0,
            };
        }
        let blocks = values.len().div_ceil(BLOCK_LEN);
        let leaves = blocks.next_power_of_two();
        let mut index = Self {
            leaves,
            extrema: vec![(f64::INFINITY, f64::NEG_INFINITY); 2 * leaves],
            #[cfg(test)]
            work: 0,
        };
        for block in 0..blocks {
            index.summarize(values, block);
        }
        for node in (1..leaves).rev() {
            index.pull(node);
        }
        index
    }

    fn tick(&mut self) {
        #[cfg(test)]
        {
            self.work += 1;
        }
    }

    fn summarize(&mut self, values: &[f64], block: usize) {
        let start = block * BLOCK_LEN;
        let end = (start + BLOCK_LEN).min(values.len());
        let mut bounds = (f64::INFINITY, f64::NEG_INFINITY);
        for &value in &values[start..end] {
            self.tick();
            bounds.0 = bounds.0.min(value);
            bounds.1 = bounds.1.max(value);
        }
        self.extrema[self.leaves + block] = bounds;
    }

    fn pull(&mut self, node: usize) {
        self.tick();
        let a = self.extrema[node * 2];
        let b = self.extrema[node * 2 + 1];
        self.extrema[node] = (a.0.min(b.0), a.1.max(b.1));
    }

    fn update(&mut self, values: &[f64], position: usize) {
        if self.leaves == 0 {
            return;
        }
        let block = position / BLOCK_LEN;
        self.summarize(values, block);
        let mut node = (self.leaves + block) / 2;
        while node > 0 {
            self.pull(node);
            node /= 2;
        }
    }

    // First/last qualifying block in the half-open block interval [start, end).
    fn find(&mut self, node: usize, lo: usize, hi: usize, query: &ScanQuery) -> Option<usize> {
        self.tick();
        let (min, max) = self.extrema[node];
        if hi <= query.start
            || lo >= query.end
            || min > max
            || (query.forward && max < query.pivot)
            || (!query.forward && min > query.pivot)
        {
            return None;
        }
        if hi - lo == 1 {
            return Some(lo);
        }
        let mid = lo + (hi - lo) / 2;
        let children = if query.forward {
            [(node * 2, lo, mid), (node * 2 + 1, mid, hi)]
        } else {
            [(node * 2 + 1, mid, hi), (node * 2, lo, mid)]
        };
        for (child, start, end) in children {
            if let Some(block) = self.find(child, start, end, query) {
                return Some(block);
            }
        }
        None
    }

    fn scan(&mut self, values: &[f64], start: usize, pivot: f64, forward: bool) -> usize {
        let block = start / BLOCK_LEN;
        let range = if forward {
            start..((block + 1) * BLOCK_LEN).min(values.len())
        } else {
            block * BLOCK_LEN..start + 1
        };
        let matches = |value: f64| {
            if forward {
                value >= pivot
            } else {
                value <= pivot
            }
        };
        if forward {
            for position in range {
                self.tick();
                if matches(values[position]) {
                    return position;
                }
            }
        } else {
            for position in range.rev() {
                self.tick();
                if matches(values[position]) {
                    return position;
                }
            }
        }
        let query = ScanQuery {
            start: if forward { block + 1 } else { 0 },
            end: if forward { self.leaves } else { block },
            pivot,
            forward,
        };
        // The pivot or a previously swapped boundary value is a scan sentinel,
        // exactly as in pandas' unbounded inner scans.
        let block = self
            .find(1, 0, self.leaves, &query)
            .expect("pivot sentinel");
        let range = block * BLOCK_LEN..((block + 1) * BLOCK_LEN).min(values.len());
        if forward {
            for position in range {
                self.tick();
                if matches(values[position]) {
                    return position;
                }
            }
        } else {
            for position in range.rev() {
                self.tick();
                if matches(values[position]) {
                    return position;
                }
            }
        }
        unreachable!("block extrema contain a scan sentinel")
    }
}

struct ScanQuery {
    start: usize,
    end: usize,
    pivot: f64,
    forward: bool,
}

pub(super) fn median(values: &mut [f64]) -> f64 {
    let mid = values.len() / 2;
    let mut index = ScanIndex::new(values);
    let upper = select(values, mid, &mut index);
    if values.len() % 2 == 1 {
        upper
    } else {
        let lower = select(values, mid - 1, &mut index);
        super::average_f64_middle_values(upper, lower)
    }
}

fn select(values: &mut [f64], k: usize, index: &mut ScanIndex) -> f64 {
    let target = k as isize;
    let mut left = 0isize;
    let mut right = values.len() as isize - 1;
    while left < right {
        let pivot = values[k];
        let mut i = left;
        let mut j = right;
        loop {
            i = index.scan(values, i as usize, pivot, true) as isize;
            j = index.scan(values, j as usize, pivot, false) as isize;
            if i <= j {
                values.swap(i as usize, j as usize);
                if i as usize / BLOCK_LEN != j as usize / BLOCK_LEN {
                    index.update(values, i as usize);
                    index.update(values, j as usize);
                }
                i += 1;
                j -= 1;
            }
            if i > j {
                break;
            }
        }
        // Each partition discards at least the number of swaps;
        // even a maximally unbalanced partition discards at least one index.
        // Thus O(n) scans/swaps, each O(BLOCK_LEN + log n), bound total work.
        if j < target {
            left = i;
        }
        if target < i {
            right = j;
        }
    }
    values[k]
}

#[cfg(test)]
mod tests {
    use super::{select, ScanIndex, SCRATCH_BYTES_PER_VALUE};

    #[test]
    fn index_scratch_fits_routing_bound_including_rounding() {
        for n in [1, 2, 63, 64, 65, 127, 128, 129, 1025, 1_000_000] {
            let index = ScanIndex::new(&vec![0.0; n]);
            assert!(
                index.extrema.len() * std::mem::size_of::<(f64, f64)>()
                    <= n * SCRATCH_BYTES_PER_VALUE
            );
            if n <= 64 {
                assert_eq!(index.extrema.capacity(), 0);
            }
        }
    }

    // Independent literal pandas quickselect oracle: deliberately unoptimized,
    // only used on small inputs to certify every swap, not just numeric rank.
    fn pandas_reference(values: &mut [f64], k: usize) -> f64 {
        let mut left = 0isize;
        let mut right = values.len() as isize - 1;
        while left < right {
            let pivot = values[k];
            let (mut i, mut j) = (left, right);
            loop {
                while values[i as usize] < pivot {
                    i += 1;
                }
                while pivot < values[j as usize] {
                    j -= 1;
                }
                if i <= j {
                    values.swap(i as usize, j as usize);
                    i += 1;
                    j -= 1;
                }
                if i > j {
                    break;
                }
            }
            if j < k as isize {
                left = i;
            }
            if (k as isize) < i {
                right = j;
            }
        }
        values[k]
    }

    fn check_sequence(mut values: Vec<f64>) {
        let mut reference = values.clone();
        let mut index = ScanIndex::new(&values);
        for k in [values.len() / 2, (values.len() - 1) / 2] {
            assert_eq!(
                select(&mut values, k, &mut index).to_bits(),
                pandas_reference(&mut reference, k).to_bits()
            );
            assert!(values
                .iter()
                .zip(&reference)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }

    #[test]
    fn indexed_selection_preserves_pandas_swaps_across_blocks() {
        let alphabet = [f64::NEG_INFINITY, -1.0, -0.0, 0.0, 1.0, f64::INFINITY];
        for len in 1..=6 {
            for mut code in 0..6usize.pow(len) {
                let values = (0..len)
                    .map(|_| {
                        let value = alphabet[code % 6];
                        code /= 6;
                        value
                    })
                    .collect();
                check_sequence(values);
            }
        }
        let mut state = 123456789_u64;
        for len in [63, 64, 65, 127, 128, 129, 257, 1025] {
            for _ in 0..100 {
                let values = (0..len)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        alphabet[state as usize % alphabet.len()]
                    })
                    .collect();
                check_sequence(values);
            }
        }
    }

    #[test]
    fn mixed_zero_adversary_has_bounded_index_work() {
        for mid in [1_000, 10_000, 100_000] {
            let mut values: Vec<f64> = (1..=mid).map(f64::from).collect();
            values.push(-0.0);
            values.extend((mid + 1..=2 * mid).map(f64::from));
            values.push(0.0);
            let n = values.len();
            let mut index = ScanIndex::new(&values);
            let upper = select(&mut values, n / 2, &mut index);
            let lower = select(&mut values, n / 2 - 1, &mut index);
            assert_eq!((lower + upper) / 2.0, f64::from(mid) - 0.5);
            let bound = n * (512 + 32 * n.ilog2() as usize);
            println!(
                "mixed-zero n={n}: indexed work={} bound={bound}",
                index.work
            );
            assert!(index.work <= bound);
        }
    }
}
