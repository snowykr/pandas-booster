use std::mem::size_of;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::radix_sort_digit::checked_route_digit_workspace_bytes;
pub(super) use crate::radix_sort_digit::route_vec as checked_route_vec;
use rayon::prelude::*;

pub(super) const MAX_KEY_COLUMNS: usize = 10;
pub(super) const MIN_SORT_FIRST_ROWS: usize = 4_096;
pub(super) const ROUTE_MEMORY_CAP: usize = 512 * 1024 * 1024;
const BITMAP_CAP: usize = 64 * 1024 * 1024;

#[cfg(test)]
thread_local! {
    static COLLISION_BITS: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    static SAMPLE_COLLISION_BITS: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    static SAMPLE_POSITIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static SCRATCH_EVENTS: std::cell::RefCell<Vec<(&'static str, usize)>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
pub(super) fn observe_scratch(phase: &'static str, live_bytes: usize) {
    SCRATCH_EVENTS.with(|events| events.borrow_mut().push((phase, live_bytes)));
}

#[cfg(test)]
pub(super) fn take_scratch_events() -> Vec<(&'static str, usize)> {
    SCRATCH_EVENTS.with(|events| std::mem::take(&mut *events.borrow_mut()))
}

#[cfg(test)]
pub(super) fn with_collision_bits<R>(bits: usize, action: impl FnOnce() -> R) -> R {
    struct Restore(Option<usize>);
    impl Drop for Restore {
        fn drop(&mut self) {
            COLLISION_BITS.with(|slot| slot.set(self.0));
        }
    }
    let _restore = Restore(COLLISION_BITS.with(|slot| slot.replace(Some(bits.max(1)))));
    action()
}

#[cfg(test)]
pub(super) fn with_sample_control<R>(
    bits: Option<usize>,
    positive: bool,
    action: impl FnOnce() -> R,
) -> R {
    struct Restore(Option<usize>, bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            SAMPLE_COLLISION_BITS.with(|slot| slot.set(self.0));
            SAMPLE_POSITIVE.with(|slot| slot.set(self.1));
        }
    }
    let _restore = Restore(
        SAMPLE_COLLISION_BITS.with(|slot| slot.replace(bits.map(|bits| bits.max(1)))),
        SAMPLE_POSITIVE.with(|slot| slot.replace(positive)),
    );
    action()
}

pub(super) fn tuple_hash(key_slices: &[&[i64]], row: usize) -> u64 {
    let mut hash = 0x6a09e667f3bcc909u64;
    for col in key_slices {
        let mut word = (col[row] as u64).wrapping_add(0x9e3779b97f4a7c15);
        word = (word ^ (word >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        word = (word ^ (word >> 27)).wrapping_mul(0x94d049bb133111eb);
        hash ^= word ^ (word >> 31);
        hash = hash
            .rotate_left(27)
            .wrapping_mul(0x3c79ac492ba7b653)
            .wrapping_add(0x1c69b3f74ac4ae35);
    }
    // Avalanche the final tuple, including correlated adjacent integer keys.
    hash = (hash ^ (hash >> 33)).wrapping_mul(0xff51afd7ed558ccd);
    hash = (hash ^ (hash >> 33)).wrapping_mul(0xc4ceb9fe1a85ec53);
    hash ^ (hash >> 33)
}

pub(super) fn sample_row(idx: usize, n_rows: usize, sample_len: usize) -> Result<usize, String> {
    idx.checked_mul(n_rows / sample_len)
        .and_then(|base| {
            idx.checked_mul(n_rows % sample_len)
                .and_then(|remainder| base.checked_add(remainder / sample_len))
        })
        .ok_or_else(|| "sort-first sample index overflow".to_string())
}

pub(super) fn reaches_threshold(occupied: usize, rows: usize) -> Result<bool, String> {
    occupied
        .checked_mul(10)
        .zip(rows.checked_mul(9))
        .map(|(occupied, threshold)| occupied >= threshold)
        .ok_or_else(|| "sort-first admission arithmetic overflow".to_string())
}

// This lower bound is rejection-only: collisions can never add distinct tuples.
pub(super) fn sample_occupancy(key_slices: &[&[i64]], n_rows: usize) -> Result<usize, String> {
    let sample_len = n_rows.min(8192);
    if sample_len == 0 {
        return Ok(0);
    }
    let (words, _) = bitmap_storage(sample_len).ok_or("sort-first sample bitmap overflow")?;
    let bits = words
        .checked_mul(64)
        .ok_or("sort-first sample bitmap overflow")?;
    #[cfg(test)]
    let bits =
        SAMPLE_COLLISION_BITS.with(|slot| slot.get().map_or(bits, |forced| forced.min(bits)));
    let mut sample = checked_route_vec::<u64>(words)?;
    #[cfg(test)]
    observe_scratch("sample_live", sample.capacity() * size_of::<u64>());
    sample.resize(words, 0);
    for idx in 0..sample_len {
        let row = sample_row(idx, n_rows, sample_len)?;
        let bit = (tuple_hash(key_slices, row) % bits as u64) as usize;
        sample[bit / 64] |= 1u64 << (bit % 64);
    }
    let occupied = sample.iter().map(|word| word.count_ones() as usize).sum();
    drop(sample);
    #[cfg(test)]
    observe_scratch("sample_released", 0);
    Ok(occupied)
}

pub(super) fn phases_fit(phases: &[usize; 5], cap: usize) -> bool {
    phases.iter().all(|bytes| *bytes <= cap)
}

pub(super) fn bitmap_storage(n: usize) -> Option<(usize, usize)> {
    let words = n.checked_mul(8)?.checked_add(63)?.checked_div(64)?;
    Some((words, words.checked_mul(size_of::<AtomicU64>())?))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SortFirstReducer {
    #[cfg(test)]
    SumF64,
    MaxF64,
    MaxI64,
    #[cfg(test)]
    MinF64,
    CountF64,
    CountI64,
    #[cfg(test)]
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SortFirstRoute {
    HashFirst,
    SortFirst,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SortFirstFallbackReason {
    UnsupportedReducer,
    UnsupportedKeyCount,
    InvalidInput,
    TooFewRows,
    ArithmeticOverflow,
    MemoryLimit,
    BitmapLimit,
    CardinalityLowerBound,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SortFirstRoutingDecision {
    pub route: SortFirstRoute,
    pub fallback_reason: Option<SortFirstFallbackReason>,
    pub cardinality_lower_bound: usize,
    pub bitmap_bytes: usize,
    pub modeled_peak_bytes: usize,
    pub normalized_input_bytes: usize,
}

impl SortFirstRoutingDecision {
    pub(super) fn reason_label(self) -> &'static str {
        match self.fallback_reason {
            None => "cardinality_lower_bound_admitted",
            Some(SortFirstFallbackReason::UnsupportedReducer) => "unsupported_reducer",
            Some(SortFirstFallbackReason::UnsupportedKeyCount) => "unsupported_key_count",
            Some(SortFirstFallbackReason::InvalidInput) => "invalid_input",
            Some(SortFirstFallbackReason::TooFewRows) => "too_few_rows",
            Some(SortFirstFallbackReason::ArithmeticOverflow) => "arithmetic_overflow",
            Some(SortFirstFallbackReason::MemoryLimit) => "memory_limit",
            Some(SortFirstFallbackReason::BitmapLimit) => "bitmap_limit",
            Some(SortFirstFallbackReason::CardinalityLowerBound) => {
                "cardinality_lower_bound_rejected"
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct SortFirstMemoryBudget {
    pub chunk_size: usize,
    pub chunks: usize,
    pub bitmap_words: usize,
    pub bitmap_bytes: usize,
    pub peak_bytes: usize,
    pub normalized_input_bytes: usize,
    pub phase_bytes: [usize; 5],
}

// The maximum is over disjoint lifetimes, not their sum. G=n and packed P=8n
// conservatively cover either radix layout and the final NumPy copy overlap.
pub(super) fn memory_budget(
    n: usize,
    k: usize,
) -> Result<SortFirstMemoryBudget, SortFirstFallbackReason> {
    let budget = calculate_memory_budget(n, k)?;
    if budget.bitmap_bytes > BITMAP_CAP {
        return Err(SortFirstFallbackReason::BitmapLimit);
    }
    if !phases_fit(&budget.phase_bytes, ROUTE_MEMORY_CAP) {
        return Err(SortFirstFallbackReason::MemoryLimit);
    }
    Ok(budget)
}

pub(super) fn calculate_memory_budget(
    n: usize,
    k: usize,
) -> Result<SortFirstMemoryBudget, SortFirstFallbackReason> {
    let overflow = SortFirstFallbackReason::ArithmeticOverflow;
    let calculate = || -> Option<SortFirstMemoryBudget> {
        let p = size_of::<usize>();
        let t = rayon::current_num_threads().max(1);
        let d = (n / t).max(1024);
        let c = n.checked_add(d.checked_sub(1)?)?.checked_div(d)?;
        let (words, b) = bitmap_storage(n)?;
        let u = n.checked_mul(p)?;
        let packed = n.checked_mul(8)?;
        let s = n.checked_add(1)?.checked_mul(p)?;
        let r = n.checked_mul(k.checked_mul(8)?.checked_add(8)?)?;
        let e = t
            .checked_add(c)?
            .checked_add(k)?
            .checked_mul(4096)?
            .checked_add(65536)?;
        let digit = checked_route_digit_workspace_bytes(n, t).ok()?;
        let sample = n.min(8192).checked_mul(p)?;
        let estimator = b.max(sample).checked_add(e)?; // Sample is dropped before bitmap allocation.
        let radix = u
            .checked_mul(2)?
            .checked_add(packed)?
            .checked_add(digit)?
            .checked_add(e)?;
        // Discovery retains packed keys but no counts/starts. Setup drops
        // packing before allocating either. On 64-bit hosts packed=8n <=
        // s=8(n+1), so discovery fits the existing setup envelope: no cap or
        // model inflation. Keep the max explicit for checked, portable sizing.
        let discovery = u.checked_add(n)?.checked_add(packed)?.checked_add(e)?;
        let setup = u
            .checked_add(n)?
            .checked_add(s)?
            .checked_add(c.checked_mul(2)?.checked_mul(p)?)?
            .checked_add(e)?;
        let boundaries = discovery.max(setup);
        let reduction = u.checked_add(s)?.checked_add(r)?.checked_add(e)?;
        let conversion = r.checked_mul(2)?.checked_add(e)?;
        let normalized = n.checked_mul(k.checked_add(1)?)?.checked_mul(8)?;
        Some(SortFirstMemoryBudget {
            chunk_size: d,
            chunks: c,
            bitmap_words: words,
            bitmap_bytes: b,
            peak_bytes: estimator
                .max(radix)
                .max(boundaries)
                .max(reduction)
                .max(conversion),
            normalized_input_bytes: normalized,
            phase_bytes: [estimator, radix, boundaries, reduction, conversion],
        })
    };
    calculate().ok_or(overflow)
}

#[cfg(test)]
pub(super) fn choose_sort_first_route(
    reducer: SortFirstReducer,
    key_slices: &[&[i64]],
    n_rows: usize,
) -> SortFirstRoutingDecision {
    // Existing diagnostic helper cannot return allocation errors; real dispatch uses
    // the fallible admission API and never retries hashing after allocation failure.
    try_choose_sort_first_route(reducer, key_slices, n_rows)
        .unwrap_or_else(|_| hash_first(SortFirstFallbackReason::MemoryLimit))
}

pub(super) fn try_choose_sort_first_route(
    reducer: SortFirstReducer,
    key_slices: &[&[i64]],
    n_rows: usize,
) -> Result<SortFirstRoutingDecision, String> {
    let reason = if !matches!(
        reducer,
        SortFirstReducer::MaxF64
            | SortFirstReducer::MaxI64
            | SortFirstReducer::CountF64
            | SortFirstReducer::CountI64
    ) {
        Some(SortFirstFallbackReason::UnsupportedReducer)
    } else if !(2..=MAX_KEY_COLUMNS).contains(&key_slices.len()) {
        Some(SortFirstFallbackReason::UnsupportedKeyCount)
    } else if key_slices.iter().any(|col| col.len() != n_rows) {
        Some(SortFirstFallbackReason::InvalidInput)
    } else if n_rows < MIN_SORT_FIRST_ROWS {
        Some(SortFirstFallbackReason::TooFewRows)
    } else {
        None
    };
    if let Some(reason) = reason {
        return Ok(hash_first(reason));
    }
    let budget = match calculate_memory_budget(n_rows, key_slices.len()) {
        Ok(budget) => budget,
        Err(reason) => return Ok(hash_first(reason)),
    };
    let refusal = if budget.bitmap_bytes > BITMAP_CAP {
        Some(SortFirstFallbackReason::BitmapLimit)
    } else if !phases_fit(&budget.phase_bytes, ROUTE_MEMORY_CAP) {
        Some(SortFirstFallbackReason::MemoryLimit)
    } else {
        None
    };
    if let Some(reason) = refusal {
        return Ok(SortFirstRoutingDecision {
            bitmap_bytes: budget.bitmap_bytes,
            modeled_peak_bytes: budget.peak_bytes,
            normalized_input_bytes: budget.normalized_input_bytes,
            ..hash_first(reason)
        });
    }
    // Rejection only: distributed tuple sampling avoids an O(n) bitmap pass on
    // ordinary/low-cardinality inputs. A good sample never establishes admission.
    let sample_len = n_rows.min(8192);
    let sample_distinct = sample_occupancy(key_slices, n_rows)?;
    #[cfg(test)]
    let sample_distinct = SAMPLE_POSITIVE.with(|slot| {
        if slot.get() {
            sample_len
        } else {
            sample_distinct
        }
    });
    if !reaches_threshold(sample_distinct, sample_len)? {
        return Ok(SortFirstRoutingDecision {
            bitmap_bytes: budget.bitmap_bytes,
            modeled_peak_bytes: budget.peak_bytes,
            normalized_input_bytes: budget.normalized_input_bytes,
            ..hash_first(SortFirstFallbackReason::CardinalityLowerBound)
        });
    }
    let mut bitmap = checked_route_vec::<AtomicU64>(budget.bitmap_words)?;
    #[cfg(test)]
    observe_scratch("bitmap_live", bitmap.capacity() * size_of::<AtomicU64>());
    bitmap.resize_with(budget.bitmap_words, || AtomicU64::new(0));
    let bits = budget
        .bitmap_words
        .checked_mul(64)
        .ok_or("sort-first bitmap overflow")?;
    #[cfg(test)]
    let bits = COLLISION_BITS.with(|slot| slot.get().map_or(bits, |forced| forced.min(bits)));
    (0..n_rows).into_par_iter().for_each(|row| {
        let hash = tuple_hash(key_slices, row);
        let bit = (hash % bits as u64) as usize;
        bitmap[bit / 64].fetch_or(1u64 << (bit % 64), Ordering::Relaxed);
    });
    let lower_bound: usize = bitmap
        .par_iter()
        .map(|word| word.load(Ordering::Relaxed).count_ones() as usize)
        .sum();
    drop(bitmap);
    #[cfg(test)]
    observe_scratch("bitmap_released", 0);
    let admitted = reaches_threshold(lower_bound, n_rows)?;
    Ok(SortFirstRoutingDecision {
        route: if admitted {
            SortFirstRoute::SortFirst
        } else {
            SortFirstRoute::HashFirst
        },
        fallback_reason: if admitted {
            None
        } else {
            Some(SortFirstFallbackReason::CardinalityLowerBound)
        },
        cardinality_lower_bound: lower_bound,
        bitmap_bytes: budget.bitmap_bytes,
        modeled_peak_bytes: budget.peak_bytes,
        normalized_input_bytes: budget.normalized_input_bytes,
    })
}

const fn hash_first(reason: SortFirstFallbackReason) -> SortFirstRoutingDecision {
    SortFirstRoutingDecision {
        route: SortFirstRoute::HashFirst,
        fallback_reason: Some(reason),
        cardinality_lower_bound: 0,
        bitmap_bytes: 0,
        modeled_peak_bytes: 0,
        normalized_input_bytes: 0,
    }
}
