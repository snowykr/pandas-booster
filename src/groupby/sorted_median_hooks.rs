#[cfg(test)]
use std::cell::Cell;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(test)]
thread_local! {
    static DIRECT_ENGINE_CALLS: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
static PARALLEL_SLICE_MEDIAN_CALLS: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
pub(super) fn reset_sorted_median_direct_call_count() {
    DIRECT_ENGINE_CALLS.with(|calls| calls.set(0));
}

#[cfg(test)]
pub(super) fn sorted_median_direct_call_count() -> usize {
    DIRECT_ENGINE_CALLS.with(Cell::get)
}

#[cfg(test)]
pub(super) fn record_sorted_median_direct_call() {
    DIRECT_ENGINE_CALLS.with(|calls| calls.set(calls.get() + 1));
}

#[cfg(not(test))]
pub(super) fn record_sorted_median_direct_call() {}

#[cfg(test)]
pub(super) fn reset_sorted_median_parallel_slice_median_count() {
    PARALLEL_SLICE_MEDIAN_CALLS.store(0, Ordering::SeqCst);
}

#[cfg(test)]
pub(super) fn sorted_median_parallel_slice_median_count() -> usize {
    PARALLEL_SLICE_MEDIAN_CALLS.load(Ordering::SeqCst)
}

#[cfg(test)]
pub(super) fn record_parallel_slice_median_call() {
    PARALLEL_SLICE_MEDIAN_CALLS.fetch_add(1, Ordering::SeqCst);
}

#[cfg(not(test))]
pub(super) fn record_parallel_slice_median_call() {}
