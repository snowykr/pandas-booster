use super::*;

/// Returns the minimum dataset size threshold for Rust acceleration.
#[pyfunction]
pub(crate) fn get_fallback_threshold() -> usize {
    FALLBACK_THRESHOLD
}

/// Returns the number of threads used by the Rayon thread pool.
#[pyfunction]
pub(crate) fn get_thread_count() -> usize {
    rayon::current_num_threads()
}

/// Marker certifying the ordered single-key float product ABI is available.
#[pyfunction]
pub(crate) fn has_ordered_single_key_float_prod_abi() -> bool {
    true
}
